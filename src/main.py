from collections import deque
import heapq
import itertools
import multiprocessing
import os
from pathlib import Path
import secrets
import sys

import av
from av.container.input import InputContainer
import cv2
from gooey import Gooey, GooeyParser
import numpy as np
import openvino as ov
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction
import shortuuid
from tqdm import tqdm
from ultralytics.models.yolo import YOLO


# Store the original compile_model function
original_compile_model = ov.Core.compile_model


# Define a "hacked" version that always forces THROUGHPUT
def hacked_compile_model(self, model, device_name=None, config=None, *, weights=None):
    print("Applying OpenVINO PERFORMANCE_HINT=THROUGHPUT for CPU inference")
    if config is None:
        config = {}
    # Force the hint regardless of what Ultralytics wants
    config["PERFORMANCE_HINT"] = "THROUGHPUT"
    config["NUM_STREAMS"] = "AUTO"
    config["INFERENCE_NUM_THREADS"] = "8"
    return original_compile_model(
        self, model, device_name, config=config, weights=weights
    )


# Apply the patch
ov.Core.compile_model = hacked_compile_model


def load_model(*, model_path: str, threshold: float) -> UltralyticsDetectionModel:
    ov_model = YOLO(model_path, task="detect")
    return UltralyticsDetectionModel(
        model=ov_model,
        device="cpu",
        confidence_threshold=threshold,
        image_size=640,
    )


def producer_task(*, input_queue, input_path, num_consumers, limit_sem):
    # 1. Open Input with PyAV
    container = av.open(input_path, mode="r")
    if not isinstance(container, InputContainer):
        raise TypeError("Failed to open input video")

    if not container.streams.video:
        raise ValueError("Input video has no video stream")

    w = container.streams.video[0].width
    h = container.streams.video[0].height

    idx = 0
    for idx, frame in enumerate(container.decode(video=0)):
        # --- CRITICAL STEP ---
        # Block here if the system is full.
        # This prevents flooding the queues/RAM.
        limit_sem.acquire()

        # Convert to OpenCV format
        input_queue.put((idx, frame.to_ndarray(format="bgr24"), w, h))

    # Release semaphore slot after putting frame in queue
    limit_sem.release()

    container.close()

    for _ in range(num_consumers):
        input_queue.put(None)


def processor_task(
    *,
    input_queue: multiprocessing.Queue,
    processed_queue,
    model_path,
    mode,
    threshold,
):
    model = load_model(model_path=model_path, threshold=threshold)
    while True:
        item = input_queue.get()
        if item is None:
            processed_queue.put(None)
            break

        idx, frame, w, h = item
        results = get_sliced_prediction(
            frame,
            model,
            slice_height=960,
            slice_width=960,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.1,
            perform_standard_pred=True,
            verbose=0,
        )

        # 1. Read Frame and Create Mask
        mask = np.zeros((h, w), dtype=np.uint8)

        for result in results.object_prediction_list:
            if result.category.id != 0:
                continue  # Only process person class
            bx1, by1, bx2, by2 = map(int, result.bbox.to_xyxy())
            if mode == "head":
                cx = bx1 + (bx2 - bx1) // 2
                # Approximate heads using 80% width, 40% height
                hw, hh = int((bx2 - bx1) * 0.8), int((by2 - by1) * 0.4)
                x1, y1, x2, y2 = cx - hw // 2, by1, cx + hw // 2, by1 + hh
            else:
                x1, y1, x2, y2 = bx1, by1, bx2, by2

            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        processed_queue.put((idx, frame, mask))


def output_task(
    *, processed_queue, temporal, input_path, output_path, avg_rate, limit_sem
):
    window_size = 5 if temporal else 1
    frame_buffer = deque()
    mask_buffer = deque()

    heap = []
    idx_expected = 0

    container = av.open(input_path, mode="r")
    if not isinstance(container, InputContainer):
        raise TypeError("Failed to open input video")

    if not container.streams.video:
        raise ValueError("Input video has no video stream")

    w = container.streams.video[0].width
    h = container.streams.video[0].height

    audio_stream = container.streams.audio[0] if container.streams.audio else None

    progress = tqdm(total=container.streams.video[0].frames)

    progress.display()

    # Setup Output
    out_container = av.open(output_path, mode="w")
    out_video = out_container.add_stream("libx264", rate=avg_rate)
    out_video.width = w
    out_video.height = h
    out_video.pix_fmt = "yuv420p"
    out_video.time_base = container.streams.video[0].time_base

    # Get input video bitrate
    if input_bitrate := container.streams.video[0].bit_rate:
        out_video.bit_rate = input_bitrate

    # Setup audio stream if present
    out_audio = None
    if audio_stream:
        out_audio = out_container.add_stream(
            audio_stream.codec.name, rate=audio_stream.rate
        )
        out_audio.time_base = audio_stream.time_base

    tiebreaker = itertools.count()
    frame_pts = 0

    def filter_and_encode(frame_idx, pop=True):
        nonlocal frame_pts
        target_frame = frame_buffer[frame_idx]

        # 1. TEMPORAL UNION: Merge M(-2) through M(+2)
        union_mask = np.zeros((h, w), dtype=np.uint8)
        for m_idx in range(min(window_size, len(mask_buffer))):
            m = mask_buffer[m_idx]
            union_mask = cv2.bitwise_or(union_mask, m)

        if np.any(union_mask):
            # 2. TEMPORAL JITTER: Randomize kernel size to defeat AI unblurring
            # Kernel must be odd: 91, 93, 95... up to 121
            jitter = secrets.randbelow(16) * 2 + 91
            blurred_frame = cv2.GaussianBlur(target_frame, (jitter, jitter), 30)

            # 3. MASK STITCHING: Replace pixels
            mask_3ch = cv2.merge([union_mask] * 3)
            result = np.where(mask_3ch == 255, blurred_frame, target_frame)
        else:
            result = target_frame

        progress.update(1)

        # 4. Encode Video Frame
        new_frame = av.VideoFrame.from_ndarray(result, format="bgr24")
        new_frame.pts = frame_pts
        frame_pts += 1
        for packet in out_video.encode(new_frame):
            out_container.mux(packet)

        if pop:
            frame_buffer.popleft()
            mask_buffer.popleft()

    while True:
        item = processed_queue.get()
        if item is None:
            break

        idx, frame, mask = item
        heapq.heappush(heap, (idx, next(tiebreaker), frame, mask))
        # Process frames in order
        while heap and heap[0][0] == idx_expected:
            idx, _, frame, mask = heapq.heappop(heap)
            idx_expected += 1

            frame_buffer.append(frame)
            mask_buffer.append(mask)

            limit_sem.release()

            if idx < window_size // 2:
                while len(frame_buffer) > idx + 1:
                    # Not enough previous frames to form full window
                    filter_and_encode(0, pop=False)

            while len(frame_buffer) >= window_size:
                # Target the middle frame (M0)
                filter_and_encode(window_size // 2)

    # After producer is done, process remaining frames
    # Process remaining frames in buffer with available window
    for idx in range(len(frame_buffer)):
        filter_and_encode(idx, pop=False)

    # Flush video encoder before copying audio
    for packet in out_video.encode():
        out_container.mux(packet)

    # Copy Audio (Muxing)
    if audio_stream and out_audio:
        container.seek(0)  # Reset to beginning
        for packet in container.demux(audio_stream):
            if packet.dts is None:
                continue
            packet.stream = out_audio
            out_container.mux(packet)

    out_container.close()
    container.close()


def is_subprocess():
    return "GOOEY" in os.environ and os.environ["GOOEY"] == "1"


def print_with_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


NUM_CONSUMERS = 4


def anonify(*, input_path, output_path, model_path, mode, temporal, threshold):
    multiprocessing.freeze_support()

    # Detect input video frame rate
    container = av.open(input_path, mode="r")
    if not isinstance(container, InputContainer):
        raise TypeError("Failed to open input video")

    if not container.streams.video:
        raise ValueError("Input video has no video stream")

    input_fps = container.streams.video[0].average_rate
    container.close()

    limit_semaphore = multiprocessing.Semaphore(NUM_CONSUMERS * 5)

    input_queue = multiprocessing.Queue(maxsize=NUM_CONSUMERS * 5)
    processed_queue = multiprocessing.Queue(maxsize=NUM_CONSUMERS * 5)

    processors = []

    # Producer
    producer = multiprocessing.Process(
        target=producer_task,
        kwargs={
            "input_queue": input_queue,
            "input_path": input_path,
            "num_consumers": NUM_CONSUMERS,
            "limit_sem": limit_semaphore,
        },
    )

    for _ in range(NUM_CONSUMERS):
        processor = multiprocessing.Process(
            target=processor_task,
            kwargs={
                "input_queue": input_queue,
                "processed_queue": processed_queue,
                "model_path": model_path,
                "mode": mode,
                "threshold": threshold,
            },
        )
        processors.append(processor)

    # Output
    encoder = multiprocessing.Process(
        target=output_task,
        kwargs={
            "processed_queue": processed_queue,
            "temporal": temporal,
            "input_path": input_path,
            "output_path": output_path,
            "avg_rate": input_fps,
            "limit_sem": limit_semaphore,
        },
    )

    # Start Processes
    producer.start()
    encoder.start()
    for p in processors:
        p.start()

    # Wait for completion
    producer.join()
    for p in processors:
        p.join()
    encoder.join()


def main():
    parser = GooeyParser()
    parser.add_argument(
        "input",
        help="Path to video file",
        widget="FileChooser",
        metavar="Input Video",
        gooey_options={"full_width": True},
    )
    parser.add_argument(
        "--full-body",
        action="store_true",
        help="Blur full body instead of just head",
        widget="BlockCheckbox",
        metavar="Anonymize Full Body",
        gooey_options={},
    )
    parser.add_argument(
        "--no-temporal",
        action="store_true",
        help="Only anonymize current frame (no temporal smoothing)",
        widget="BlockCheckbox",
        metavar="Disable Temporal Masking",
        gooey_options={},
    )
    parser.add_argument(
        "--detection-rate",
        type=float,
        default=75,
        help="Sensitivity of detection (higher = more detections)",
        widget="Slider",
        metavar="Detection Rate (%)",
        gooey_options={
            "full_width": True,
            "min": 0,
            "max": 100,
            "step": 1,
        },
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Your video file will be saved here. If not specified, '.\\outputs' will be created and used.",
        widget="DirChooser",
        metavar="Output Directory",
        gooey_options={"full_width": True},
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = "./outputs"
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    # Generate a random output name using shortuuid
    random_id = shortuuid.ShortUUID().random(length=6)
    args.output = Path(args.output_dir) / f"anon-{random_id}.mp4"

    base_path = "."
    if is_subprocess() and getattr(sys, "frozen", False):
        base_path = sys._MEIPASS  # pyright: ignore[reportAttributeAccessIssue] # noqa: SLF001

    model_root = Path(base_path) / "models"

    anonify(
        input_path=args.input,
        output_path=args.output,
        model_path=str(model_root / "yolo11n_openvino_model"),
        mode="body" if args.full_body else "head",
        temporal=not args.no_temporal,
        threshold=1.0 - (args.detection_rate / 100.0),
    )


@Gooey(
    program_name="Anonify - Video Anonymizer",
    default_size=(600, 700),
    advanced=True,
    progress_regex=r"\bProcessing frame: (?P<current>\d+)/(?P<total>\d+)\b",
    progress_expr="current / total * 100",
    hide_progress_msg=True,
    timing_options={
        "show_time_remaining": True,
        "hide_time_remaining_on_complete": True,
    },
    show_restart_button=False,
)
def main_gui():
    main()


if __name__ == "__main__":
    try:
        import pyi_splash

        pyi_splash.close()
    except ImportError:
        pass

    if len(sys.argv) > 1 or "--ignore-gooey" in sys.argv:
        main()
    else:
        main_gui()
