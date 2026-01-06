from collections import deque
import math
import os
from pathlib import Path
import secrets
import sys

import av
from av.container.input import InputContainer
import cv2
from gooey import Gooey, GooeyParser
import numpy as np
import shortuuid
from tqdm import tqdm
from ultralytics.models.yolo import YOLO


def is_subprocess():
    return "GOOEY" in os.environ and os.environ["GOOEY"] == "1"


def print_with_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def anonify(input_path, output_path, *, mode="head", temporal=True, threshold=0.25):
    model_path = "./models/yolo11n.pt"
    # If we are using pyinstaller, adjust the model path
    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS  # pyright: ignore[reportAttributeAccessIssue] # noqa: SLF001
        model_path = Path(base_path) / "models" / "yolo11n.pt"

    # YOLO11-n is the 2026 sweet spot for CPU speed
    model = YOLO(model_path)

    # 1. Open Input with PyAV
    container = av.open(input_path, mode="r")
    if not isinstance(container, InputContainer):
        raise TypeError("Failed to open input video")
    video_stream = container.streams.video[0]
    audio_stream = container.streams.audio[0] if container.streams.audio else None

    # 2. Setup Output
    out_container = av.open(output_path, mode="w")
    out_video = out_container.add_stream("libx264", rate=video_stream.average_rate)
    out_video.width = video_stream.width
    out_video.height = video_stream.height
    out_video.pix_fmt = "yuv420p"

    out_audio = None
    if audio_stream:
        out_audio = out_container.add_stream(
            audio_stream.codec.name, rate=audio_stream.rate
        )

    # Buffer for Temporal Union (Window of 5 frames: -2, -1, 0, +1, +2)
    window_size = 5 if temporal else 1
    frame_buffer = deque(maxlen=window_size)
    mask_buffer = deque(maxlen=window_size)

    print(f"Starting Anonify | Mode: {mode} | Temporal: {temporal}")

    progress = tqdm(total=video_stream.frames, delay=math.nextafter(0, 1))
    if is_subprocess():
        progress.bar_format = "Processing frame: {n_fmt}/{total_fmt}"
        progress.sp = print_with_flush
    progress.display()
    for frame in container.decode(video=0):
        progress.update(1)
        # Convert to OpenCV format
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        # 1. Read Frame and Create Mask
        mask = np.zeros((h, w), dtype=np.uint8)
        # Limit imgsz to file size to avoid excessive memory use
        imgsz = min(2048, max(h, w))
        # Higher imgsz improves detection of small/distant heads
        results = model.predict(
            img, classes=[0], conf=threshold, verbose=False, imgsz=imgsz
        )

        if len(results) != 1:
            raise ValueError("Expected a single result per frame")
        if results[0].boxes is None:
            raise ValueError("No boxes detected")

        for box in results[0].boxes:
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            if mode == "head":
                cx = bx1 + (bx2 - bx1) // 2
                # Approximate heads using 80% width, 40% height
                hw, hh = int((bx2 - bx1) * 0.8), int((by2 - by1) * 0.4)
                x1, y1, x2, y2 = cx - hw // 2, by1, cx + hw // 2, by1 + hh
            else:
                x1, y1, x2, y2 = bx1, by1, bx2, by2

            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        frame_buffer.append(img)
        mask_buffer.append(mask)

        # 2. Process and Write the middle frame of the buffer
        if len(frame_buffer) == window_size:
            # Target the middle frame (M0)
            target_frame = frame_buffer[len(frame_buffer) // 2]

            # 1. TEMPORAL UNION: Merge M(-2) through M(+2)
            union_mask = np.zeros((h, w), dtype=np.uint8)
            for m in mask_buffer:
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

            # 4. Encode Video Frame
            new_frame = av.VideoFrame.from_ndarray(result, format="bgr24")
            for packet in out_video.encode(new_frame):
                out_container.mux(packet)
    progress.sp = lambda _: None
    progress.close()

    # 4. Copy Audio (Muxing)
    if audio_stream and out_audio:
        container.seek(0)  # Reset to beginning
        for packet in container.demux(audio_stream):
            if packet.dts is None:
                continue
            packet.stream = out_audio
            out_container.mux(packet)

    # 5. Flush encoders
    for packet in out_video.encode():
        out_container.mux(packet)
    out_container.close()
    container.close()

    print("Done.")


def main():
    parser = GooeyParser()
    parser.add_argument("input", help="Input video", widget="FileChooser")
    parser.add_argument(
        "--full-body",
        action="store_true",
        help="Anonymize full body instead of just head",
        widget="BlockCheckbox",
    )
    parser.add_argument(
        "--no-temporal",
        action="store_true",
        help="Disable Union Masking (over 5 frames)",
        widget="BlockCheckbox",
    )
    parser.add_argument(
        "--detection-rate",
        type=float,
        default=75,
        help="Detection rate. Higher values increase detection sensitivity but may slow processing. (0-100)",
        widget="Slider",
    )
    parser.add_argument(
        "--output-dir", default=None, help="Output video directory", widget="DirChooser"
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = "./outputs"
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    # Generate a random output name using shortuuid
    random_id = shortuuid.ShortUUID().random(length=6)
    args.output = Path(args.output_dir) / f"anon-{random_id}.mp4"

    anonify(
        args.input,
        args.output,
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
    show_restart=False,
)
def main_gui():
    main()


if __name__ == "__main__":
    if len(sys.argv) > 1 or "--ignore-gooey" in sys.argv:
        main()
    else:
        main_gui()
