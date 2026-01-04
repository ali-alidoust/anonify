import argparse
from collections import deque
import secrets

import av
from av.container.input import InputContainer
import cv2
import numpy as np
import shortuuid
from tqdm import tqdm
from ultralytics.models.yolo import YOLO


def anonify(input_path, output_path, mode="head", *, temporal=True):
    # YOLO11-n is the 2026 sweet spot for CPU speed
    model = YOLO("./models/yolo11n.pt")

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

    for frame in tqdm(container.decode(video=0), total=video_stream.frames):
        # Convert to OpenCV format
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        # 1. Read Frame and Create Mask
        mask = np.zeros((h, w), dtype=np.uint8)
        # Limit imgsz to file size to avoid excessive memory use
        imgsz = min(2048, max(h, w))
        # Higher imgsz improves detection of small/distant heads
        results = model.predict(img, classes=[0], conf=0.10, verbose=False, imgsz=imgsz)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input video")
    parser.add_argument("--mode", choices=["head", "body"], default="head")
    parser.add_argument(
        "--temporal", action="store_true", help="Enable Union Mask over 5 frames"
    )
    parser.add_argument("--output", default=None, help="Output video path")
    args = parser.parse_args()

    if args.output is None:
        # Generate a random output name using shortuuid
        random_id = shortuuid.ShortUUID().random(length=6)
        args.output = f"anon-{random_id}.mp4"

    anonify(args.input, args.output, args.mode, temporal=args.temporal)
