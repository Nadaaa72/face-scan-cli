"""
facescan: scan a video, find every face in it, and build a small report.

Usage:
    python facescan.py video.mp4
    python facescan.py video.mp4 --skip 30 --out results
    python facescan.py video.mp4 --blur          # also write a face-blurred video
    python facescan.py video.mp4 --save-frames   # also keep the sampled frames
    python facescan.py video.mp4 --json report.json

What it does, in order:
    1. Walks through the video and keeps one frame out of every --skip frames.
    2. Runs a face detector (MTCNN) on each kept frame.
    3. Draws a green box around every face and saves the marked-up frames.
    4. Prints a report: total faces, average faces per frame, the single
       busiest frame, and how many faces were too small to be useful.
    5. With --blur, writes a copy of the video with every face blurred out,
       which is handy any time footage has to be shared without exposing
       people's identities.
"""
import argparse
import json
import os
import sys


# ---------------------------------------------------------------------------
# Small pure helpers. These are kept free of OpenCV so the unit tests can run
# on any machine, even one without the vision libraries installed.
# ---------------------------------------------------------------------------

def frame_filename(out_dir, frame_number):
    """frames/frame_000090.jpg style names, so files sort in video order."""
    return os.path.join(out_dir, f"frame_{frame_number:06d}.jpg")


def is_small_face(box, min_px=65):
    """A face narrower than min_px pixels is usually too blurry to work with."""
    x1, _, x2, _ = box
    return (x2 - x1) < min_px


def clamp_box(box, width, height):
    """Keep a detector box inside the frame.

    The detector occasionally returns coordinates a few pixels outside the
    image when a face is cut off at the edge. Slicing with those numbers gives
    an empty or lopsided crop, so every box is clamped before it is used.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


def build_report(frame_count, saved_count, total_faces, best_frame,
                 best_count, small_faces):
    """Turn raw counters into the printable summary. Pure function on purpose,
    so the maths is testable without running any video through it."""
    avg = (total_faces / saved_count) if saved_count else 0.0
    small_pct = (100.0 * small_faces / total_faces) if total_faces else 0.0
    return {
        "frames_in_video": frame_count,
        "frames_scanned": saved_count,
        "total_faces": total_faces,
        "avg_faces_per_frame": round(avg, 2),
        "busiest_frame": best_frame,
        "busiest_frame_faces": best_count,
        "small_faces": small_faces,
        "small_faces_pct": round(small_pct, 1),
    }


# ---------------------------------------------------------------------------
# The video pipeline. Heavy imports live inside the functions so that simply
# importing this file stays cheap and the tests above never need them.
# ---------------------------------------------------------------------------

def _load_detector():
    """Build the face detector. keep_all means every face, not just the biggest."""
    from facenet_pytorch import MTCNN
    return MTCNN(keep_all=True)


def _open_video(path):
    """Open a video and fail loudly if OpenCV cannot decode it.

    Without this check a missing codec looks exactly like an empty video: the
    first cap.read() returns False and the tool cheerfully reports zero frames.
    """
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"could not open video: {path}")
    return cap


def scan_video(video_path, out_dir, skip=30, save_frames=False, min_px=65):
    """Sample the video, detect faces, save marked-up frames, return counters."""
    import cv2

    detector = _load_detector()
    boxes_dir = os.path.join(out_dir, "faces")
    os.makedirs(boxes_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "frames")
    if save_frames:
        os.makedirs(frames_dir, exist_ok=True)

    cap = _open_video(video_path)
    frame_count = 0
    saved_count = 0
    total_faces = 0
    small_faces = 0
    best_frame = None
    best_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break                      # (False, None) means the video is finished
        if frame_count % skip == 0:
            saved_count += 1
            if save_frames:
                cv2.imwrite(frame_filename(frames_dir, frame_count), frame)

            # The detector expects RGB but OpenCV loads frames as BGR.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = detector.detect(rgb)

            faces_here = 0
            if boxes is not None:
                height, width = frame.shape[:2]
                for box in boxes:
                    x1, y1, x2, y2 = clamp_box(box, width, height)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    faces_here += 1
                    total_faces += 1
                    if is_small_face((x1, y1, x2, y2), min_px):
                        small_faces += 1
            if faces_here > best_count:
                best_count = faces_here
                best_frame = frame_filename(boxes_dir, frame_count)
            if faces_here:
                cv2.imwrite(frame_filename(boxes_dir, frame_count), frame)
        frame_count += 1

    cap.release()
    return build_report(frame_count, saved_count, total_faces,
                        best_frame, best_count, small_faces)


def blur_video(video_path, out_path):
    """Write a copy of the video with every detected face blurred.
    Every frame is processed here, not just the sampled ones, because a blur
    that flickers off for 29 of every 30 frames would not protect anyone."""
    import cv2

    detector = _load_detector()
    cap = _open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        cap.release()
        sys.exit(f"could not open {out_path} for writing (missing mp4v codec?)")

    done = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = detector.detect(rgb)
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = clamp_box(box, width, height)
                face = frame[y1:y2, x1:x2]
                if face.size:
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(face, (51, 51), 30)
        writer.write(frame)
        done += 1
        if done % 100 == 0:
            print(f"  blurred {done} frames...")
    cap.release()
    writer.release()
    return done


def print_report(report, min_face):
    print("\n--- report ---")
    print(f"frames in video      : {report['frames_in_video']}")
    print(f"frames scanned       : {report['frames_scanned']}")
    print(f"faces found          : {report['total_faces']}")
    print(f"average faces/frame  : {report['avg_faces_per_frame']}")
    print(f"busiest frame        : {report['busiest_frame']} "
          f"({report['busiest_frame_faces']} faces)")
    print(f"small faces (<{min_face}px)  : {report['small_faces']} "
          f"({report['small_faces_pct']}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Scan a video for faces and build a small report.")
    parser.add_argument("video", help="path to the video file")
    parser.add_argument("--skip", type=int, default=30,
                        help="keep 1 frame out of every N (default 30)")
    parser.add_argument("--out", default="results",
                        help="output folder (default: results)")
    parser.add_argument("--min-face", type=int, default=65,
                        help="faces narrower than this count as 'small'")
    parser.add_argument("--save-frames", action="store_true",
                        help="also keep the raw sampled frames")
    parser.add_argument("--blur", action="store_true",
                        help="also write a face-blurred copy of the video")
    parser.add_argument("--json", metavar="FILE",
                        help="also write the report as JSON to FILE")
    args = parser.parse_args()

    if args.skip < 1:
        sys.exit("--skip must be 1 or more")
    if not os.path.isfile(args.video):
        sys.exit(f"video not found: {args.video}")

    print(f"scanning {args.video} (1 frame in every {args.skip})...")
    report = scan_video(args.video, args.out, skip=args.skip,
                        save_frames=args.save_frames, min_px=args.min_face)
    print_report(report, args.min_face)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nreport written to {args.json}")

    if args.blur:
        out_path = os.path.join(args.out, "blurred.mp4")
        print(f"\nwriting blurred copy to {out_path}...")
        frames = blur_video(args.video, out_path)
        print(f"done, {frames} frames blurred.")


if __name__ == "__main__":
    main()
