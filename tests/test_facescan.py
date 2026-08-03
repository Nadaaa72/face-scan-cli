"""Tests for the pure logic in facescan.py.

These run on any machine, no video files or models needed, because the
functions under test do maths and string work only.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from facescan import frame_filename, is_small_face, clamp_box, build_report


def test_frame_filename_pads_to_six_digits():
    assert frame_filename("frames", 90).endswith("frame_000090.jpg")
    assert frame_filename("frames", 123456).endswith("frame_123456.jpg")


def test_frame_filename_uses_out_dir():
    path = frame_filename("myfolder", 0)
    assert path.startswith("myfolder")


def test_small_face_is_detected():
    assert is_small_face((10, 10, 60, 80)) is True      # 50px wide
    assert is_small_face((10, 10, 100, 80)) is False    # 90px wide


def test_small_face_respects_custom_threshold():
    assert is_small_face((0, 0, 70, 70), min_px=80) is True
    assert is_small_face((0, 0, 70, 70), min_px=50) is False


def test_clamp_box_leaves_a_box_inside_the_frame_alone():
    assert clamp_box((10, 20, 100, 200), 640, 480) == (10, 20, 100, 200)


def test_clamp_box_pulls_edges_back_into_the_frame():
    # a face cut off at the top left, and one running past the right edge
    assert clamp_box((-15, -8, 100, 200), 640, 480) == (0, 0, 100, 200)
    assert clamp_box((600, 400, 900, 700), 640, 480) == (600, 400, 640, 480)


def test_clamp_box_rounds_float_coordinates():
    # the detector returns floats, slicing needs ints
    assert clamp_box((10.7, 20.2, 100.9, 200.4), 640, 480) == (10, 20, 100, 200)


def test_report_maths():
    r = build_report(frame_count=4260, saved_count=142, total_faces=210,
                     best_frame="f.jpg", best_count=6, small_faces=34)
    assert r["avg_faces_per_frame"] == 1.48
    assert r["small_faces_pct"] == 16.2
    assert r["busiest_frame"] == "f.jpg"


def test_report_handles_empty_video():
    r = build_report(0, 0, 0, None, 0, 0)
    assert r["avg_faces_per_frame"] == 0.0
    assert r["small_faces_pct"] == 0.0


if __name__ == "__main__":
    # allow running without pytest: python tests/test_facescan.py
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
