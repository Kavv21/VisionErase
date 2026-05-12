"""Unit tests for pipeline/segmenter/segment_splitter.py — compute_segments().

extract_segment() is not tested here because it performs real S3 and ffmpeg I/O;
those belong in integration tests.
"""
from __future__ import annotations

import pytest

from pipeline.segmenter.segment_splitter import VideoSegment, compute_segments


# ── Helpers ───────────────────────────────────────────────────────────────────

def _total_logical_frames(segments: list[VideoSegment]) -> int:
    return sum(s.end_frame - s.start_frame for s in segments)


# ── compute_segments: basic contract ─────────────────────────────────────────

@pytest.mark.unit
class TestComputeSegmentsCount:
    def test_returns_correct_number_of_segments(self):
        """compute_segments must return exactly num_workers segments when frames allow."""
        segs = compute_segments(total_frames=300, fps=30.0, num_workers=4, overlap_frames=10)
        assert len(segs) == 4

    def test_single_worker_returns_one_segment(self):
        """With num_workers=1 the entire video is one segment."""
        segs = compute_segments(total_frames=100, fps=25.0, num_workers=1, overlap_frames=5)
        assert len(segs) == 1

    def test_clamped_when_more_workers_than_frames(self):
        """If num_workers > total_frames, segments are clamped to total_frames."""
        segs = compute_segments(total_frames=3, fps=30.0, num_workers=10, overlap_frames=0)
        assert len(segs) == 3

    def test_segment_indices_are_sequential(self):
        """segment_index must be 0, 1, 2, … with no gaps."""
        segs = compute_segments(total_frames=90, fps=30.0, num_workers=3, overlap_frames=5)
        assert [s.segment_index for s in segs] == [0, 1, 2]


# ── compute_segments: full frame coverage ─────────────────────────────────────

@pytest.mark.unit
class TestComputeSegmentsCoverage:
    def test_segments_cover_all_frames_no_gaps(self):
        """The union of logical [start_frame, end_frame) ranges must cover [0, total_frames)."""
        total = 300
        segs = compute_segments(total_frames=total, fps=30.0, num_workers=4, overlap_frames=15)
        assert segs[0].start_frame == 0
        assert segs[-1].end_frame == total
        for a, b in zip(segs, segs[1:]):
            assert a.end_frame == b.start_frame, "Gap detected between consecutive segments"

    def test_logical_frame_sum_equals_total(self):
        """Sum of each segment's logical length must equal total_frames."""
        total = 301   # odd number to exercise remainder distribution
        segs = compute_segments(total_frames=total, fps=24.0, num_workers=5, overlap_frames=10)
        assert _total_logical_frames(segs) == total

    def test_remainder_frames_distributed_to_first_segments(self):
        """When total_frames is not divisible, extra frames go to the first segments."""
        segs = compute_segments(total_frames=10, fps=25.0, num_workers=3, overlap_frames=0)
        # 10 / 3 = 3 remainder 1  → segment 0 has 4 frames, segments 1-2 have 3
        assert segs[0].end_frame - segs[0].start_frame == 4
        assert segs[1].end_frame - segs[1].start_frame == 3
        assert segs[2].end_frame - segs[2].start_frame == 3


# ── compute_segments: overlap boundaries ──────────────────────────────────────

@pytest.mark.unit
class TestComputeSegmentsOverlap:
    def test_first_segment_overlap_start_is_zero(self):
        """The first segment has no preceding segment, so overlap_start must be 0."""
        segs = compute_segments(total_frames=300, fps=30.0, num_workers=4, overlap_frames=50)
        assert segs[0].overlap_start == 0

    def test_last_segment_overlap_end_equals_total_frames(self):
        """The last segment has no following segment, so overlap_end must be total_frames."""
        total = 300
        segs = compute_segments(total_frames=total, fps=30.0, num_workers=4, overlap_frames=50)
        assert segs[-1].overlap_end == total

    def test_middle_segment_has_leading_and_trailing_overlap(self):
        """A middle segment must extend into both neighbours by overlap_frames."""
        overlap = 15
        segs = compute_segments(total_frames=300, fps=30.0, num_workers=4, overlap_frames=overlap)
        mid = segs[1]
        assert mid.overlap_start == mid.start_frame - overlap
        assert mid.overlap_end == mid.end_frame + overlap

    def test_overlap_clipped_at_zero(self):
        """overlap_start must never be negative even with a large overlap value."""
        segs = compute_segments(total_frames=100, fps=30.0, num_workers=4, overlap_frames=999)
        for s in segs:
            assert s.overlap_start >= 0

    def test_overlap_clipped_at_total_frames(self):
        """overlap_end must never exceed total_frames."""
        total = 100
        segs = compute_segments(total_frames=total, fps=30.0, num_workers=4, overlap_frames=999)
        for s in segs:
            assert s.overlap_end <= total

    def test_zero_overlap_produces_no_extension(self):
        """With overlap_frames=0, overlap_start==start_frame and overlap_end==end_frame."""
        segs = compute_segments(total_frames=90, fps=30.0, num_workers=3, overlap_frames=0)
        for s in segs:
            assert s.overlap_start == s.start_frame
            assert s.overlap_end == s.end_frame


# ── compute_segments: frame range correctness ────────────────────────────────

@pytest.mark.unit
class TestComputeSegmentsFrameRanges:
    def test_frame_ranges_for_two_workers(self):
        """Spot-check exact frame ranges for a simple 2-worker split."""
        segs = compute_segments(total_frames=100, fps=25.0, num_workers=2, overlap_frames=5)
        assert segs[0].start_frame == 0
        assert segs[0].end_frame == 50
        assert segs[1].start_frame == 50
        assert segs[1].end_frame == 100

    def test_fps_stored_on_segment(self):
        """fps must be forwarded unchanged to each VideoSegment."""
        segs = compute_segments(total_frames=60, fps=29.97, num_workers=2, overlap_frames=0)
        for s in segs:
            assert s.fps == 29.97

    def test_s3_key_is_empty_string_before_extract(self):
        """s3_key is unpopulated until extract_segment() runs."""
        segs = compute_segments(total_frames=60, fps=30.0, num_workers=2, overlap_frames=0)
        for s in segs:
            assert s.s3_key == ""

    def test_frame_ranges_for_three_workers_equal_split(self):
        """Verify exact ranges when total_frames divides evenly into num_workers."""
        segs = compute_segments(total_frames=90, fps=30.0, num_workers=3, overlap_frames=0)
        assert (segs[0].start_frame, segs[0].end_frame) == (0, 30)
        assert (segs[1].start_frame, segs[1].end_frame) == (30, 60)
        assert (segs[2].start_frame, segs[2].end_frame) == (60, 90)


# ── compute_segments: validation ─────────────────────────────────────────────

@pytest.mark.unit
class TestComputeSegmentsValidation:
    def test_raises_on_zero_workers(self):
        """num_workers=0 is nonsensical and must raise ValueError."""
        with pytest.raises(ValueError, match="num_workers"):
            compute_segments(total_frames=100, fps=30.0, num_workers=0, overlap_frames=0)

    def test_raises_on_zero_frames(self):
        """total_frames=0 is nonsensical and must raise ValueError."""
        with pytest.raises(ValueError, match="total_frames"):
            compute_segments(total_frames=0, fps=30.0, num_workers=2, overlap_frames=0)
