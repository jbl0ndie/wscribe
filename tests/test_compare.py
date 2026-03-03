import pytest
from wscribe.compare import parse_timestamp, normalize_text, align_by_time_window


class TestParseTimestamp:
    def test_zero(self):
        assert parse_timestamp("00:00:00.000") == 0.0

    def test_milliseconds(self):
        assert parse_timestamp("00:00:07.260") == pytest.approx(7.260, abs=1e-6)

    def test_minutes(self):
        assert parse_timestamp("00:01:00.000") == pytest.approx(60.0, abs=1e-6)

    def test_hours(self):
        assert parse_timestamp("01:00:00.000") == pytest.approx(3600.0, abs=1e-6)

    def test_full(self):
        assert parse_timestamp("01:23:45.678") == pytest.approx(
            3600 + 23 * 60 + 45 + 0.678, abs=1e-6
        )

    def test_float_passthrough(self):
        assert parse_timestamp(7.26) == pytest.approx(7.26, abs=1e-6)


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("Hello") == "hello"

    def test_strips_punctuation(self):
        assert normalize_text("hello,") == "hello"
        assert normalize_text("it's") == "its"

    def test_strips_surrounding_whitespace(self):
        assert normalize_text("  hello  ") == "hello"

    def test_empty(self):
        assert normalize_text("") == ""


def _word(start: float, text: str) -> dict:
    return {"start": start, "end": start + 0.5, "text": text, "score": 1.0}


class TestAlignByTimeWindow:
    def test_exact_match(self):
        ref = [_word(1.0, "hello")]
        other = [[_word(1.0, "hello")]]
        result = align_by_time_window(ref, other, tolerance=0.2)
        assert len(result) == 1
        assert result[0][0]["text"] == "hello"

    def test_within_tolerance(self):
        ref = [_word(1.0, "hello")]
        other = [[_word(1.15, "hello")]]
        result = align_by_time_window(ref, other, tolerance=0.2)
        assert result[0][0] is not None

    def test_outside_tolerance(self):
        ref = [_word(1.0, "hello")]
        other = [[_word(1.21, "hello")]]
        result = align_by_time_window(ref, other, tolerance=0.2)
        assert result[0][0] is None

    def test_exactly_at_tolerance_boundary(self):
        ref = [_word(1.0, "hello")]
        other = [[_word(1.2, "hello")]]
        result = align_by_time_window(ref, other, tolerance=0.2)
        assert result[0][0] is not None

    def test_no_match_in_other(self):
        ref = [_word(1.0, "hello")]
        other = [[_word(5.0, "world")]]
        result = align_by_time_window(ref, other, tolerance=0.2)
        assert result[0][0] is None

    def test_multiple_transcriptions(self):
        ref = [_word(1.0, "hello")]
        other = [[_word(1.0, "hello")], [_word(9.0, "world")]]
        result = align_by_time_window(ref, other, tolerance=0.2)
        assert result[0][0] is not None
        assert result[0][1] is None

    def test_multiple_reference_words(self):
        ref = [_word(1.0, "hello"), _word(2.0, "world")]
        other = [[_word(1.0, "hello"), _word(2.0, "world")]]
        result = align_by_time_window(ref, other, tolerance=0.2)
        assert result[0][0]["text"] == "hello"
        assert result[1][0]["text"] == "world"
