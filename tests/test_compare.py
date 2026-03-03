import pytest
from wscribe.compare import parse_timestamp, normalize_text


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
