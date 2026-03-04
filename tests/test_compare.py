import pytest
from wscribe.compare import (
    parse_timestamp,
    normalize_text,
    align_by_time_window,
    score_word_group,
    compare_transcriptions,
)

import json
import os
import tempfile
from click.testing import CliRunner
from wscribe.cli.main import cli


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


class TestScoreWordGroup:
    def test_all_agree(self):
        ref = _word(1.0, "hello")
        others = [_word(1.0, "hello"), _word(1.0, "hello")]
        score, text = score_word_group(ref, others)
        assert score == pytest.approx(1.0)
        assert text == "hello"

    def test_all_disagree(self):
        ref = _word(1.0, "hello")
        others = [_word(1.0, "world"), _word(1.0, "foo")]
        score, text = score_word_group(ref, others)
        # ref + 0 matching others = 1 of 3
        assert score == pytest.approx(1 / 3, abs=0.01)

    def test_majority_wins(self):
        ref = _word(1.0, "hello")
        others = [_word(1.0, "hello"), _word(1.0, "world")]
        score, text = score_word_group(ref, others)
        assert score == pytest.approx(2 / 3, abs=0.01)
        assert text == "hello"

    def test_none_match_scores_zero(self):
        ref = _word(1.0, "hello")
        others = [None, None]
        score, text = score_word_group(ref, others)
        assert score == pytest.approx(0.0)
        assert text == "hello"  # falls back to reference

    def test_punctuation_ignored_in_comparison(self):
        ref = _word(1.0, "hello,")
        others = [_word(1.0, "hello")]
        score, text = score_word_group(ref, others)
        assert score == pytest.approx(1.0)

    def test_case_ignored_in_comparison(self):
        ref = _word(1.0, "Hello")
        others = [_word(1.0, "hello")]
        score, text = score_word_group(ref, others)
        assert score == pytest.approx(1.0)


def _seg(start: float, end: float, text: str, words: list[dict]) -> dict:
    return {"start": start, "end": end, "text": text, "score": 1.0, "words": words}


class TestCompareTranscriptions:
    def _two_identical(self):
        w = [_word(1.0, "hello"), _word(2.0, "world")]
        seg = [_seg(0.0, 3.0, "hello world", w)]
        return [seg, seg]

    def test_identical_inputs_score_one(self):
        result = compare_transcriptions(self._two_identical(), tolerance=0.2)
        for seg in result:
            assert seg["score"] == pytest.approx(1.0)
            for w in seg["words"]:
                assert w["score"] == pytest.approx(1.0)

    def test_output_schema(self):
        result = compare_transcriptions(self._two_identical(), tolerance=0.2)
        assert isinstance(result, list)
        seg = result[0]
        assert set(seg.keys()) == {"text", "start", "end", "score", "words"}
        word = seg["words"][0]
        assert set(word.keys()) == {"text", "start", "end", "score"}

    def test_disjoint_words_score_zero(self):
        w_a = [_word(1.0, "hello")]
        w_b = [_word(9.0, "world")]
        seg_a = [_seg(0.0, 2.0, "hello", w_a)]
        seg_b = [_seg(8.0, 10.0, "world", w_b)]
        result = compare_transcriptions([seg_a, seg_b], tolerance=0.2)
        for seg in result:
            for w in seg["words"]:
                assert w["score"] == pytest.approx(0.0)

    def test_segment_score_is_mean_of_word_scores(self):
        w = [_word(1.0, "hello"), _word(2.0, "world")]
        # second transcription agrees on first word, differs on second
        w2 = [_word(1.0, "hello"), _word(2.0, "other")]
        seg = [_seg(0.0, 3.0, "hello world", w)]
        seg2 = [_seg(0.0, 3.0, "hello other", w2)]
        result = compare_transcriptions([seg, seg2], tolerance=0.2)
        word_scores = [w["score"] for w in result[0]["words"]]
        expected_seg_score = round(sum(word_scores) / len(word_scores), 4)
        assert result[0]["score"] == pytest.approx(expected_seg_score, abs=0.01)


SAMPLE_JSON = os.path.join(
    os.environ.get("PROJECT_ROOT", ""),
    "examples", "output", "sample.json",
)


class TestCompareCLI:
    def test_requires_two_files(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", SAMPLE_JSON, "--output", "/tmp/x.json"])
        assert result.exit_code != 0

    def test_smoke(self):
        if not os.path.exists(SAMPLE_JSON):
            pytest.skip("sample.json not available")
        import logging
        runner = CliRunner(mix_stderr=False)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name
        try:
            # Suppress Python logging during invoke to avoid pytest log_cli /
            # Click CliRunner stream conflict on Python 3.14
            logging.disable(logging.WARNING)
            try:
                result = runner.invoke(
                    cli,
                    ["compare", SAMPLE_JSON, SAMPLE_JSON, "--output", out_path],
                )
            finally:
                logging.disable(logging.NOTSET)
            assert result.exit_code == 0, result.output
            data = json.loads(open(out_path).read())
            assert isinstance(data, list)
            assert len(data) > 0
            seg = data[0]
            assert {"text", "start", "end", "score", "words"} <= set(seg.keys())
            # timestamp strings
            assert isinstance(seg["start"], str)
            assert ":" in seg["start"]
        finally:
            os.unlink(out_path)
