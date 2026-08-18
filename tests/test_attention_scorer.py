import pytest

from driver_state_detection.attention_scorer import AttentionScorer


def make_scorer(start=0.0):
    return AttentionScorer(t_now=start, ear_thresh=0.15, gaze_thresh=0.2)


def test_missing_face_time_does_not_trigger_sleep_alert():
    scorer = make_scorer()
    scorer.eval_scores(0.5, 0.3, 0.0, 0.0, 0.0, 0.0)
    scorer.mark_face_missing(100.0)

    asleep, _, _ = scorer.eval_scores(100.1, 0.1, 0.0, 0.0, 0.0, 0.0)

    assert not asleep
    assert scorer.closure_time == pytest.approx(0.1)


def test_perclos_does_not_alert_before_window_is_full():
    scorer = make_scorer()

    tired, score = scorer.get_rolling_PERCLOS(0.0, 0.1)
    assert not tired
    assert score == 0.0
    assert not scorer.perclos_ready

    tired, score = scorer.get_rolling_PERCLOS(30.0, 0.1)
    assert not tired
    assert score == 1.0
    assert not scorer.perclos_ready

    tired, score = scorer.get_rolling_PERCLOS(60.0, 0.1)
    assert tired
    assert score == 1.0
    assert scorer.perclos_ready


def test_perclos_excludes_missing_face_intervals():
    scorer = make_scorer()
    scorer.get_rolling_PERCLOS(0.0, 0.3)
    scorer.mark_face_missing(10.0)
    scorer.mark_face_missing(60.0)

    tired, score = scorer.get_rolling_PERCLOS(61.0, 0.1)

    assert not tired
    assert score == 0.0
    assert not scorer.perclos_ready


def test_short_missing_gap_keeps_mature_perclos_ready_until_coverage_expires():
    scorer = make_scorer()
    scorer.get_rolling_PERCLOS(0.0, 0.3)
    scorer.get_rolling_PERCLOS(60.0, 0.3)
    assert scorer.perclos_ready

    scorer.mark_face_missing(61.0)
    assert scorer.perclos_ready

    scorer.mark_face_missing(75.0)
    assert not scorer.perclos_ready


def test_perclos_excludes_unavailable_ear_intervals():
    scorer = make_scorer()
    scorer.get_rolling_PERCLOS(0.0, 0.1)
    scorer.get_rolling_PERCLOS(10.0, None)

    tired, score = scorer.get_rolling_PERCLOS(60.0, 0.3)

    assert not tired
    assert score == 1.0


def test_fixed_perclos_handles_zero_fps():
    assert make_scorer().get_PERCLOS(1.0, 0.0, 0.1) == (False, 0.0)
