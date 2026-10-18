import jiwer
import pytest

from punarvaani.eval.metrics import (
    LatencyMeter,
    cer,
    edit_counts,
    equal_error_rate,
    mos_summary,
    paired_bootstrap,
    viseme_error_rate,
    wer,
)


def test_edit_counts_breakdown():
    c = edit_counts("a b c d".split(), "a x c d e".split())
    assert (c.substitutions, c.deletions, c.insertions, c.ref_len) == (1, 0, 1, 4)


@pytest.mark.parametrize(
    "ref,hyp",
    [("मुझे पानी चाहिए", "मुझे बानी चाहिए"), ("दर्द हो रहा है", "दर्द रहा है जी"), ("धन्यवाद", "")],
)
def test_wer_matches_jiwer(ref, hyp):
    expected = jiwer.wer(ref, hyp) if hyp else 1.0
    assert abs(wer([ref], [hyp]) - expected) < 1e-9


def test_viseme_error_rate_ignores_lookalikes():
    refs, hyps = ["मुझे पानी चाहिए"], ["भुझे बानी चाहिए"]
    assert wer(refs, hyps) > 0.5 and cer(refs, hyps) > 0
    assert viseme_error_rate(refs, hyps) == 0.0


def test_paired_bootstrap_detects_clear_win():
    refs = ["मुझे पानी चाहिए"] * 40
    good = refs
    bad = ["मुझे बानी चाहिए"] * 40
    res = paired_bootstrap(refs, good, bad, n_resamples=200)
    assert res["rate_a"] == 0 and res["p_a_not_better"] == 0.0 and res["diff_ci95"][1] < 0


def test_mos_eer_latency():
    m = mos_summary([4, 5, 4, 3, 4, 5])
    assert m["ci95"][0] <= m["mos"] <= m["ci95"][1]
    eer, _ = equal_error_rate([0.9, 0.8, 0.7, 0.2, 0.1, 0.3], [1, 1, 1, 0, 0, 0])
    assert eer == 0.0
    meter = LatencyMeter()
    with meter.measure("x"):
        pass
    assert meter.summary()["x"]["n"] == 1
