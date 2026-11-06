import pytest
from pinj import metrics


def test_micro_average_handles_unequal_categories():
    value = metrics.aggregate_asr_counts({"a": (1, 2), "b": (1, 8)})
    assert value == 0.2


def test_quality_must_be_measured_range():
    assert metrics.pds(0.1, 0.1, 10.0, quality=0.9) > 0
    with pytest.raises(ValueError):
        metrics.pds(0.1, 0.1, 10.0, quality=1.2)


def test_wilson_zero_successes():
    lo, hi = metrics.wilson_ci(0, 300)
    assert lo == 0.0 and hi < 0.02
