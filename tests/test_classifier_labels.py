import pytest
from pinj.proxies.classifier_guard import positive_probability


def test_prompt_guard_2_malicious_label():
    scores = [
        {"label": "BENIGN", "score": 0.05},
        {"label": "MALICIOUS", "score": 0.95},
    ]
    assert positive_probability(scores) == pytest.approx(0.95)


def test_unknown_labels_fail_closed_for_configuration():
    with pytest.raises(ValueError):
        positive_probability([{"label": "CAT", "score": 1.0}])
