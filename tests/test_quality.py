import pytest

from src.quality import check_quality


@pytest.mark.parametrize("value", [0.65, 0.7149321266968326, 1.0])
def test_quality_accepts_threshold_and_valid_scores(value):
    assert check_quality(value) == value


@pytest.mark.parametrize("value", [0.0, 0.6499, -1, 1.01, float("nan"), float("inf"), "bad"])
def test_quality_blocks_low_or_invalid_scores(value):
    with pytest.raises(ValueError):
        check_quality(value)
