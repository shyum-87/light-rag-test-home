"""tests/test_stats.py"""
from src.stats import calc_sigma_outliers, chi_square_test


def test_sigma_outliers_detects_anomaly():
    """평균 + 3sigma를 초과하는 값을 특이치로 선별한다."""
    data = {"DA-01": 2.0, "DA-02": 2.1, "DA-03": 12.5, "DA-04": 1.9, "DA-05": 2.2}
    outliers = calc_sigma_outliers(data, sigma_threshold=3.0)
    assert "DA-03" in outliers
    assert "DA-01" not in outliers
    assert outliers["DA-03"]["sigma"] > 3.0


def test_sigma_outliers_no_anomaly():
    """특이치가 없으면 빈 dict를 반환한다."""
    data = {"DA-01": 2.0, "DA-02": 2.1, "DA-03": 2.0}
    outliers = calc_sigma_outliers(data, sigma_threshold=3.0)
    assert len(outliers) == 0


def test_chi_square_test():
    """카이제곱 검정 결과에 p_value와 significant 판정이 포함된다."""
    result = chi_square_test(
        defect_in_suspect=50, total_in_suspect=100,
        defect_in_others=10, total_in_others=200,
        alpha=0.05,
    )
    assert "p_value" in result
    assert "significant" in result
    assert result["significant"] is True
    assert result["p_value"] < 0.05
