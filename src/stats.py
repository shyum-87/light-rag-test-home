"""src/stats.py — 통계 검정 (숫자 계산은 코드가, 해석은 LLM이)."""
import numpy as np
from scipy import stats


def calc_sigma_outliers(
    data: dict[str, float],
    sigma_threshold: float = 3.0,
) -> dict[str, dict]:
    """
    설비별 수치에서 평균 + N*sigma를 초과하는 특이치를 찾는다.

    각 설비에 대해 leave-one-out 방식으로 나머지 설비들의 평균·표준편차를
    구한 뒤, 해당 설비의 z-score가 기준을 넘으면 특이치로 판정한다.

    Args:
        data: {설비ID: 불량률} 형태
        sigma_threshold: 기준 시그마 (기본 3.0)

    Returns:
        {설비ID: {"value": float, "sigma": float, "mean": float, "std": float}}
    """
    if len(data) < 3:
        return {}

    keys = list(data.keys())
    values = list(data.values())

    outliers = {}
    for i, (eqp_id, val) in enumerate(zip(keys, values)):
        others = values[:i] + values[i + 1:]
        mean = float(np.mean(others))
        std = float(np.std(others, ddof=0))

        if std == 0:
            continue

        sigma = (val - mean) / std
        if sigma > sigma_threshold:
            outliers[eqp_id] = {
                "value": val,
                "sigma": round(sigma, 2),
                "mean": round(mean, 4),
                "std": round(std, 4),
            }
    return outliers


def chi_square_test(
    defect_in_suspect: int,
    total_in_suspect: int,
    defect_in_others: int,
    total_in_others: int,
    alpha: float = 0.05,
) -> dict:
    """
    카이제곱 검정으로 혐의 설비의 불량률이 통계적으로 유의미하게 높은지 검증.

    Returns:
        {"chi2": float, "p_value": float, "significant": bool}
    """
    good_in_suspect = total_in_suspect - defect_in_suspect
    good_in_others = total_in_others - defect_in_others

    table = np.array([
        [defect_in_suspect, good_in_suspect],
        [defect_in_others, good_in_others],
    ])

    chi2, p_value, dof, expected = stats.chi2_contingency(table)

    return {
        "chi2": round(float(chi2), 4),
        "p_value": round(float(p_value), 6),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
    }
