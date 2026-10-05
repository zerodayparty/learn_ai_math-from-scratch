"""여러 계산에서 함께 사용하는 입력 검사."""

import numpy as np  # 숫자 배열을 다루는 도구를 가져온다.


def finite_array(value, ndim=None):  # 숫자 배열의 기본 조건을 확인한다.
    """비어 있지 않은 유한한 실수 배열을 반환한다."""
    array = np.asarray(value, dtype=float)  # 입력을 실수 배열로 바꾼다.
    if array.size == 0 or not np.all(np.isfinite(array)):  # 빈 배열과 무한대, NaN을 거른다.
        raise ValueError("A nonempty finite array is required.")  # 계산할 수 없는 입력을 알린다.
    if ndim is not None and array.ndim != ndim:  # 필요한 차원 수와 비교한다.
        raise ValueError(f"Expected {ndim} dimensions.")  # 잘못된 차원 수를 알린다.
    return array  # 확인을 마친 배열을 돌려준다.


def positive_number(value, name):  # 양수 설정값을 확인한다.
    """유한한 양수 설정값을 확인한다."""
    if not np.isfinite(value) or value <= 0:  # 양수가 아니거나 무한대인지 검사한다.
        raise ValueError(f"{name} must be finite and positive.")  # 설정 오류를 알린다.


def positive_integer(value, name):  # 반복 횟수 같은 정수를 확인한다.
    """불리언을 제외한 양의 정수를 확인한다."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:  # 올바른 정수인지 검사한다.
        raise ValueError(f"{name} must be a positive integer.")  # 정수 설정 오류를 알린다.
