"""중심차분법으로 미분하고 두 이차 함수의 기울기를 계산한다."""

import numpy as np  # 수학 계산 도구를 가져온다.
from src.common import finite_array, positive_number  # 입력 검사 도구를 가져온다.


def central_difference(function, x, h=1e-5):  # 양쪽의 함수값으로 기울기를 구한다.
    """f'(x) ≈ (f(x+h)-f(x-h))/(2h)를 계산한다."""
    finite_array([x])  # 미분 위치를 확인한다.
    positive_number(h, "h")  # 양수 간격을 확인한다.
    return float((function(x + h) - function(x - h)) / (2 * h))  # 중심차분 공식으로 계산한다.


def numerical_gradient(function, point, h=1e-5):  # 여러 좌표의 편미분을 계산한다.
    """각 좌표 하나씩 바꿔 스칼라 함수의 수치 기울기를 구한다."""
    point = finite_array(point, 1)  # 벡터 형태를 확인한다.
    positive_number(h, "h")  # 간격을 확인한다.
    gradient = np.empty_like(point)  # 좌표마다 기울기를 담을 공간을 만든다.
    for index in range(point.size):  # 좌표를 하나씩 선택한다.
        offset = np.zeros_like(point)  # 나머지 좌표는 움직이지 않게 한다.
        offset[index] = h  # 선택한 좌표만 조금 움직인다.
        gradient[index] = (function(point + offset) - function(point - offset)) / (2 * h)  # 그 좌표의 편미분을 구한다.
    return gradient  # 모든 좌표의 기울기를 돌려준다.


def quadratic(point, anisotropy=1.0):  # 원형 또는 타원형 손실을 계산한다.
    """f(x,y)=x²+anisotropy*y²의 값을 계산한다."""
    point = finite_array(point, 1)  # 좌표를 확인한다.
    positive_number(anisotropy, "anisotropy")  # 세로 방향 계수를 확인한다.
    if point.shape != (2,):  # 두 좌표인지 확인한다.
        raise ValueError("Expected two coordinates.")  # 크기 오류를 알린다.
    return float(point[0] ** 2 + anisotropy * point[1] ** 2)  # 두 방향의 제곱을 합한다.


def quadratic_gradient(point, anisotropy=1.0):  # 이차 함수의 기울기를 구한다.
    """∇f=(2x, 2*anisotropy*y)를 반환한다."""
    quadratic(point, anisotropy)  # 함수와 같은 좌표 조건을 확인한다.
    return 2 * np.asarray(point, dtype=float) * np.array([1, anisotropy])  # 두 축의 편미분을 구한다.


def quadratic_hessian(point, anisotropy=1.0):  # 두 번 미분한 행렬을 구한다.
    """H=diag(2, 2*anisotropy)를 반환한다."""
    quadratic(point, anisotropy)  # 좌표와 계수를 확인한다.
    return np.diag([2.0, 2 * anisotropy])  # 곡면의 휘어진 정도를 반환한다.


def plot_gradient(destination):  # 미적분 모듈에서 기울기를 시각화한다.
    """원형 함수의 등고선과 Gradient를 PNG로 저장한다."""
    from src.visualization import plot_gradient as plot  # 그림 코드를 필요할 때 가져온다.
    return plot(destination)  # 공통 그림 함수에 저장을 맡긴다.
