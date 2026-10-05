"""자동 미분 없이 GD, Momentum, Adam, Newton을 직접 구현한다."""

import numpy as np  # 벡터 계산 도구를 가져온다.
from src.common import finite_array, positive_integer, positive_number  # 설정 검사 도구를 가져온다.


class VanillaGD:  # 가장 기본적인 경사하강법을 만든다.
    """theta ← theta - lr*gradient로 한 번 이동한다."""

    def __init__(self, lr=0.1):  # 학습률을 정한다.
        """유한한 양수 학습률을 저장한다."""
        positive_number(lr, "lr")  # 학습률을 확인한다.
        self.lr = lr  # 한 번 이동할 크기를 기억한다.

    def reset(self):  # 다른 실험을 시작할 준비를 한다.
        """GD에는 이전 단계 상태가 없으므로 아무것도 초기화하지 않는다."""
        return None  # 상태가 없음을 나타낸다.

    def step(self, point, gradient):  # 기울기 반대 방향으로 이동한다.
        """동일한 크기의 위치와 기울기로 새 위치를 계산한다."""
        point, gradient = finite_array(point, 1), finite_array(gradient, 1)  # 두 벡터를 확인한다.
        if point.shape != gradient.shape:  # 잘못된 자동 크기 맞춤을 막는다.
            raise ValueError("Point and gradient shapes must match.")  # 크기 오류를 알린다.
        return point - self.lr * gradient  # 내리막 방향으로 이동한다.


class Momentum(VanillaGD):  # 이전 움직임을 기억하는 경사하강법을 만든다.
    """v ← beta*v + gradient, theta ← theta - lr*v인 heavy-ball 방식."""

    def __init__(self, lr=0.1, beta=0.9):  # 학습률과 관성을 정한다.
        """0 <= beta < 1인 관성과 학습률을 저장한다."""
        super().__init__(lr)  # 공통 학습률 검사를 한다.
        if not np.isfinite(beta) or not 0 <= beta < 1:  # 관성의 범위를 확인한다.
            raise ValueError("beta must be in [0, 1).")  # 관성 오류를 알린다.
        self.beta = beta  # 지난 속도를 얼마나 남길지 기억한다.
        self.reset()  # 첫 속도를 비운다.

    def reset(self):  # 속도를 초기화한다.
        """새 실험에 이전 실험의 속도가 섞이지 않게 한다."""
        self.velocity = None  # 아직 속도 벡터가 없게 한다.

    def step(self, point, gradient):  # 관성을 더하여 이동한다.
        """기울기를 누적한 속도로 새 위치를 계산한다."""
        super().step(point, gradient)  # 기본 벡터 조건을 검사한다.
        gradient = np.asarray(gradient, dtype=float)  # 계산용 기울기로 바꾼다.
        if self.velocity is None:  # 첫 단계인지 확인한다.
            self.velocity = np.zeros_like(gradient)  # 속도를 0으로 시작한다.
        if self.velocity.shape != gradient.shape:  # 실험 중 크기가 바뀌는 것을 막는다.
            raise ValueError("Reset the optimizer before changing dimensions.")  # 초기화가 필요함을 알린다.
        self.velocity = self.beta * self.velocity + gradient  # 이전 속도에 새 기울기를 더한다.
        return np.asarray(point) - self.lr * self.velocity  # 속도 반대 방향으로 이동한다.


class Adam(VanillaGD):  # 좌표별로 이동 크기를 조절하는 옵티마이저를 만든다.
    """편향 보정된 기울기 평균과 제곱 평균으로 이동한다."""

    def __init__(self, lr=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):  # Adam 설정을 정한다.
        """학습률, 두 평균의 감쇠율, 분모 보호값을 저장한다."""
        super().__init__(lr)  # 학습률을 확인한다.
        if not all(np.isfinite(b) and 0 <= b < 1 for b in [beta1, beta2]):  # 두 감쇠율을 확인한다.
            raise ValueError("Adam beta values must be in [0, 1).")  # 설정 오류를 알린다.
        positive_number(epsilon, "epsilon")  # 분모 보호값을 확인한다.
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon  # 설정을 저장한다.
        self.reset()  # 누적 평균을 초기화한다.

    def reset(self):  # 과거 평균을 비운다.
        """평균, 제곱 평균, 단계 수를 초기화한다."""
        self.m, self.v, self.t = None, None, 0  # 새 실험 상태를 만든다.

    def step(self, point, gradient):  # Adam 공식으로 이동한다.
        """m_hat과 v_hat의 편향을 보정한 후 좌표별로 이동한다."""
        super().step(point, gradient)  # 위치와 기울기를 검사한다.
        gradient = np.asarray(gradient, dtype=float)  # 기울기를 실수 배열로 바꾼다.
        if self.m is None:  # 첫 단계인지 확인한다.
            self.m, self.v = np.zeros_like(gradient), np.zeros_like(gradient)  # 평균을 0으로 시작한다.
        if self.m.shape != gradient.shape:  # 좌표 수가 바뀌지 않았는지 확인한다.
            raise ValueError("Reset the optimizer before changing dimensions.")  # 초기화 필요를 알린다.
        self.t += 1  # 진행한 단계 수를 센다.
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient  # 기울기 평균을 갱신한다.
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient ** 2  # 기울기 제곱 평균을 갱신한다.
        m_hat, v_hat = self.m / (1 - self.beta1 ** self.t), self.v / (1 - self.beta2 ** self.t)  # 첫 단계의 작은 평균을 보정한다.
        return np.asarray(point) - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)  # 좌표마다 알맞은 크기로 이동한다.


def optimize(gradient_function, initial, optimizer, steps=100):  # 여러 번 이동하여 전체 경로를 만든다.
    """초기 위치를 포함한 (steps+1, dimensions) 경로를 반환한다."""
    positive_integer(steps, "steps")  # 반복 횟수를 확인한다.
    point = finite_array(initial, 1).copy()  # 원래 입력을 바꾸지 않게 복사한다.
    optimizer.reset()  # 다른 실험의 상태를 지운다.
    path = [point.copy()]  # 첫 위치를 기록한다.
    for _ in range(steps):  # 지정한 횟수만큼 반복한다.
        point = optimizer.step(point, gradient_function(point))  # 새 위치를 구한다.
        finite_array(point, 1)  # 발산으로 생긴 NaN이나 무한대를 감지한다.
        path.append(point.copy())  # 이번 위치를 기록한다.
    return np.array(path)  # 전체 경로를 배열로 돌려준다.


def newton_optimize(gradient_function, hessian_function, initial, steps=1):  # 곡률을 이용하여 이동한다.
    """H*delta=gradient를 풀어 theta ← theta-delta로 이동한다."""
    positive_integer(steps, "steps")  # 반복 횟수를 확인한다.
    point = finite_array(initial, 1).copy()  # 시작 위치를 복사한다.
    path = [point.copy()]  # 시작점을 기록한다.
    for _ in range(steps):  # Newton 단계를 반복한다.
        gradient, hessian = finite_array(gradient_function(point), 1), finite_array(hessian_function(point), 2)  # 기울기와 곡률을 검사한다.
        if gradient.shape != point.shape or hessian.shape != (point.size, point.size):  # 벡터와 행렬의 크기를 확인한다.
            raise ValueError("Incompatible Newton gradient or Hessian.")  # 크기 오류를 알린다.
        point -= np.linalg.solve(hessian, gradient)  # 역행렬을 직접 만들지 않고 이동량을 푼다.
        finite_array(point, 1)  # 새 위치가 유한한지 확인한다.
        path.append(point.copy())  # 새 위치를 기록한다.
    return np.array(path)  # 전체 Newton 경로를 돌려준다.


def plot_convergence(paths, anisotropy, destination):  # 최적화 모듈에서 경로를 시각화한다.
    """여러 최적화 경로를 하나의 등고선 Figure에 저장한다."""
    from src.visualization import plot_paths  # 그림 코드를 필요할 때 가져온다.
    return plot_paths(paths, anisotropy, destination)  # 공통 경로 그림을 사용한다.
