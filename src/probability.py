"""확률 분포, 안정적인 손실 계산, 정보 이론을 직접 구현한다."""

import numpy as np  # 확률과 로그 계산에 NumPy를 사용한다.
from src.common import finite_array, positive_number  # 입력 검사 도구를 가져온다.


def normal_pdf(x, mean=0.0, variance=1.0):  # 정규분포의 밀도를 계산한다.
    """N(mean, variance)의 PDF를 반환하며 두 번째 모수는 분산이다."""
    x = finite_array(x)  # 관찰 위치가 유한한지 확인한다.
    finite_array([mean])  # 평균을 확인한다.
    positive_number(variance, "variance")  # 분산은 양수여야 한다.
    return np.exp(-((x - mean) ** 2) / (2 * variance)) / np.sqrt(2 * np.pi * variance)  # 정규분포 공식을 계산한다.


def bernoulli_pmf(p):  # 두 가지 결과의 확률을 구한다.
    """실패 0과 성공 1의 확률 [1-p, p]를 반환한다."""
    if not np.isfinite(p) or not 0 <= p <= 1:  # 확률 범위를 검사한다.
        raise ValueError("p must be in [0, 1].")  # 잘못된 확률을 알린다.
    return np.array([1 - p, p])  # 두 결과의 확률을 돌려준다.


def sigmoid(x):  # 숫자를 0부터 1 사이로 바꾼다.
    """큰 양수와 음수에서도 exp 오버플로 없이 Sigmoid를 계산한다."""
    x = finite_array(x)  # 유한한 입력인지 확인한다.
    small = np.exp(-np.abs(x))  # 항상 0 이하인 지수만 계산한다.
    return np.where(x >= 0, 1 / (1 + small), small / (1 + small))  # 입력 부호에 맞는 공식을 쓴다.


def softmax(logits):  # 점수 벡터를 확률 벡터로 바꾼다.
    """최댓값을 빼 exp 오버플로를 피한 1차원 Softmax를 반환한다."""
    logits = finite_array(logits, 1)  # 점수 벡터를 확인한다.
    with np.errstate(over="ignore"):  # 매우 큰 수끼리의 차이가 음의 무한대여도 exp는 0이 된다.
        weights = np.exp(logits - np.max(logits))  # 상대적인 점수만 지수로 바꾼다.
    return weights / weights.sum()  # 합이 1이 되게 나눈다.


def bce_from_logits(logit, target):  # 분류 손실을 안정적으로 구한다.
    """BCE = log(1+exp(z))-y*z를 logaddexp로 계산한다."""
    finite_array([logit, target])  # 점수와 정답이 유한한지 확인한다.
    if not 0 <= target <= 1:  # 정답 확률의 범위를 확인한다.
        raise ValueError("target must be in [0, 1].")  # 정답 오류를 알린다.
    return float(np.logaddexp(0, logit) - target * logit)  # 확률의 log(0) 문제를 피한다.


def distribution(probabilities):  # 정보 이론의 입력 확률을 확인한다.
    """합이 1인 음수가 없는 확률 벡터를 반환한다."""
    probabilities = finite_array(probabilities, 1)  # 배열 기본 조건을 검사한다.
    if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1, rtol=0, atol=1e-10):  # 확률 조건을 검사한다.
        raise ValueError("Probabilities must be nonnegative and sum to one.")  # 잘못된 분포를 알린다.
    return probabilities  # 확인한 분포를 돌려준다.


def cross_entropy(p, q):  # 실제 분포와 예측 분포의 차이를 구한다.
    """H(p,q)=-Σp*log(q)를 자연로그 단위 nat으로 계산한다."""
    p, q = distribution(p), distribution(q)  # 두 분포를 확인한다.
    if p.shape != q.shape:  # 같은 결과 집합인지 확인한다.
        raise ValueError("Distribution shapes must match.")  # 크기 오류를 알린다.
    support = p > 0  # 실제 발생 가능한 결과만 고른다.
    if np.any(q[support] == 0):  # 가능한 결과의 예측 확률이 0인지 확인한다.
        return float("inf")  # 불가능하다고 예측한 손실은 무한대다.
    return float(-np.sum(p[support] * np.log(q[support])))  # 0*log(0)을 피하며 합한다.


def entropy(p):  # 한 분포의 불확실성을 구한다.
    """H(p)=-Σp*log(p)를 계산하며 0*log(0)=0으로 처리한다."""
    return cross_entropy(p, p)  # 자기 자신과의 교차 엔트로피를 구한다.


def kl_divergence(p, q):  # 분포 사이의 정보 차이를 구한다.
    """D_KL(p||q)=H(p,q)-H(p)를 계산한다."""
    return cross_entropy(p, q) - entropy(p)  # 실제 불확실성을 뺀 추가 손실을 구한다.
