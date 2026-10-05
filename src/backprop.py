"""입력 2개, 은닉 2개, 출력 1개인 신경망의 순전파와 역전파."""

import numpy as np  # 텐서 연산을 직접 계산한다.
from src.common import finite_array  # 입력 조건을 검사한다.
from src.probability import bce_from_logits, sigmoid  # 안정적인 활성화와 손실을 가져온다.


def fixed_example():  # 요구사항의 고정 예제를 준비한다.
    """빈 y_true는 요구사항 결과 예시를 근거로 1로 둔다."""
    return {"x": np.array([1.0, 0.0]), "W1": np.array([[0.1, 0.2], [0.3, 0.4]]), "b1": np.zeros(2), "W2": np.array([0.5, 0.6]), "b2": np.array(0.0), "y_true": 1.0}  # 원문의 입력과 가중치를 만든다.


def forward(x, W1, b1, W2, b2, y_true):  # 입력에서 손실까지 계산한다.
    """열벡터 관례 z1=W1@x+b1, z2=W2@a1+b2를 사용한다."""
    x, W1, b1, W2 = finite_array(x, 1), finite_array(W1, 2), finite_array(b1, 1), finite_array(W2, 1)  # 각 배열의 차원을 확인한다.
    if x.shape != (2,) or W1.shape != (2, 2) or b1.shape != (2,) or W2.shape != (2,):  # 고정 신경망 구조인지 확인한다.
        raise ValueError("Expected a 2 -> 2 -> 1 network.")  # 잘못된 구조를 알린다.
    if np.asarray(b2).shape != ():  # 출력 편향이 스칼라인지 확인한다.
        raise ValueError("b2 must be scalar.")  # 편향 형태 오류를 알린다.
    finite_array([float(b2), y_true])  # 편향과 정답을 확인한다.
    z1 = W1 @ x + b1  # 첫 번째 층의 선형 계산을 한다.
    a1 = sigmoid(z1)  # 은닉층 활성화를 계산한다.
    z2 = float(W2 @ a1 + b2)  # 출력층의 선형 계산을 한다.
    y_pred = float(sigmoid(z2))  # 출력 확률을 계산한다.
    loss = bce_from_logits(z2, y_true)  # 정답과 예측의 손실을 계산한다.
    return {"z1": z1, "a1": a1, "z2": z2, "y_pred": y_pred, "loss": loss}  # 손계산 체크포인트를 반환한다.


def backward(x, W1, b1, W2, b2, y_true):  # 연쇄 법칙으로 모든 기울기를 구한다.
    """BCE와 Sigmoid 결합에서 dL/dz2=y_pred-y_true를 사용한다."""
    values = forward(x, W1, b1, W2, b2, y_true)  # 역전파에 필요한 중간값을 구한다.
    p, a1 = values["y_pred"], values["a1"]  # 출력 확률과 은닉 활성화를 꺼낸다.
    if not 0 < p < 1:  # dL/dp가 수치적으로 무한대가 되는 경우를 막는다.
        raise ValueError("Use moderate logits to inspect dL/dy_pred; sigmoid is saturated.")  # 손계산 예제의 유효 범위를 알린다.
    dy_pred = -y_true / p + (1 - y_true) / (1 - p)  # BCE를 예측 확률에 대해 미분한다.
    dz2 = p - y_true  # Sigmoid 미분을 곱해 간단해진 기울기를 구한다.
    dW2 = dz2 * a1  # 출력층 가중치의 기울기를 구한다.
    da1 = dz2 * np.asarray(W2)  # 은닉 활성화로 기울기를 전달한다.
    dz1 = da1 * a1 * (1 - a1)  # 은닉 Sigmoid의 기울기를 곱한다.
    dW1 = np.outer(dz1, x)  # 각 은닉 기울기와 각 입력을 모두 곱한다.
    gradients = {"y_pred": dy_pred, "z2": dz2, "W2": dW2, "a1": da1, "z1": dz1, "W1": dW1, "b2": dz2, "b1": dz1.copy()}  # 편향 기울기까지 묶는다.
    return values, gradients  # 순전파와 역전파 기록을 함께 반환한다.


def gradient_check(example, h=1e-5):  # 역전파를 다른 계산 방법으로 검사한다.
    """모든 가중치와 편향의 중심차분 기울기와 분석 기울기를 비교한다."""
    from src.common import positive_number  # 수치 미분 간격 검사를 가져온다.
    positive_number(h, "h")  # 간격이 양수인지 확인한다.
    _, gradients = backward(**example)  # 분석적으로 구한 기울기를 꺼낸다.
    errors = {}  # 매개변수별 최대 차이를 담는다.
    for name in ["W1", "b1", "W2", "b2"]:  # 모든 학습 대상 매개변수를 확인한다.
        parameter = np.asarray(example[name], dtype=float)  # 매개변수를 배열로 다룬다.
        numerical = np.empty_like(parameter)  # 수치 기울기를 담을 공간을 만든다.
        for index in np.ndindex(parameter.shape):  # 스칼라를 포함한 모든 원소를 순회한다.
            plus, minus = parameter.copy(), parameter.copy()  # 위아래로 바꿀 두 복사본을 만든다.
            plus[index] += h  # 한 원소를 조금 늘린다.
            minus[index] -= h  # 한 원소를 조금 줄인다.
            high, low = dict(example), dict(example)  # 나머지 입력은 그대로 둔다.
            high[name], low[name] = plus, minus  # 이번 매개변수만 바꾼다.
            numerical[index] = (forward(**high)["loss"] - forward(**low)["loss"]) / (2 * h)  # 손실의 중심차분을 계산한다.
        errors[name] = float(np.max(abs(numerical - gradients[name])))  # 가장 큰 기울기 차이를 기록한다.
    return errors  # 모든 매개변수의 검사 결과를 반환한다.
