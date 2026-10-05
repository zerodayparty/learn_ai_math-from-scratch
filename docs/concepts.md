# 🟩 개념 사전과 다음 공부

## 🟢 1. 용어 사전

| 용어 | 풀네임 또는 뜻 | 쉬운 설명 |
| --- | --- | --- |
| AI | Artificial Intelligence | 데이터에서 규칙을 찾아 사용하는 인공지능 |
| NumPy | Numerical Python | 숫자 배열을 빠르게 계산하는 도구 |
| Matplotlib | MATLAB 스타일의 plotting library에서 유래한 이름 | 숫자를 그림으로 보여주는 도구 |
| ndarray | n-dimensional array | 여러 차원의 숫자 배열 |
| shape | 모양 | 배열의 각 축에 숫자가 몇 개 있는지 표현 |
| scalar | 스칼라 | 숫자 하나. NumPy shape은 () |
| vector | 벡터 | 방향과 크기를 표현하는 숫자 묶음 |
| matrix | 행렬 | 숫자를 행과 열로 놓은 표 |
| determinant, det | 행렬식 | 방향을 포함한 공간의 크기 변화 배율 |
| eigenvalue / eigenvector | 고유값 / 고유벡터 | 변환해도 방향이 유지되는 벡터와 배율 |
| Power Iteration | 거듭제곱 반복법 | 행렬을 반복해서 곱해 지배 방향을 찾는 방법 |
| SVD | Singular Value Decomposition | 중요한 방향과 크기로 행렬을 나누는 특이값 분해 |
| rank | 계수 | 독립적인 방향의 개수 |
| Gradient, grad | 기울기 벡터 | 각 입력을 늘릴 때 함수값이 변하는 정도 |
| Jacobian | 야코비안 | 여러 출력의 1차 편미분을 모은 행렬 |
| Hessian | 헤시안 | 스칼라 함수의 2차 편미분을 모은 행렬 |
| GD | Gradient Descent | 기울기의 반대로 내려가는 경사하강법 |
| lr | learning rate | 한 번 이동할 크기를 조절하는 학습률 |
| Momentum | 관성 | 이전 움직임을 일부 기억하는 최적화 방법 |
| Adam | Adaptive Moment Estimation | 평균과 제곱 평균으로 이동을 조절하는 방법 |
| RMSProp | Root Mean Square Propagation | 제곱 평균의 제곱근으로 크기를 조절하는 방법 |
| PDF | Probability Density Function | 연속 값의 확률 밀도 함수 |
| PMF | Probability Mass Function | 이산 값의 확률 질량 함수 |
| MSE | Mean Squared Error | 차이를 제곱해 평균낸 손실 |
| BCE | Binary Cross-Entropy | 두 가지 정답을 위한 교차 엔트로피 |
| MLE | Maximum Likelihood Estimation | 관찰 데이터의 우도가 가장 큰 매개변수 선택 |
| MAP | Maximum A Posteriori | 우도와 사전 분포를 함께 반영한 추정 |
| KL | Kullback–Leibler divergence | 두 분포의 방향성 있는 정보 차이 |
| tol | tolerance | 허용할 수 있는 오차 |
| NaN | Not a Number | 정상적인 숫자가 아님 |
| PNG | Portable Network Graphics | 그림 파일 형식 |
| JSON | JavaScript Object Notation | 이름과 값을 저장하는 텍스트 형식 |
| API | Application Programming Interface | 프로그램 사이의 호출 규칙 |
| CLI | Command-Line Interface | 글자로 명령을 입력하는 화면 |
| AST | Abstract Syntax Tree | 코드의 문법 구조를 나무처럼 표현 |

`uv`, `Python`, `Sigmoid`, `Softmax`에는 여기에서 더 확장할 공식 약자 풀네임이 없다. 이름을 억지로 만들어 외우지 않는다.

<br><br>

## 🟢 2. 과제 전 알아야 하는 작은 기초

### 🟡 행과 열

`W1`은 (2,2), x는 (2,)이다. `W1 @ x`의 결과는 (2,)다. 첫 행은 첫 은닉 뉴런의 두 입력 가중치이고 둘째 행은 둘째 은닉 뉴런의 가중치다. 행벡터 방식 `x @ W1`은 이 프로젝트의 약속과 다르다.

### 🟡 같은 위치 곱과 행렬 곱

`a * b`는 같은 위치의 숫자를 곱한다. `A @ x`는 행마다 곱한 값을 더한다. `np.outer(a,x)`는 a의 모든 원소와 x의 모든 원소를 조합해 행렬을 만든다.

### 🟡 지수와 로그

지수는 반복 곱을 표현하고 로그는 지수의 반대 계산이다. 양수의 곱에 로그를 붙이면 각 로그의 합이 된다. 우도를 계산할 때 아주 작은 확률의 곱이 0으로 뭉개지는 문제도 줄여준다.

### 🟡 수식에서 숨기는 조건

미분이 존재하는지, 분산이 양수인지, 행렬이 대칭인지, 입력의 shape이 맞는지 확인해야 한다. 수식이 짧아 보여도 아무 입력에 적용할 수 있는 것은 아니다.

<br><br>

## 🟢 3. 추가로 공부할 순서

| 순서 | 내용 | 이번 과제와 연결 |
| --- | --- | --- |
| 1 | 부동소수점·상쇄 오차 | 중심차분 h를 지나치게 줄이지 않기 |
| 2 | 조건수·고유값 간격 | GD 수렴과 Power Iteration의 속도 이해 |
| 3 | 배치 역전파 | 여러 입력의 손실을 평균내는 학습 |
| 4 | 선 탐색과 정규화 | 큰 이동을 조절하고 복잡한 모델을 제한 |
| 5 | 훈련·검증·테스트 분리 | 학습 손실과 일반화 성능 구별 |
| 6 | MAP와 가중치 정규화 | 확률적 가정이 추가 손실로 바뀌는 과정 |
| 7 | 계산 그래프 | 작은 연산의 미분 규칙을 연결 |
| 8 | 벡터와 Jacobian의 곱 | 큰 미분 행렬을 만들지 않고 역전파 |

모든 추가 주제를 구현했다고 주장하지 않는다. Adam·Newton·정보 이론·간격 실험은 구현했고, 배치 역전파·MAP·자동 미분 엔진은 다음 단계의 공부 방향이다.

<br><br>

## 🟢 4. 누군가에게 설명하는 연습

- “이미지는 숫자 행렬이라 SVD로 중요한 방향만 남길 수 있어.”
- “기울기는 함수가 가장 빨리 증가하는 방향이고, 그 반대로 움직이면 작아질 수 있어.”
- “역전파는 각 계산 단계의 미분을 연결해 가중치의 영향을 구하는 방법이야.”
- “학습률을 크게 잡으면 빠르기만 한 것이 아니라 진동하거나 발산할 수 있어.”
- “어떤 확률 분포를 가정했는지에 따라 MSE 또는 교차 엔트로피가 나와.”

각 문장의 예외도 함께 설명한다. 큰 이동은 손실을 늘릴 수 있고, rank가 높으면 압축 저장량이 오히려 늘 수 있다.
