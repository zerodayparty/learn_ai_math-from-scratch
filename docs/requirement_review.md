# 🟩 요구사항 대응과 계산 기준

## 🟢 1. 원문 보존

`docs/requirement.md`는 변경하지 않는다. 불완전하거나 수학적으로 맞지 않는 부분은 아래 근거를 따라 별도 구현한다.

| 원문 항목 | 실제 처리 | 이유 |
| --- | --- | --- |
| 정답 y_true가 빈 값 | 1을 사용 | 원문 결과 예시에 y_true=1이 명시됨 |
| 원형 함수에서 lr≥0.5 발산 | 0.5, 0.9의 수렴, 1.0의 진동, 1.1의 발산 비교 | x_next=(1-2lr)x이므로 0<lr<1에서 수렴 |
| 이미지가 64×64 이하인데 k=100 | 요청 k와 effective_rank를 별도 기록 | 행렬의 rank 한계는 min(m,n) |
| Momentum의 이점 확인 | beta=0.9의 실제 결과와 추가 튜닝 결과를 함께 제시 | 관성은 잘못 조절하면 더 느리거나 진동함 |
| 은닉 기울기 예시 -0.0513 | 정확한 결과 -0.0517335044를 사용 | BCE와 Sigmoid의 연쇄 법칙으로 직접 계산 |
| 순전파 z2 예시 0.6071 | 정확히 반올림한 0.6072 사용 | 원래 값은 0.6071551038 |
| GD 100회 후 예시 (0.08,0.08) | 실제 값 약 (1.02e-9,1.02e-9) 사용 | 5×0.8¹⁰⁰으로 독립 검증 |
| N(2,0.5)의 두 번째 모수 | 분산 0.5로 명시 | 정규분포를 N(평균,분산)으로 일관되게 정의 |
| Grayscale 이미지 | 제공 PNG를 흑백 최대 64로 전처리 | 원본 데이터와 계산 크기를 분리 |

<br><br>

## 🟢 2. 필수 요구사항 대응표

| 요구사항 | 구현 또는 결과 | 확인 방법 |
| --- | --- | --- |
| 회전·스케일링·전단 전후 비교 | linear_algebra.plot_transformations | transformations.png |
| 행렬식과 면적비 오차 1% 이내 | area_check, 신발끈 공식 | 면적 판정 및 반사·특이 행렬 테스트 |
| Power Iteration 직접 구현 | power_iteration | np.linalg.eig 비교, 독립적인 특성방정식, 잔차 |
| 고유값 오차 5% 이내 | experiments의 검증 코드 | summary의 relative_error |
| SVD k=10,50,100 비교 | svd_compress, plot_svd | MSE·실제 rank·저장량과 이미지 |
| 수치 미분 6, 오차 1e-4 이내 | central_difference | 독립적인 해석적 미분 |
| Gradient 등고선·화살표·수직 | calculus.plot_gradient | gradient.png, 접선 내적=0 |
| 2층 순전파·역전파 수식 | backprop_derivation.ipynb | 모든 식과 중간값 설명 |
| 필수 여섯 미분과 shape | backward, 노트북 표 | 각 미분 출력과 4자리 손계산 비교 |
| NumPy로 손계산 검증 | 노트북의 manual 상수 | 반올림 값의 정확한 배열 비교 |
| 모든 학습 매개변수 독립 검증 | gradient_check | 가중치·편향 전 원소 중심차분 |
| VanillaGD와 Momentum 구현 | optimizer.py | 점화식·beta=0·reset 테스트 |
| lr=0.1,100회 원점 반경 0.1 이내 | circle_optimization | 반경 판정 |
| 수렴 점선 경로와 두 방법 겹침 | plot_convergence | 원형·타원형 PNG |
| 발산 그래프 | lr=1.1 | 손실과 등고선 경로 |
| 타원형 Momentum 비교 | 고정 beta=0.9와 튜닝 비교 | 최초 진입·지속 진입 단계 |
| 정규 PDF 2종·Bernoulli PMF 2종 | probability_loss.ipynb | 한 Figure의 두 패널, 적분·합 검사 |
| Softmax 합 오차 1e-6 이내 | softmax | 큰 점수·상수 이동·합 검사 |
| MSE-MLE 수식 연결 | 확률 노트북 | 고정 분산·독립 관측 가정과 작은 수치 예제 |
| BCE·Categorical CE-MLE 연결 | 확률 노트북 | 우도 곱→로그 합→음의 로그 |
| seed=42 | 전체 실행과 노트북 시작 셀 | 반복 실행 JSON 수치 일치 검사 |
| 모든 그림 PNG 저장 | outputs | 실제 파일·화면 점검 |
| README와 requirements.txt | 루트 파일 | 명령 실행과 잠금 버전 확인 |
| 모든 함수·클래스 Docstring | src, tests, scripts | AST 검사 |
| 쉬운 줄별 코드 주석 | 모든 Python 및 노트북 코드 | 토큰 검사 |
| 자동 미분·scikit-learn 사용 금지 | NumPy 직접 구현 | import 검사 |

<br><br>

## 🟢 3. 보너스와 추가 학습

| 내용 | 구현·학습 위치 |
| --- | --- |
| Adam과 편향 보정 | optimizer.Adam, numerical_checks.ipynb |
| GD·Momentum·Adam 수렴 속도 | optimizer_comparison.png와 실제 반경 기준 |
| Newton과 Hessian | newton_optimize, 정확한 이차 함수 한 단계 검증 |
| 엔트로피·교차 엔트로피·KL | probability.py와 확률 노트북 |
| 미분 간격·상쇄 오차 | 추가 노트북의 sin(x) 간격 실험 |
| 조건수와 학습률 안정성 | 추가 노트북과 _practice/32 |
| 실제 저장량과 rank | SVD 정보 및 _practice/22 |
| 배치·정규화·과적합·계산 그래프 | docs/concepts.md와 노트북의 다음 학습 안내 |

<br><br>

## 🟢 4. 구현의 범위

Power Iteration은 실수 대칭 행렬을 대상으로 하고 기본 예제는 양의 정부호 행렬이다. 유일한 지배 고유값과 그 방향 성분이 있는 초기 벡터가 필요하다. 잔차가 작아도 초기 벡터가 다른 고유벡터라면 지배 고유값이라는 보장은 없다. 이 한계는 일반적인 Power Iteration 자체의 조건이다.

SVD는 요구사항이 허용한 NumPy의 `linalg.svd`를 사용한다. `linalg.eig`는 구현 함수에서 사용하지 않고 결과를 비교하는 검증 코드에서만 호출한다.

수학 알고리즘은 NumPy로 계산한다. Python 표준 라이브러리는 파일·옵션·테스트, Matplotlib은 그림·PNG 입출력, Jupyter 도구는 노트북 실행·표시에 사용한다. Seaborn은 선택적인 시각화 도구라 별도 의존성을 추가하지 않았다.
