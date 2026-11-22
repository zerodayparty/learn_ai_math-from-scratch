"""수학 실험을 실행하고 수치 결과와 그림을 모은다."""

from pathlib import Path  # 결과 파일의 위치를 다룬다.
import json  # 수치 결과를 JSON 파일로 저장한다.
import numpy as np  # 모든 수학 계산을 담당한다.
from src.backprop import backward, fixed_example, gradient_check  # 역전파와 검사 도구를 가져온다.
from src.calculus import central_difference, plot_gradient, quadratic, quadratic_gradient, quadratic_hessian  # 미분과 기울기 그림을 가져온다.
from src.linear_algebra import load_grayscale, plot_svd, plot_transformations, power_iteration  # 고유값과 이미지 그림을 가져온다.
from src.optimizer import Adam, Momentum, VanillaGD, newton_optimize, optimize, plot_convergence as plot_paths  # 네 방법과 경로 그림을 가져온다.
from src.probability import cross_entropy, entropy, kl_divergence, normal_pdf, softmax  # 확률 계산을 가져온다.
from src.visualization import plot_distributions, plot_losses  # 공통 분포와 손실 그림을 가져온다.



def path_summary(path, anisotropy=1.0, tolerance=0.1):  # 경로의 수렴 상태를 요약한다.
    """마지막 반경과 최초 및 지속적 허용 반경 도달 단계를 기록한다."""
    radii = np.linalg.norm(path, axis=1)  # 매 단계의 원점 거리를 구한다.
    hits = np.flatnonzero(radii < tolerance)  # 반경 안에 든 단계를 찾는다.
    sustained = np.flatnonzero(np.logical_and.accumulate((radii < tolerance)[::-1])[::-1])  # 이후에도 반경 안인 단계를 찾는다.
    return {"final_point": path[-1].tolist(), "final_radius": float(radii[-1]), "final_loss": quadratic(path[-1], anisotropy), "first_hit": int(hits[0]) if hits.size else None, "sustained_hit": int(sustained[0]) if sustained.size else None}  # 순간 통과와 지속 수렴을 구별한다.



# ----------------------------------------------------
# ❤️ 핵심 함수 = run_experiments
# 인자로 출력결과물 저장 경로와 사용할 이미지 2개를 받는다.
def run_experiments(output_dir="outputs", image_path=None):  # 과제의 모든 실험을 실행한다.
    """seed를 고정하고 PNG 그림과 검증된 summary.json을 저장한다."""

    # ========================================================================
    np.random.seed(42)  # 넘파이의 기본 난수 생성기를 42에 해당하는 시작 상태로 맞춰.
    # 시드는 씨앗이라는 뜻이야. 컴퓨터는 무작위 숫자를 그냥 뿅 하고 만드는 게 아니라 계산으로 만들어내거든. 
    # 그때 처음에 넣는 숫자가 시드야. 
    # 한번 42를 넣고 숫자 3개를 뽑았잖아? 나중에 또 42를 넣고 3개를 뽑으면 똑같은 3개가 나와. 
    # 그래서 결과를 반복해서 테스트하기 쉬워져. 
    # 변수에 저장하지 않는 건, 그 숫자를 어딘가에 담으려는 게 아니라 난수 생성기의 시작점을 바꾸는 설정이라서야.

    # ========================================================================
    output = Path(output_dir)  # 결과 폴더를 정한다.
    output.mkdir(parents=True, exist_ok=True)  # 결과 폴더를 만든다.

    # ========================================================================    
    report = {"seed": 42, "image_source": "local image" if image_path else "generated example"}  # 개인 경로 없는 조건을 기록한다.
    # 이걸 먼저 2개의 정보를 넣었는데 왜 이러는 걸까?
        # 이번 실험의 시드가 42였다는 것, 그리고 이미지가 파일에서 왔는지 예제인지 시작 조건을 먼저 적어두는 거야. 
        # 나중에 계산 결과를 같은 report에 계속 추가하려고 이렇게 시작하는 거야.




    # ========================================================================
    # ✅ 선형대수 | 행렬 변환과 고유값·고유벡터 (np.linalg = 넘파이 선형대수)
    # ========================================================================
    report["area"] = plot_transformations(output / "transformations.png")  # 행렬 변환과 면적을 저장한다.

    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])  # 요구사항의 양의 정부호 행렬을 만든다.
    
    value, vector, iterations, residual = power_iteration(matrix)  # 행렬을 벡터에 반복해서 곱해 가장 큰 고유값과 그 방향을 구한다.
    # value = 가장 큰 고유값의 근삿값 (4.61803399)
    # vector = 그 고유값에 해당하는 방향의 벡터 ((0.85065081, 0.52573111))
    # iterations = 답을 구할 때까지 반복한 횟수 (36)
    # residual(잔차) = 구한 고유값과 벡터가 정확히 맞아떨어지기까지 남은 차이의 크기
    

    eigenvalues, eigenvectors = np.linalg.eig(matrix)  # 검증 목적으로만 NumPy 고유쌍을 계산 / eig는 고유값과 고유벡터를 찾는 것임.
    index = np.argmax(eigenvalues)  # 예제의 최대 고유값을 찾는다.
    
    # 넘파이로 나온 값을 기준으로, 직접 만든 파워 이터레이션이 목표였던 가장 큰 고유값을 찾았는지 비교.
    eigen_error = float(abs(value - eigenvalues[index]) / abs(eigenvalues[index]))  # 고유값의 상대 오차를 구한다.
    
    alignment_error = float(abs(1 - abs(vector @ eigenvectors[:, index])))  # 부호와 관계없이 방향을 비교한다.
    
    # 레포트에 추가
    report["power_iteration"] = {"value": value, "vector": vector.tolist(), "iterations": iterations, "residual": residual, "relative_error": eigen_error, "alignment_error": alignment_error}  # 고유쌍 결과를 기록한다.





    # ========================================================================
    # ✅ 선형대수 | 특이값 분해(Singular Value Decomposition, SVD)로 이미지 압축
    # ========================================================================
    # 이미지 경로가 None이면 예제 이미지를 만든다.
    image = load_grayscale(image_path)  # 흑백 이미지의 긴 변을 최대 64픽셀로 제한한다.
    report["image_shape"] = list(image.shape)  # 실제 이미지 크기를 기록한다.
    report["svd"] = plot_svd(image, output / "svd_comparison.png")  # 남길 특이값 개수를 바꾸며 복원 이미지, 오차, 저장량을 비교한다.





    # ========================================================================
    # 미분과 기울기 | 한 점에서 함수값이 얼마나 달라지는지 살펴본다.
    # ========================================================================
    derivative = central_difference(lambda x: x ** 2, 3)  # x=3의 수치 미분을 구한다.
    report["derivative"] = {"value": derivative, "absolute_error": abs(derivative - 6)}  # 정답 6과 비교한다.
    plot_gradient(output / "gradient.png")  # 기울기와 등고선을 그린다.
    report["gradient_tangent_dot"] = float(np.array([2.0, 2.0]) @ np.array([-1.0, 1.0]))  # 기울기와 접선의 수직성을 검사한다.





    # ========================================================================
    # 신경망 | 아래 한 줄에서 입력 준비 → 순전파 → 손실 계산 → 역전파 순서로 진행한다.
    # ========================================================================
    # [입력 준비]
    # fixed_example()이 입력, 가중치, 편향, 정답을 준비한다.
    #
    # [순전파]
    # backward()가 forward()를 호출한다.
    # forward()는 중간층 값과 마지막 점수 z2를 구하고 z2를 예측 확률로 바꾼다.
    #
    # [손실 함수]
    # forward() 안의 bce_from_logits()가 z2와 정답을 받는다.
    # 이 함수는 예측의 오차를 이진 교차 엔트로피(Binary Cross-Entropy, BCE)로 계산한다.
    #
    # [역전파]
    # backward()는 예측 확률과 중간층 값을 받아 출력층부터 거꾸로 계산한다.
    # 가중치와 편향을 조금 바꿀 때 손실이 얼마나 달라지는지 기울기로 구한다.
    values, gradients = backward(**fixed_example())  # values에는 예측·손실·중간값, gradients에는 기울기가 들어간다.

    # [기울기 검증]
    # gradient_check()는 가중치나 편향의 값 하나를 조금 늘리거나 줄인다.
    # 두 경우의 손실을 forward()로 다시 구해 수치 기울기를 계산한다.
    # 이 값을 backward()의 기울기와 비교해 가중치·편향별 최대 오차를 기록한다.
    report["backprop"] = {"forward": {key: np.asarray(item).tolist() for key, item in values.items()}, "gradients": {key: np.asarray(item).tolist() for key, item in gradients.items()}, "gradient_check": gradient_check(fixed_example())}  # 값과 기울기 오차를 기록한다.





    # ========================================================================
    # 옵티마이저 비교 | 신경망 실험과 별개로 x²+y², x²+10y²의 최솟값을 찾는다.
    # ========================================================================
    # optimize()는 이차 함수의 기울기를 매번 구하고, 옵티마이저가 정한 방향과 크기만큼 좌표를 옮긴다.
    # 원형 등고선(x²+y²)에서 기본 경사하강법(Gradient Descent, GD)과 모멘텀의 이동 경로를 비교한다.
    circle_paths = {"GD lr=0.1": optimize(quadratic_gradient, [5, 5], VanillaGD(0.1)), "Momentum lr=0.1 beta=0.9": optimize(quadratic_gradient, [5, 5], Momentum(0.1, 0.9))}  # 원형 함수에서 두 방법을 비교한다.
    plot_paths(circle_paths, 1, output / "optimization_circle.png")  # 같은 등고선 위에 경로를 겹친다.
    report["circle_optimization"] = {name: path_summary(path) for name, path in circle_paths.items()}  # 원형 함수의 결과를 기록한다.

    # 원형 등고선에서 학습률(learning rate, lr)을 바꿔 원점에 가까워지는지, 진동하는지, 멀어지는지 본다.
    rate_paths = {f"lr={lr}": optimize(quadratic_gradient, [5, 5], VanillaGD(lr), 30) for lr in [0.1, 0.5, 0.9, 1.0, 1.1]}  # 수렴, 진동, 발산을 비교한다.
    plot_losses({(name + " (overlaps lr=0.1: |1-2lr|=0.8)" if name == "lr=0.9" else name): [quadratic(point) for point in path] for name, path in rate_paths.items()}, output / "learning_rates.png", "Circle: converge lr<1, oscillate lr=1, diverge lr>1")  # 실제 학습률 경계를 그린다.
    plot_paths({"GD lr=1.1 (divergence)": rate_paths["lr=1.1"]}, 1, output / "divergence_path.png")  # 발산 경로를 등고선에 표시한다.
    report["learning_rates"] = {name: path_summary(path) for name, path in rate_paths.items()}  # 학습률별 수치를 기록한다.

    # 타원형 등고선(x²+10y²)에서도 학습률을 바꿔 비교한다. 이 함수는 y 방향으로 더 가파르다.
    ellipse_gradient = lambda point: quadratic_gradient(point, 10)  # 타원형 함수의 기울기를 만든다.
    ellipse_rate_paths = {f"lr={lr}": optimize(ellipse_gradient, [5, 5], VanillaGD(lr), 30) for lr in [0.01, 0.05, 0.09, 0.1, 0.5]}  # 다섯 학습률의 경로를 구한다. 여기서는 lr=0.5에서 발산한다.
    plot_losses({name: [quadratic(point, 10) for point in path] for name, path in ellipse_rate_paths.items()}, output / "learning_rates_ellipse.png", "Ellipse x^2+10y^2: converge lr<0.1, oscillate lr=0.1, diverge lr=0.5")  # 타원형 함수의 학습률 경계를 그린다.
    plot_paths({"GD lr=0.5 (divergence)": ellipse_rate_paths["lr=0.5"][:5]}, 10, output / "divergence_ellipse_lr05.png")  # y가 매번 -9배가 되는 발산 경로를 4단계만 표시한다.
    report["learning_rates_ellipse"] = {name: path_summary(path, 10) for name, path in ellipse_rate_paths.items()}  # 타원형 학습률별 수치를 기록한다.

    # 타원형 등고선에서 경사하강법, 모멘텀, Adam(Adaptive Moment Estimation)의 경로를 구한다.
    ellipse_paths = {"GD lr=0.01": optimize(ellipse_gradient, [5, 5], VanillaGD(0.01), 300), "Momentum lr=0.01 beta=0.9": optimize(ellipse_gradient, [5, 5], Momentum(0.01, 0.9), 300), "GD lr=0.05": optimize(ellipse_gradient, [5, 5], VanillaGD(0.05), 300), "Momentum lr=0.05 beta=0.9": optimize(ellipse_gradient, [5, 5], Momentum(0.05, 0.9), 300), "Adam lr=0.1": optimize(ellipse_gradient, [5, 5], Adam(0.1), 300)}  # 같은 시작점과 반복 예산으로 비교한다.
    plot_paths({name: ellipse_paths[name] for name in ["GD lr=0.01", "Momentum lr=0.01 beta=0.9"]}, 10, output / "optimization_ellipse.png")  # 같은 lr에서 Momentum이 완만한 x 방향을 가속하는 모습을 겹친다.
    plot_paths({name: ellipse_paths[name] for name in ["GD lr=0.05", "Momentum lr=0.05 beta=0.9"]}, 10, output / "optimization_ellipse_lr005.png")  # GD가 이미 빠른 설정에서는 관성이 진동을 만드는 반례를 겹친다.

    # 함수가 얼마나 휘어 있는지 나타내는 곡률을 이용해 조정한 모멘텀과 뉴턴 방법도 비교한다.
    tuned_lr = 4 / (np.sqrt(20) + np.sqrt(2)) ** 2  # 곡률 2와 20에 맞는 학습률을 계산한다.
    tuned_beta = ((np.sqrt(20) - np.sqrt(2)) / (np.sqrt(20) + np.sqrt(2))) ** 2  # 이차 함수에 맞는 관성을 계산한다.
    tuned_path = optimize(ellipse_gradient, [5, 5], Momentum(tuned_lr, tuned_beta), 300)  # 튜닝한 관성의 효과를 확인한다.
    ellipse_paths[f"Tuned Momentum lr={tuned_lr:.4f} beta={tuned_beta:.4f}"] = tuned_path  # 튜닝 결과를 추가한다.
    ellipse_paths["Newton exact Hessian"] = newton_optimize(ellipse_gradient, lambda point: quadratic_hessian(point, 10), [5, 5])  # 정확한 Hessian을 쓰는 Newton을 비교한다.
    plot_paths({"GD lr=0.05": ellipse_paths["GD lr=0.05"], "Tuned Momentum": tuned_path}, 10, output / "momentum_tuned.png")  # 튜닝 전후 이점을 보여준다.
    plot_losses({name: [quadratic(point, 10) for point in path] for name, path in ellipse_paths.items() if "lr=0.01" not in name}, output / "optimizer_comparison.png", "Ellipse lr=0.05 setting: GD, Momentum, Adam, tuned Momentum, Newton")  # lr=0.05 기준으로 다섯 방법의 손실을 비교한다.
    report["ellipse_optimization"] = {name: path_summary(path, 10) for name, path in ellipse_paths.items()}  # 같은 반경 기준으로 결과를 기록한다.
    report["tuned_momentum"] = {"lr": tuned_lr, "beta": tuned_beta, "scope": "positive definite quadratic with Hessian eigenvalues 2 and 20"}  # 튜닝 공식의 한계를 기록한다.





    # ========================================================================
    # 확률과 통계 | 확률분포, 점수를 확률로 바꾸는 방법, 분포의 정보량을 살펴본다.
    # ========================================================================
    plot_distributions(output / "probability_distributions.png")  # 정규분포의 밀도와 베르누이 분포의 0·1 확률을 그린다.
    probabilities = softmax([1000, 1001, 1002])  # 큰 점수 세 개를 합이 1인 확률로 바꾼다.
    support = np.linspace(-10, 10, 20001)  # 정규분포 곡선 아래의 넓이를 근사할 점들을 만든다.
    integrals = [float(np.trapezoid(normal_pdf(support, mean, variance), support)) for mean, variance in [(0, 1), (2, 0.5)]]  # 밀도 아래 면적을 확인한다.
    p, q = np.array([0.3, 0.7]), np.array([0.6, 0.4])  # 같은 두 결과에 대해 서로 다른 확률을 주는 분포를 준비한다.
    # 엔트로피는 p의 불확실성, 교차 엔트로피는 실제 분포 p를 q로 예측할 때의 평균 정보량이다.
    # 쿨백·라이블러 발산(Kullback–Leibler Divergence, KL)은 교차 엔트로피에서 엔트로피를 뺀 값이다.
    report["probability"] = {"softmax": probabilities.tolist(), "sum_error": float(abs(probabilities.sum() - 1)), "pdf_integrals": integrals, "entropy": entropy(p), "cross_entropy": cross_entropy(p, q), "kl": kl_divergence(p, q)}  # 확률 검증과 정보량을 기록한다.





    # ========================================================================
    # 검증 | 각 실험 결과가 기대값이나 허용 오차 기준을 만족하는지 확인한다.
    # ========================================================================
    checks = {}  # 검증 결과를 모을 공간을 만든다.
    checks["area_within_1_percent"] = all(item["error"] <= 0.01 for item in report["area"].values())  # 면적 오차를 검사한다.
    checks["eigenvalue_within_5_percent"] = eigen_error <= 0.05  # 고유값 오차를 검사한다.
    checks["eigenvector_alignment"] = alignment_error < 1e-6  # 고유벡터 방향을 검사한다.
    checks["derivative_within_1e_minus_4"] = abs(derivative - 6) <= 1e-4  # 수치 미분 오차를 검사한다.
    checks["gradient_perpendicular"] = report["gradient_tangent_dot"] == 0  # 접선과 기울기의 내적을 검사한다.
    checks["gd_radius_below_0_1"] = report["circle_optimization"]["GD lr=0.1"]["final_radius"] < 0.1  # 원점 도달을 검사한다.
    checks["circle_lr_above_1_diverges"] = report["learning_rates"]["lr=1.1"]["final_radius"] > report["learning_rates"]["lr=0.1"]["final_radius"] * 1e3 and report["learning_rates"]["lr=1.1"]["final_radius"] > 5 * np.sqrt(2)  # 원형 함수의 lr=1.1 발산을 검사한다.
    checks["ellipse_lr_0_5_diverges"] = report["learning_rates_ellipse"]["lr=0.5"]["final_radius"] > 1e6  # 타원형 함수의 lr=0.5 발산을 검사한다.
    checks["momentum_faster_same_lr_ellipse"] = report["ellipse_optimization"]["Momentum lr=0.01 beta=0.9"]["sustained_hit"] < report["ellipse_optimization"]["GD lr=0.01"]["sustained_hit"]  # 같은 lr에서 Momentum이 먼저 반경 0.1 안에 머무는지 검사한다.
    checks["all_parameter_gradients_checked"] = max(report["backprop"]["gradient_check"].values()) < 1e-8  # 모든 가중치와 편향을 검사한다.
    checks["softmax_sum_within_1e_minus_6"] = report["probability"]["sum_error"] <= 1e-6  # 확률 합을 검사한다.
    checks["svd_mse_nonincreasing"] = all(a["mse"] + 1e-14 >= b["mse"] for a, b in zip(report["svd"], report["svd"][1:]))  # 부동소수점 반올림 허용값을 두고 오차 감소를 확인한다.
    checks["full_rank_reconstruction"] = report["svd"][-1]["mse"] < 1e-20  # 전체 rank 복원을 확인한다.
    checks["normal_pdf_integrates_to_one"] = all(abs(value - 1) < 1e-6 for value in integrals)  # 확률밀도함수(Probability Density Function, PDF) 아래의 넓이가 1에 가까운지 검사한다.





    # ========================================================================
    # 저장 | 검증 결과까지 summary.json에 쓴 뒤 실패 여부를 알린다.
    # ========================================================================
    report["checks"] = {name: bool(value) for name, value in checks.items()}  # NumPy 참거짓도 JSON에 저장되게 Python 참거짓으로 바꿔 기록한다.
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # 개인 경로 없는 JSON을 저장한다.

    if not all(checks.values()):  # 실패가 있으면 완료로 표시하지 않는다.
        raise RuntimeError("Experiment verification failed; inspect summary.json.")  # 검증 실패를 알린다.




    return report  # 실제 실험 결과를 돌려준다.
