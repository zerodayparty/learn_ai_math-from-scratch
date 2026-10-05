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


def run_experiments(output_dir="outputs", image_path=None):  # 과제의 모든 실험을 실행한다.
    """seed를 고정하고 PNG 그림과 검증된 summary.json을 저장한다."""
    np.random.seed(42)  # 요구사항의 난수 시드를 고정한다.
    output = Path(output_dir)  # 결과 폴더를 정한다.
    output.mkdir(parents=True, exist_ok=True)  # 결과 폴더를 만든다.
    report = {"seed": 42, "image_source": "local image" if image_path else "generated example"}  # 개인 경로 없는 조건을 기록한다.
    report["area"] = plot_transformations(output / "transformations.png")  # 행렬 변환과 면적을 저장한다.
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])  # 요구사항의 양의 정부호 행렬을 만든다.
    value, vector, iterations, residual = power_iteration(matrix)  # 직접 지배 고유쌍을 계산한다.
    eigenvalues, eigenvectors = np.linalg.eig(matrix)  # 검증 목적으로만 NumPy 고유쌍을 계산한다.
    index = np.argmax(eigenvalues)  # 예제의 최대 고유값을 찾는다.
    eigen_error = float(abs(value - eigenvalues[index]) / abs(eigenvalues[index]))  # 고유값의 상대 오차를 구한다.
    alignment_error = float(abs(1 - abs(vector @ eigenvectors[:, index])))  # 부호와 관계없이 방향을 비교한다.
    report["power_iteration"] = {"value": value, "vector": vector.tolist(), "iterations": iterations, "residual": residual, "relative_error": eigen_error, "alignment_error": alignment_error}  # 고유쌍 결과를 기록한다.
    image = load_grayscale(image_path)  # 이미지를 흑백 64 이하로 준비한다.
    report["image_shape"] = list(image.shape)  # 실제 이미지 크기를 기록한다.
    report["svd"] = plot_svd(image, output / "svd_comparison.png")  # 세 rank의 복원 결과를 비교한다.
    derivative = central_difference(lambda x: x ** 2, 3)  # x=3의 수치 미분을 구한다.
    report["derivative"] = {"value": derivative, "absolute_error": abs(derivative - 6)}  # 정답 6과 비교한다.
    plot_gradient(output / "gradient.png")  # 기울기와 등고선을 그린다.
    report["gradient_tangent_dot"] = float(np.array([2.0, 2.0]) @ np.array([-1.0, 1.0]))  # 기울기와 접선의 수직성을 검사한다.
    values, gradients = backward(**fixed_example())  # 고정 신경망의 역전파를 계산한다.
    report["backprop"] = {"forward": {key: np.asarray(item).tolist() for key, item in values.items()}, "gradients": {key: np.asarray(item).tolist() for key, item in gradients.items()}, "gradient_check": gradient_check(fixed_example())}  # 값과 기울기 오차를 기록한다.
    circle_paths = {"GD lr=0.1": optimize(quadratic_gradient, [5, 5], VanillaGD(0.1)), "Momentum lr=0.1 beta=0.9": optimize(quadratic_gradient, [5, 5], Momentum(0.1, 0.9))}  # 원형 함수에서 두 방법을 비교한다.
    plot_paths(circle_paths, 1, output / "optimization_circle.png")  # 같은 등고선 위에 경로를 겹친다.
    report["circle_optimization"] = {name: path_summary(path) for name, path in circle_paths.items()}  # 원형 함수의 결과를 기록한다.
    rate_paths = {f"lr={lr}": optimize(quadratic_gradient, [5, 5], VanillaGD(lr), 30) for lr in [0.1, 0.5, 0.9, 1.0, 1.1]}  # 수렴, 진동, 발산을 비교한다.
    plot_losses({name: [quadratic(point) for point in path] for name, path in rate_paths.items()}, output / "learning_rates.png", "Circle: converge lr<1, oscillate lr=1, diverge lr>1")  # 실제 학습률 경계를 그린다.
    plot_paths({"GD lr=1.1 (divergence)": rate_paths["lr=1.1"]}, 1, output / "divergence_path.png")  # 발산 경로를 등고선에 표시한다.
    report["learning_rates"] = {name: path_summary(path) for name, path in rate_paths.items()}  # 학습률별 수치를 기록한다.
    ellipse_gradient = lambda point: quadratic_gradient(point, 10)  # 타원형 함수의 기울기를 만든다.
    ellipse_paths = {"GD lr=0.05": optimize(ellipse_gradient, [5, 5], VanillaGD(0.05), 300), "Momentum lr=0.05 beta=0.9": optimize(ellipse_gradient, [5, 5], Momentum(0.05, 0.9), 300), "Adam lr=0.1": optimize(ellipse_gradient, [5, 5], Adam(0.1), 300)}  # 같은 시작점과 반복 예산으로 비교한다.
    plot_paths({name: path for name, path in ellipse_paths.items() if not name.startswith("Adam")}, 10, output / "optimization_ellipse.png")  # 요구된 두 경로를 겹친다.
    tuned_lr = 4 / (np.sqrt(20) + np.sqrt(2)) ** 2  # 곡률 2와 20에 맞는 학습률을 계산한다.
    tuned_beta = ((np.sqrt(20) - np.sqrt(2)) / (np.sqrt(20) + np.sqrt(2))) ** 2  # 이차 함수에 맞는 관성을 계산한다.
    tuned_path = optimize(ellipse_gradient, [5, 5], Momentum(tuned_lr, tuned_beta), 300)  # 튜닝한 관성의 효과를 확인한다.
    ellipse_paths[f"Tuned Momentum lr={tuned_lr:.4f} beta={tuned_beta:.4f}"] = tuned_path  # 튜닝 결과를 추가한다.
    ellipse_paths["Newton exact Hessian"] = newton_optimize(ellipse_gradient, lambda point: quadratic_hessian(point, 10), [5, 5])  # 정확한 Hessian을 쓰는 Newton을 비교한다.
    plot_paths({"GD lr=0.05": ellipse_paths["GD lr=0.05"], "Tuned Momentum": tuned_path}, 10, output / "momentum_tuned.png")  # 튜닝 전후 이점을 보여준다.
    plot_losses({name: [quadratic(point, 10) for point in path] for name, path in ellipse_paths.items()}, output / "optimizer_comparison.png", "Ellipse: settings matter; Newton uses an exact Hessian")  # 네 방법의 손실을 비교한다.
    report["ellipse_optimization"] = {name: path_summary(path, 10) for name, path in ellipse_paths.items()}  # 같은 반경 기준으로 결과를 기록한다.
    report["tuned_momentum"] = {"lr": tuned_lr, "beta": tuned_beta, "scope": "positive definite quadratic with Hessian eigenvalues 2 and 20"}  # 튜닝 공식의 한계를 기록한다.
    plot_distributions(output / "probability_distributions.png")  # PDF와 PMF 그림을 저장한다.
    probabilities = softmax([1000, 1001, 1002])  # 큰 점수의 안정적인 확률을 구한다.
    support = np.linspace(-10, 10, 20001)  # 정규분포 적분의 구간을 만든다.
    integrals = [float(np.trapezoid(normal_pdf(support, mean, variance), support)) for mean, variance in [(0, 1), (2, 0.5)]]  # 밀도 아래 면적을 확인한다.
    p, q = np.array([0.3, 0.7]), np.array([0.6, 0.4])  # 두 정보 분포를 정한다.
    report["probability"] = {"softmax": probabilities.tolist(), "sum_error": float(abs(probabilities.sum() - 1)), "pdf_integrals": integrals, "entropy": entropy(p), "cross_entropy": cross_entropy(p, q), "kl": kl_divergence(p, q)}  # 확률 검증과 정보량을 기록한다.
    checks = {}  # 검증 결과를 모을 공간을 만든다.
    checks["area_within_1_percent"] = all(item["error"] <= 0.01 for item in report["area"].values())  # 면적 오차를 검사한다.
    checks["eigenvalue_within_5_percent"] = eigen_error <= 0.05  # 고유값 오차를 검사한다.
    checks["eigenvector_alignment"] = alignment_error < 1e-6  # 고유벡터 방향을 검사한다.
    checks["derivative_within_1e_minus_4"] = abs(derivative - 6) <= 1e-4  # 수치 미분 오차를 검사한다.
    checks["gradient_perpendicular"] = report["gradient_tangent_dot"] == 0  # 접선과 기울기의 내적을 검사한다.
    checks["gd_radius_below_0_1"] = report["circle_optimization"]["GD lr=0.1"]["final_radius"] < 0.1  # 원점 도달을 검사한다.
    checks["all_parameter_gradients_checked"] = max(report["backprop"]["gradient_check"].values()) < 1e-8  # 모든 가중치와 편향을 검사한다.
    checks["softmax_sum_within_1e_minus_6"] = report["probability"]["sum_error"] <= 1e-6  # 확률 합을 검사한다.
    checks["svd_mse_nonincreasing"] = all(a["mse"] + 1e-14 >= b["mse"] for a, b in zip(report["svd"], report["svd"][1:]))  # 부동소수점 반올림 허용값을 두고 오차 감소를 확인한다.
    checks["full_rank_reconstruction"] = report["svd"][-1]["mse"] < 1e-20  # 전체 rank 복원을 확인한다.
    checks["normal_pdf_integrates_to_one"] = all(abs(value - 1) < 1e-6 for value in integrals)  # PDF의 넓이를 검사한다.
    report["checks"] = checks  # 판정 결과를 기록한다.
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # 개인 경로 없는 JSON을 저장한다.
    if not all(checks.values()):  # 실패가 있으면 완료로 표시하지 않는다.
        raise RuntimeError("Experiment verification failed; inspect summary.json.")  # 검증 실패를 알린다.
    return report  # 실제 실험 결과를 돌려준다.
