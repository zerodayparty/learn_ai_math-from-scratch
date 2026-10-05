"""독립 수식, 중심차분, 경계 입력으로 구현을 검증한다."""

from pathlib import Path  # 임시 결과를 프로젝트 안에 둔다.
import tempfile  # 검사 후 자동으로 지울 폴더를 만든다.
import unittest  # Python 기본 테스트 도구를 사용한다.
import numpy as np  # 수치 비교와 기준 계산을 한다.
import matplotlib  # 화면 없이 테스트 그림을 저장한다.

matplotlib.use("Agg")  # 테스트 중 그래프 창을 열지 않는다.
from src.backprop import backward, fixed_example, forward, gradient_check  # 신경망 도구를 가져온다.
from src.calculus import central_difference, numerical_gradient, quadratic, quadratic_gradient, quadratic_hessian  # 미분 도구를 가져온다.
from src.experiments import path_summary, run_experiments  # 전체 실행을 가져온다.
from src.linear_algebra import area_check, load_grayscale, power_iteration, rotation, scaling, shear, svd_compress, synthetic_image, transform, unit_circle  # 행렬 도구를 가져온다.
from src.optimizer import Adam, Momentum, VanillaGD, newton_optimize, optimize  # 최적화 도구를 가져온다.
from src.probability import bce_from_logits, bernoulli_pmf, cross_entropy, entropy, kl_divergence, normal_pdf, sigmoid, softmax  # 확률 도구를 가져온다.


class MathTests(unittest.TestCase):  # 수학의 실제 성질을 검증한다.
    """필수 계산과 실패하기 쉬운 경계 조건을 확인한다."""

    def test_rotation_preserves_length(self):  # 회전이 길이를 유지하는지 확인한다.
        """회전 후에도 모든 점이 단위 원에 있다."""
        moved = transform(rotation(np.pi / 4), unit_circle())  # 원을 45도 돌린다.
        np.testing.assert_allclose(np.linalg.norm(moved, axis=0), 1, atol=1e-12)  # 각 점의 길이를 비교한다.

    def test_area_including_reflection_and_singular(self):  # 반사와 붕괴의 면적도 확인한다.
        """면적은 abs(det)이며 특이 행렬은 면적 0이다."""
        for matrix in [rotation(0.7), scaling(), shear(2), scaling(-2, 3), scaling(0, 1)]:  # 여러 변환을 검사한다.
            self.assertLess(area_check(matrix)["error"], 1e-12)  # 신발끈 면적과 행렬식을 비교한다.

    def test_power_against_independent_closed_form(self):  # 직접 풀 수 있는 고유값으로 확인한다.
        """2x2 특성방정식의 해 (7+sqrt(5))/2와 비교한다."""
        value, vector, _, residual = power_iteration([[4, 1], [1, 3]])  # 반복법 결과를 구한다.
        self.assertAlmostEqual(value, (7 + np.sqrt(5)) / 2, places=10)  # 독립적인 근의 공식과 비교한다.
        self.assertAlmostEqual(np.linalg.norm(vector), 1)  # 고유벡터 길이를 확인한다.
        self.assertLess(residual, 1e-8)  # 고유방정식의 잔차를 확인한다.

    def test_negative_dominant_eigenvalue(self):  # 부호가 바뀌는 반복도 확인한다.
        """가장 큰 절댓값의 고유값이 음수인 경우를 처리한다."""
        value, _, _, residual = power_iteration(np.diag([-5.0, 2.0]))  # 음의 지배 고유값을 계산한다.
        self.assertAlmostEqual(value, -5, places=9)  # 최대 수치값 2와 구별한다.
        self.assertLess(residual, 1e-8)  # 수렴을 잔차로 확인한다.

    def test_power_zero_and_failed_convergence(self):  # 반복법의 실패를 숨기지 않는지 확인한다.
        """영행렬은 처리하고 동률 및 영공간 시작은 거부한다."""
        self.assertEqual(power_iteration(np.zeros((2, 2)))[0], 0)  # 영행렬의 고유값을 확인한다.
        with self.assertRaises(RuntimeError):  # 동률 반복의 수렴 실패를 확인한다.
            power_iteration(np.diag([1.0, -1.0]), initial=[1, 1], max_iter=10)  # 진동하는 방향을 만든다.
        with self.assertRaises(ValueError):  # 영공간 시작을 확인한다.
            power_iteration(np.diag([1.0, 0.0]), initial=[0, 1])  # 지배 방향이 없는 시작을 만든다.

    def test_power_input_validation(self):  # 부적합한 행렬과 설정을 확인한다.
        """비대칭, 영벡터, 잘못된 반복 횟수를 거부한다."""
        with self.assertRaises(ValueError):  # 대칭 조건을 확인한다.
            power_iteration([[1, 2], [0, 1]])  # 지원 범위 밖의 행렬을 넣는다.
        with self.assertRaises(ValueError):  # 시작 방향을 확인한다.
            power_iteration(np.eye(2), initial=[0, 0])  # 영벡터를 넣는다.
        with self.assertRaises(ValueError):  # 정수 횟수를 확인한다.
            power_iteration(np.eye(2), max_iter=1.5)  # 소수 횟수를 넣는다.

    def test_svd_monotonic_and_full_reconstruction(self):  # 실제 이미지 복원을 확인한다.
        """rank가 늘면 오차가 줄고 k=100은 64로 제한된다."""
        image = synthetic_image()  # 저작권 문제 없는 예제 이미지를 만든다.
        results = [svd_compress(image, k) for k in [10, 50, 100]]  # 세 rank를 비교한다.
        self.assertGreaterEqual(results[0][1]["mse"], results[1][1]["mse"])  # 오차 감소를 확인한다.
        self.assertEqual(results[-1][1]["effective_rank"], 64)  # 실제 rank 제한을 확인한다.
        np.testing.assert_allclose(results[-1][0], image, atol=1e-12)  # 전체 복원을 확인한다.
        self.assertGreater(results[-1][1]["storage_ratio"], 1)  # 높은 rank는 저장량이 늘 수 있음을 확인한다.

    def test_svd_zero_invalid_rank_and_rectangular(self):  # 이미지 경계 조건을 검사한다.
        """0 이미지와 직사각형도 복원하고 k=0은 거부한다."""
        np.testing.assert_allclose(svd_compress(np.zeros((4, 3)), 10)[0], 0)  # 검은 이미지 복원을 확인한다.
        self.assertEqual(svd_compress(np.ones((4, 3)), 100)[1]["effective_rank"], 3)  # 작은 쪽의 크기로 제한한다.
        with self.assertRaises(ValueError):  # 잘못된 k를 거부하는지 확인한다.
            svd_compress(np.ones((4, 3)), 0)  # 0개 특이값을 요청한다.

    def test_rgba_resize(self):  # 실제 PNG 읽기와 투명도 전처리를 확인한다.
        """투명한 검정은 흰 배경이 되고 종횡비를 유지한다."""
        from matplotlib.image import imsave  # 테스트용 PNG를 저장한다.
        root = Path("outputs")  # 임시 파일을 둘 프로젝트 폴더를 정한다.
        root.mkdir(exist_ok=True)  # 출력 폴더를 준비한다.
        with tempfile.TemporaryDirectory(dir=root) as folder:  # 테스트 후 자동으로 지운다.
            image_path = Path(folder) / "transparent.png"  # 임시 PNG 이름을 정한다.
            imsave(image_path, np.zeros((128, 256, 4)))  # 완전히 투명한 이미지를 만든다.
            image = load_grayscale(image_path)  # 프로젝트의 전처리를 실행한다.
            self.assertEqual(image.shape, (32, 64))  # 종횡비와 크기를 확인한다.
            np.testing.assert_allclose(image, 1)  # 흰 배경에 합성했는지 확인한다.

    def test_derivative(self):  # 중심차분의 정확도를 확인한다.
        """요구된 위치의 미분과 추가 사인 미분을 확인한다."""
        self.assertLess(abs(central_difference(lambda x: x ** 2, 3) - 6), 1e-4)  # 필수 오차를 확인한다.
        self.assertLess(abs(central_difference(np.sin, 0.7) - np.cos(0.7)), 1e-8)  # 다른 함수로도 확인한다.
        with self.assertRaises(ValueError):  # 0으로 나누는 간격을 거부한다.
            central_difference(np.sin, 1, 0)  # 잘못된 간격을 넣는다.

    def test_gradient_and_tangent(self):  # 분석 기울기와 수치 기울기를 비교한다.
        """타원형 편미분과 원형 등고선의 수직성을 확인한다."""
        point = np.array([1.3, -0.7])  # 일반적인 검사 위치를 만든다.
        np.testing.assert_allclose(numerical_gradient(lambda p: quadratic(p, 10), point), quadratic_gradient(point, 10), atol=1e-8)  # 두 계산 방법을 비교한다.
        self.assertAlmostEqual(float(quadratic_gradient(point) @ np.array([-point[1], point[0]])), 0)  # 접선과의 내적을 확인한다.

    def test_backprop_fixed_values(self):  # 반올림 독립 기준과 비교한다.
        """고정 예제 중간값과 모든 기울기의 shape을 확인한다."""
        values, gradients = backward(**fixed_example())  # 고정 예제를 계산한다.
        np.testing.assert_allclose(values["z1"], [0.1, 0.3])  # 첫 선형 계산을 확인한다.
        np.testing.assert_allclose(values["a1"], [0.5249791875, 0.5744425168], atol=1e-10)  # 은닉 활성화를 확인한다.
        self.assertEqual(np.asarray(gradients["W1"]).shape, (2, 2))  # 첫 가중치 shape을 확인한다.
        self.assertEqual(np.asarray(gradients["z2"]).shape, ())  # 출력 기울기 shape을 확인한다.
        np.testing.assert_allclose(gradients["W1"][:, 1], 0)  # 두 번째 입력이 0인 효과를 확인한다.

    def test_backprop_all_parameters_two_targets(self):  # 양쪽 정답에서 역전파를 확인한다.
        """모든 가중치와 편향을 중심차분 손실로 검증한다."""
        for target in [0.0, 1.0]:  # 두 정답 경우를 검사한다.
            example = fixed_example()  # 기본 예제를 가져온다.
            example.update(x=np.array([0.7, -0.3]), b1=np.array([0.2, -0.1]), y_true=target)  # 모든 가중치가 영향을 주는 입력으로 바꾼다.
            self.assertLess(max(gradient_check(example).values()), 1e-8)  # 모든 매개변수의 오차를 확인한다.

    def test_gradient_update_decreases_network_loss(self):  # 기울기가 실제 학습 방향인지 확인한다.
        """작은 학습률의 한 단계가 BCE를 낮춘다."""
        example = fixed_example()  # 기본 예제를 준비한다.
        values, gradients = backward(**example)  # 손실과 기울기를 구한다.
        for name in ["W1", "b1", "W2", "b2"]:  # 학습할 매개변수를 고른다.
            example[name] = example[name] - 0.1 * gradients[name]  # 기울기 반대 방향으로 이동한다.
        self.assertLess(forward(**example)["loss"], values["loss"])  # 실제 손실 감소를 확인한다.

    def test_gd_closed_form_and_radius(self):  # GD 결과를 점화식과 비교한다.
        """x_t=5*(1-2*lr)^t와 100회 도달 반경을 확인한다."""
        path = optimize(quadratic_gradient, [5, 5], VanillaGD(0.1))  # 100회 GD를 실행한다.
        np.testing.assert_allclose(path[:, 0], 5 * 0.8 ** np.arange(101), atol=1e-12)  # 독립적인 점화식 해와 비교한다.
        self.assertLess(np.linalg.norm(path[-1]), 0.1)  # 원점 근처 도달을 확인한다.

    def test_learning_rate_boundary(self):  # 잘못된 발산 문장을 실험으로 검사한다.
        """lr=0.5는 즉시 수렴, 1은 진동, 1.1은 발산한다."""
        for lr in [0.5, 1.0, 1.1]:  # 세 핵심 학습률을 비교한다.
            path = optimize(quadratic_gradient, [5, 5], VanillaGD(lr), 20)  # 짧은 실험을 실행한다.
            expected = 5 * (1 - 2 * lr) ** np.arange(21)  # 이론상 좌표를 계산한다.
            np.testing.assert_allclose(path[:, 0], expected, atol=1e-10)  # 실험과 점화식을 비교한다.

    def test_momentum_reset_and_zero_beta(self):  # 관성 상태의 재현성을 확인한다.
        """beta=0은 GD이고 반복 실행 시 속도가 초기화된다."""
        plain = optimize(quadratic_gradient, [5, 5], VanillaGD(0.1), 10)  # GD 경로를 만든다.
        no_memory = optimize(quadratic_gradient, [5, 5], Momentum(0.1, 0), 10)  # 관성 없는 경로를 만든다.
        np.testing.assert_allclose(plain, no_memory)  # 두 결과가 같은지 확인한다.
        optimizer = Momentum()  # 상태가 있는 옵티마이저를 만든다.
        first = optimize(quadratic_gradient, [5, 5], optimizer, 10)  # 첫 실험을 한다.
        np.testing.assert_array_equal(first, optimize(quadratic_gradient, [5, 5], optimizer, 10))  # 다시 실행해 같은 결과인지 확인한다.

    def test_adam_bias_correction_and_reset(self):  # 첫 단계의 편향 보정을 확인한다.
        """첫 단계는 gradient의 부호 방향으로 거의 lr만큼 이동한다."""
        optimizer = Adam(0.1)  # Adam을 준비한다.
        gradient = np.array([2.0, -4.0])  # 다른 크기와 부호의 기울기를 준다.
        expected = np.array([1.0, 1.0]) - 0.1 * gradient / (abs(gradient) + 1e-8)  # 독립적으로 첫 단계 공식을 구한다.
        np.testing.assert_allclose(optimizer.step([1, 1], gradient), expected, atol=1e-12)  # 평균 편향 보정을 확인한다.
        first = optimize(quadratic_gradient, [5, 5], optimizer, 20)  # Adam 경로를 구한다.
        np.testing.assert_array_equal(first, optimize(quadratic_gradient, [5, 5], optimizer, 20))  # 초기화의 재현성을 확인한다.

    def test_newton_one_step_and_singular(self):  # Newton의 이점과 한계를 확인한다.
        """양의 정부호 이차 함수는 한 번에 풀고 특이 Hessian은 실패한다."""
        path = newton_optimize(lambda p: quadratic_gradient(p, 10), lambda p: quadratic_hessian(p, 10), [5, 5])  # 정확한 곡률을 사용한다.
        np.testing.assert_allclose(path[-1], [0, 0], atol=1e-12)  # 한 번에 최솟값에 도달했는지 확인한다.
        with self.assertRaises(np.linalg.LinAlgError):  # 역으로 풀 수 없는 곡률을 확인한다.
            newton_optimize(quadratic_gradient, lambda p: np.zeros((2, 2)), [5, 5])  # 특이 Hessian을 넣는다.

    def test_optimizer_input_validation(self):  # 잘못된 설정과 벡터 크기를 확인한다.
        """음수 학습률, 잘못된 관성, 서로 다른 shape을 거부한다."""
        for constructor in [lambda: VanillaGD(-1), lambda: Momentum(beta=1), lambda: Adam(beta2=1)]:  # 세 설정 오류를 검사한다.
            with self.assertRaises(ValueError):  # 오류 발생을 확인한다.
                constructor()  # 잘못된 옵티마이저를 만든다.
        with self.assertRaises(ValueError):  # shape 오류를 검사한다.
            VanillaGD().step([1, 2], [1])  # 크기가 다른 기울기를 준다.

    def test_softmax_extremes_and_shift_invariance(self):  # 큰 점수에서도 확률을 확인한다.
        """Softmax 합은 1이며 공통 상수를 더해도 같다."""
        np.testing.assert_allclose(softmax([1, 2, 3]), softmax([1001, 1002, 1003]), atol=1e-12)  # 상수 이동의 불변성을 확인한다.
        self.assertLess(abs(softmax([-1e308, 1e308]).sum() - 1), 1e-6)  # 극단 입력의 합을 확인한다.
        np.testing.assert_allclose(softmax([3, 3]), [0.5, 0.5])  # 같은 점수의 균등 분포를 확인한다.
        with self.assertRaises(ValueError):  # NaN 입력을 거부한다.
            softmax([np.nan])  # 계산 불가능한 입력을 넣는다.

    def test_stable_sigmoid_and_bce(self):  # 큰 점수에서 손실을 확인한다.
        """Sigmoid와 BCE가 오버플로 없이 극단값을 처리한다."""
        np.testing.assert_allclose(sigmoid([-1000, 0, 1000]), [0, 0.5, 1])  # 활성화의 극한을 확인한다.
        self.assertEqual(bce_from_logits(1000, 0), 1000)  # 큰 오답의 손실을 확인한다.
        self.assertEqual(bce_from_logits(-1000, 1), 1000)  # 반대쪽 큰 오답을 확인한다.
        self.assertAlmostEqual(bce_from_logits(0, 1), np.log(2))  # 확률 0.5의 손실을 확인한다.

    def test_distributions_integrate_and_sum(self):  # 확률 분포의 기본 성질을 확인한다.
        """두 정규분포 PDF 면적과 두 Bernoulli PMF 합이 1이다."""
        x = np.linspace(-10, 10, 20001)  # 충분히 넓은 적분 구간을 만든다.
        for mean, variance in [(0, 1), (2, 0.5)]:  # 두 정규분포를 검사한다.
            self.assertAlmostEqual(float(np.trapezoid(normal_pdf(x, mean, variance), x)), 1, places=6)  # 밀도 넓이를 확인한다.
        for p in [0, 0.3, 0.7, 1]:  # 경계를 포함한 베르누이 분포를 검사한다.
            self.assertEqual(float(bernoulli_pmf(p).sum()), 1)  # 확률 합을 확인한다.
        with self.assertRaises(ValueError):  # 음수 분산을 거부한다.
            normal_pdf(x, variance=-1)  # 잘못된 분포를 만든다.

    def test_information_identity_and_zero_support(self):  # 정보 이론 공식과 경계를 확인한다.
        """H(p,q)=H(p)+KL이며 q=0인 가능한 사건은 무한 손실이다."""
        p, q = [0.3, 0.7], [0.6, 0.4]  # 두 확률 분포를 만든다.
        self.assertAlmostEqual(cross_entropy(p, q), entropy(p) + kl_divergence(p, q))  # 분해 공식을 확인한다.
        self.assertGreaterEqual(kl_divergence(p, q), 0)  # KL의 비음수성을 확인한다.
        self.assertEqual(entropy([1, 0]), 0)  # 확실한 결과의 불확실성은 0이다.
        self.assertTrue(np.isinf(cross_entropy([1, 0], [0, 1])))  # 불가능하다는 예측의 무한 손실을 확인한다.
        with self.assertRaises(ValueError):  # 합이 1이 아닌 확률을 거부한다.
            entropy([0.2, 0.2])  # 잘못된 분포를 넣는다.

    def test_first_hit_is_not_sustained_convergence(self):  # 순간 통과와 지속 수렴을 구별한다.
        """잠깐 원점을 통과한 경로를 지속 수렴으로 표시하지 않는다."""
        result = path_summary(np.array([[5, 5], [0, 0], [1, 1]]))  # 원점을 지나 다시 떠나는 경로를 만든다.
        self.assertEqual(result["first_hit"], 1)  # 첫 도달을 기록한다.
        self.assertIsNone(result["sustained_hit"])  # 지속 수렴은 아니라고 확인한다.

    def test_full_pipeline_artifacts(self):  # 모든 실험의 실제 파일 생성을 검사한다.
        """전체 검증 통과와 10개 PNG 저장을 확인한다."""
        root = Path("outputs")  # 임시 파일을 프로젝트 안에 둔다.
        root.mkdir(exist_ok=True)  # 출력 폴더를 만든다.
        with tempfile.TemporaryDirectory(dir=root) as folder:  # 검사 후 결과를 자동으로 지운다.
            report = run_experiments(folder)  # 외부 이미지 없이 전체 실험을 실행한다.
            self.assertTrue(all(report["checks"].values()))  # 전체 수치 판정을 확인한다.
            self.assertEqual(len(list(Path(folder).glob("*.png"))), 10)  # 요구된 그림이 모두 만들어졌는지 확인한다.
            self.assertTrue((Path(folder) / "summary.json").is_file())  # 요약 데이터 파일을 확인한다.


if __name__ == "__main__":  # 직접 테스트 파일을 실행한 경우를 처리한다.
    unittest.main()  # 모든 테스트를 실행한다.
