"""명령 하나로 AI 수학 실험과 그림을 만든다."""

import argparse  # 터미널 실행 옵션을 읽는다.
from pathlib import Path  # 로컬 이미지 위치를 찾는다.
import matplotlib  # 화면 없는 환경의 그림 저장을 설정한다.

matplotlib.use("Agg")  # 그래프 창 없이 PNG를 저장한다.
from src.experiments import run_experiments  # 전체 수학 실험을 가져온다.

CHECK_EXPLANATIONS = {  # 검사 이름을 처음 보는 사람도 이해할 수 있도록 설명한다.
    "area_within_1_percent": "행렬 변환의 면적 변화 오차가 1% 이내",  # 변환 전후의 넓이를 비교한다.
    "eigenvalue_within_5_percent": "직접 구한 고유값의 상대 오차가 5% 이내",  # 기준 고유값과 비교한다.
    "eigenvector_alignment": "직접 구한 고유벡터의 방향이 기준과 일치",  # 고유벡터의 부호 차이는 허용한다.
    "derivative_within_1e_minus_4": "수치 미분과 정답 6의 차이가 0.0001 이내",  # 중심차분 결과를 확인한다.
    "gradient_perpendicular": "기울기 화살표가 등고선의 접선과 수직",  # 두 방향의 내적을 확인한다.
    "gd_radius_below_0_1": "경사하강법으로 원점에서 반경 0.1 이내에 도착",  # 최종 점과 원점의 거리를 확인한다.
    "circle_lr_above_1_diverges": "원형 함수에서 lr=1.1이면 원점에서 멀어지며 발산",  # 원형 함수의 학습률 경계를 확인한다.
    "ellipse_lr_0_5_diverges": "타원형 함수에서 lr=0.5이면 발산",  # 원문의 lr≥0.5 발산 조건을 타원형 함수에서 확인한다.
    "momentum_faster_same_lr_ellipse": "타원형 함수에서 같은 lr=0.01일 때 Momentum이 GD보다 먼저 수렴",  # 같은 조건에서 관성의 이점을 확인한다.
    "all_parameter_gradients_checked": "모든 가중치와 편향의 기울기를 수치 미분으로 확인",  # 손으로 구한 기울기를 비교한다.
    "softmax_sum_within_1e_minus_6": "소프트맥스 확률의 합과 1의 차이가 0.000001 이내",  # 출력 확률의 합을 확인한다.
    "svd_mse_nonincreasing": "특이값을 더 사용해 복원할수록 이미지 오차가 늘지 않음",  # 반올림 오차를 고려해 비교한다.
    "full_rank_reconstruction": "가능한 특이값을 모두 사용하면 원본 이미지에 가깝게 복원",  # 전체 계수 복원 오차를 확인한다.
    "normal_pdf_integrates_to_one": "정규분포 밀도 곡선 아래 전체 넓이가 거의 1",  # 넓이를 수치 적분으로 확인한다.
}  # 검사 설명표를 완성한다.


def print_key_numbers(report):  # 판정의 근거가 되는 숫자를 터미널에 보여준다.
    """행렬식·면적비, 고유값, 미분, 최적화, Softmax의 실제 수치를 출력한다."""
    print("\n[선형대수] det(A)와 변환 후 면적비")  # 첫 묶음의 제목을 출력한다.
    for name, item in report["area"].items():  # 세 변환을 차례로 확인한다.
        print(f"  {name}: det={item['determinant']:.4f}, 면적비={item['area_ratio']:.4f}, 오차={item['error'] * 100:.6f}%")  # 행렬식과 면적비, 백분율 오차를 출력한다.
    power = report["power_iteration"]  # 고유값 결과를 꺼낸다.
    print(f"  Power Iteration: λ={power['value']:.4f}, v=[{', '.join(f'{v:.4f}' for v in power['vector'])}], 반복 {power['iterations']}회, np.linalg.eig 대비 오차={power['relative_error'] * 100:.6f}%")  # 고유값과 고유벡터, 오차를 출력한다.
    print("  SVD: " + ", ".join(f"k={item['requested_k']}(실제 {item['effective_rank']}) MSE={item['mse']:.2e}" for item in report["svd"]))  # 세 k의 복원 오차를 한 줄로 출력한다.
    print(f"[미적분] f'(3) 수치 미분={report['derivative']['value']:.10f}, 정답 6과 오차={report['derivative']['absolute_error']:.2e}")  # 수치 미분 결과를 출력한다.
    circle = report["circle_optimization"]  # 원형 함수 결과를 꺼낸다.
    print(f"[최적화] 원형 GD lr=0.1 100회 후 반경={circle['GD lr=0.1']['final_radius']:.2e}, Momentum 반경={circle['Momentum lr=0.1 beta=0.9']['final_radius']:.2e}")  # 원형 함수의 최종 반경을 출력한다.
    print(f"  원형 lr=1.1 30회 후 반경={report['learning_rates']['lr=1.1']['final_radius']:.2e}, 타원형 lr=0.5 30회 후 반경={report['learning_rates_ellipse']['lr=0.5']['final_radius']:.2e} (발산)")  # 두 발산 사례의 크기를 출력한다.
    ellipse = report["ellipse_optimization"]  # 타원형 함수 결과를 꺼낸다.
    print(f"  타원형 lr=0.01 반경 0.1 지속 진입: GD {ellipse['GD lr=0.01']['sustained_hit']}회, Momentum {ellipse['Momentum lr=0.01 beta=0.9']['sustained_hit']}회")  # 같은 lr에서의 수렴 단계를 비교한다.
    print(f"  타원형 lr=0.05 반경 0.1 지속 진입: GD {ellipse['GD lr=0.05']['sustained_hit']}회, Momentum {ellipse['Momentum lr=0.05 beta=0.9']['sustained_hit']}회 (설정에 따라 결과가 바뀌는 반례)")  # 관성이 불리한 설정도 함께 보여준다.
    backprop = report["backprop"]  # 역전파 결과를 꺼낸다.
    print(f"[역전파] y_pred={backprop['forward']['y_pred']:.4f}, dL/dz2={backprop['gradients']['z2']:.4f}, dL/dz1=[{', '.join(f'{v:.4f}' for v in backprop['gradients']['z1'])}]")  # 손계산 체크포인트를 출력한다.
    print(f"[확률] Softmax 합과 1의 차이={report['probability']['sum_error']:.2e}\n")  # 소프트맥스 합 오차를 출력한다.


def main():  # 프로그램 시작점을 만든다.
    """이미지 옵션을 읽고 수학 실험과 검증 결과를 출력한다."""
    parser = argparse.ArgumentParser(description="NumPy-only AI math experiments")  # 명령 안내를 만든다.
    parser.add_argument("--output-dir", default="outputs", help="Folder for PNG and JSON results")  # 출력 폴더 옵션을 만든다.
    images = parser.add_mutually_exclusive_group()  # 충돌하는 이미지 옵션을 묶는다.
    images.add_argument("--image", help="Local PNG image to resize to <=64 pixels")  # 로컬 이미지 옵션을 만든다.
    images.add_argument("--synthetic", action="store_true", help="Use the built-in generated image")  # 생성 이미지 옵션을 만든다.
    args = parser.parse_args()  # 사용자 옵션을 읽는다.
    default_image = Path(__file__).resolve().parent / "data" / "europe.png"  # 제공된 이미지 위치를 찾는다.
    image = args.image or (default_image if default_image.exists() and not args.synthetic else None)  # 지정 이미지, 제공 이미지, 생성 이미지 순서로 고른다.
    report = run_experiments(args.output_dir, image)  # 모든 실험과 수치 검증을 실행한다.
    print("Image source:", report["image_source"], "shape:", report["image_shape"])  # 개인 경로 없이 실행 조건을 보여준다.
    print("이미지 종류:", "로컬 이미지" if image else "프로그램이 만든 예제 이미지", "| 이미지 크기(행, 열):", report["image_shape"])  # 이미지 종류와 숫자 배열 크기를 쉽게 보여준다.
    print_key_numbers(report)  # 평가에 필요한 핵심 수치를 먼저 보여준다.
    for name, passed in report["checks"].items():  # 검증 항목을 순서대로 출력한다.
        explanation = CHECK_EXPLANATIONS.get(name, "상세 수치는 summary.json에서 확인")  # 새 검사가 추가되어도 실행이 중단되지 않게 한다.
        print(f"{'PASS' if passed else 'FAIL'}: {name} — {explanation}")  # 기존 검사 이름과 쉬운 설명을 함께 표시한다.
    print("Generated PNG figures and summary.json.")  # 결과 생성을 알린다.
    print("PASS는 검사 통과, FAIL은 검사 실패를 뜻해.")  # 영어 판정의 뜻을 알려준다.
    print("결과 폴더는 기본적으로 outputs야. --output-dir을 사용했다면 지정한 폴더를 열어봐.")  # 사용자가 첫 결과를 찾도록 안내한다.
    print("먼저 transformations.png와 gradient.png를 열고, 숫자는 summary.json에서 확인해.")  # 그림과 수치를 읽는 시작점을 알려준다.


if __name__ == "__main__":  # 직접 실행할 때만 프로그램을 시작한다.
    main()  # 시작 함수를 호출한다.
