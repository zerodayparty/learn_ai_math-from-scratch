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
    "all_parameter_gradients_checked": "모든 가중치와 편향의 기울기를 수치 미분으로 확인",  # 손으로 구한 기울기를 비교한다.
    "softmax_sum_within_1e_minus_6": "소프트맥스 확률의 합과 1의 차이가 0.000001 이내",  # 출력 확률의 합을 확인한다.
    "svd_mse_nonincreasing": "특이값을 더 사용해 복원할수록 이미지 오차가 늘지 않음",  # 반올림 오차를 고려해 비교한다.
    "full_rank_reconstruction": "가능한 특이값을 모두 사용하면 원본 이미지에 가깝게 복원",  # 전체 계수 복원 오차를 확인한다.
    "normal_pdf_integrates_to_one": "정규분포 밀도 곡선 아래 전체 넓이가 거의 1",  # 넓이를 수치 적분으로 확인한다.
}  # 검사 설명표를 완성한다.


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
    for name, passed in report["checks"].items():  # 검증 항목을 순서대로 출력한다.
        explanation = CHECK_EXPLANATIONS.get(name, "상세 수치는 summary.json에서 확인")  # 새 검사가 추가되어도 실행이 중단되지 않게 한다.
        print(f"{'PASS' if passed else 'FAIL'}: {name} — {explanation}")  # 기존 검사 이름과 쉬운 설명을 함께 표시한다.
    print("Generated PNG figures and summary.json.")  # 결과 생성을 알린다.
    print("PASS는 검사 통과, FAIL은 검사 실패를 뜻해.")  # 영어 판정의 뜻을 알려준다.
    print("결과 폴더는 기본적으로 outputs야. --output-dir을 사용했다면 지정한 폴더를 열어봐.")  # 사용자가 첫 결과를 찾도록 안내한다.
    print("먼저 transformations.png와 gradient.png를 열고, 숫자는 summary.json에서 확인해.")  # 그림과 수치를 읽는 시작점을 알려준다.


if __name__ == "__main__":  # 직접 실행할 때만 프로그램을 시작한다.
    main()  # 시작 함수를 호출한다.
