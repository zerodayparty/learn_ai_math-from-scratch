"""명령 하나로 AI 수학 실험과 그림을 만든다."""

import argparse  # 터미널 실행 옵션을 읽는다.
from pathlib import Path  # 로컬 이미지 위치를 찾는다.
import matplotlib  # 화면 없는 환경의 그림 저장을 설정한다.

matplotlib.use("Agg")  # 그래프 창 없이 PNG를 저장한다.
from src.experiments import run_experiments  # 전체 수학 실험을 가져온다.


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
    for name, passed in report["checks"].items():  # 검증 항목을 순서대로 출력한다.
        print(f"{'PASS' if passed else 'FAIL'}: {name}")  # 실제 성공 여부를 표시한다.
    print("Generated PNG figures and summary.json.")  # 결과 생성을 알린다.


if __name__ == "__main__":  # 직접 실행할 때만 프로그램을 시작한다.
    main()  # 시작 함수를 호출한다.
