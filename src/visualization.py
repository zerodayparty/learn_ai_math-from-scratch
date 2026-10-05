"""실험 결과를 PNG 그림으로 저장하는 시각화 함수들."""

from pathlib import Path  # 출력 폴더 위치를 다룬다.
import numpy as np  # 그래프에 쓸 좌표를 만든다.
import matplotlib.pyplot as plt  # 그림을 그리는 도구를 가져온다.
from src.linear_algebra import area_check, rotation, scaling, shear, svd_compress, transform, unit_circle  # 행렬 실험을 가져온다.
from src.probability import bernoulli_pmf, normal_pdf  # 확률 계산을 가져온다.


def save_figure(figure, destination):  # 그림을 저장하고 메모리를 비운다.
    """상위 폴더를 만들고 PNG를 저장한 뒤 Figure를 닫는다."""
    destination = Path(destination)  # 파일 위치를 다루기 쉽게 바꾼다.
    destination.parent.mkdir(parents=True, exist_ok=True)  # 출력 폴더가 없으면 만든다.
    figure.tight_layout()  # 글씨와 축이 겹치지 않게 정리한다.
    figure.savefig(destination, dpi=150)  # 읽기 쉬운 해상도로 저장한다.
    plt.close(figure)  # 반복 실행 시 그림 메모리가 쌓이지 않게 한다.


def plot_transformations(destination):  # 세 변환을 한 그림에 비교한다.
    """단위 원, 방향 기준선, 격자를 변환 전후에 겹쳐 그린다."""
    circle = unit_circle()  # 원 위의 점을 만든다.
    gx, gy = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9))  # 격자 좌표를 만든다.
    grid = np.stack([gx.ravel(), gy.ravel()])  # 격자 점들을 두 줄로 모은다.
    matrices = {"Rotation pi/4": rotation(np.pi / 4), "Scaling (2, 0.5)": scaling(), "Shear k=1": shear()}  # 세 변환을 준비한다.
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))  # 세 그래프를 나란히 놓는다.
    checks = {}  # 면적 검증 결과를 모은다.
    for axis, (name, matrix) in zip(axes, matrices.items()):  # 변환별로 그림을 만든다.
        moved = transform(matrix, circle)  # 원을 변환한다.
        moved_grid = transform(matrix, grid)  # 격자도 같은 방식으로 변환한다.
        axis.plot(*np.column_stack([circle, circle[:, 0]]), color="royalblue", label="Before")  # 원을 닫아 그린다.
        axis.plot(*np.column_stack([moved, moved[:, 0]]), color="crimson", linestyle="--", label="After")  # 변환 결과를 겹친다.
        axis.scatter(*moved_grid, s=5, alpha=0.2, color="crimson")  # 변환된 격자를 표시한다.
        axis.plot([0, 1], [0, 0], color="royalblue")  # 회전을 구별할 원래 방향을 표시한다.
        axis.plot([0, matrix[0, 0]], [0, matrix[1, 0]], color="crimson")  # 변환된 방향을 표시한다.
        checks[name] = area_check(matrix)  # 면적 변화를 검사한다.
        axis.set_title(f"{name}\n|det|={abs(checks[name]['determinant']):.2f}, area ratio={checks[name]['area_ratio']:.2f}")  # 행렬식과 면적비를 보여준다.
        axis.set_aspect("equal")  # 가로세로를 같은 비율로 그린다.
        axis.grid(alpha=0.2)  # 위치를 비교할 격자를 넣는다.
        axis.legend()  # 색깔의 뜻을 표시한다.
    save_figure(figure, destination)  # 결과 그림을 저장한다.
    return checks  # 수치 검증 결과를 돌려준다.


def plot_svd(image, destination):  # 요청한 세 rank를 비교한다.
    """원본과 k=10,50,100 복원을 같은 밝기 범위로 그린다."""
    figure, axes = plt.subplots(1, 4, figsize=(14, 4))  # 네 이미지를 나란히 놓는다.
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)  # 원본 흑백 이미지를 그린다.
    axes[0].set_title(f"Original {image.shape}")  # 원본 크기를 표시한다.
    info_list = []  # rank별 오차와 저장량을 모은다.
    for axis, k in zip(axes[1:], [10, 50, 100]):  # 요구된 k 값을 순회한다.
        restored, info = svd_compress(image, k)  # 이미지를 압축하고 복원한다.
        info_list.append(info)  # 복원 정보를 저장한다.
        axis.imshow(restored, cmap="gray", vmin=0, vmax=1)  # 같은 밝기 기준으로 그린다.
        axis.set_title(f"k={k}, effective={info['effective_rank']}\nMSE={info['mse']:.2e}, storage={info['storage_ratio']:.2f}x")  # 실제 rank와 저장량을 표시한다.
    for axis in axes:  # 모든 이미지의 좌표 눈금을 숨긴다.
        axis.axis("off")  # 그림을 크게 보이게 한다.
    save_figure(figure, destination)  # 비교 그림을 저장한다.
    return info_list  # 화질 정보를 반환한다.


def plot_gradient(destination):  # 등고선과 기울기를 그린다.
    """f=x²+y²의 기울기와 등고선 접선을 표시한다."""
    x, y = np.meshgrid(np.linspace(-3, 3, 120), np.linspace(-3, 3, 120))  # 배경 좌표를 만든다.
    figure, axis = plt.subplots(figsize=(6, 6))  # 정사각형 그림을 준비한다.
    axis.contour(x, y, x ** 2 + y ** 2, levels=[1, 2, 4, 6, 9, 12], cmap="Blues")  # 같은 함수값의 선을 그린다.
    px, py = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))  # 화살표 위치를 만든다.
    axis.quiver(px, py, 2 * px, 2 * py, color="crimson", angles="xy", scale_units="xy", scale=6)  # 기울기를 빨간 화살표로 그린다.
    axis.quiver([1], [1], [-1], [1], color="green", angles="xy", scale_units="xy", scale=2)  # (1,1)에서 접선 방향을 그린다.
    axis.set(title="Gradient (red) perpendicular to contour tangent (green)", xlabel="x", ylabel="y", aspect="equal")  # 축과 방향의 뜻을 표시한다.
    save_figure(figure, destination)  # 기울기 그림을 저장한다.


def plot_paths(paths, anisotropy, destination):  # 최적화 경로를 등고선에 올린다.
    """여러 옵티마이저 경로를 점선으로 같은 Figure에 겹친다."""
    extent = max(6.0, max(float(np.max(abs(path))) for path in paths.values()) * 1.1)  # 모든 경로가 보이는 범위를 정한다.
    x, y = np.meshgrid(np.linspace(-extent, extent, 150), np.linspace(-extent, extent, 150))  # 등고선 좌표를 만든다.
    figure, axis = plt.subplots(figsize=(7, 6))  # 경로 그림을 준비한다.
    axis.contour(x, y, x ** 2 + anisotropy * y ** 2, levels=20, cmap="Greys", alpha=0.5)  # 손실 등고선을 그린다.
    for name, path in paths.items():  # 모든 경로를 겹쳐 그린다.
        axis.plot(path[:, 0], path[:, 1], ".--", markersize=3, label=name)  # 시작부터 끝까지 점선으로 잇는다.
    axis.scatter([0], [0], marker="*", color="black", s=100, label="Minimum")  # 최솟값 위치를 표시한다.
    axis.set(title=f"f(x,y)=x^2+{anisotropy:g}y^2", xlabel="x", ylabel="y", aspect="equal")  # 함수와 좌표를 표시한다.
    axis.legend()  # 옵티마이저 이름을 표시한다.
    save_figure(figure, destination)  # 경로 비교 그림을 저장한다.


def plot_losses(losses, destination, title):  # 수렴 속도를 그래프로 비교한다.
    """0을 작은 양수로 표시하여 로그 축에서 손실을 비교한다."""
    figure, axis = plt.subplots(figsize=(8, 5))  # 손실 그래프를 만든다.
    for name, values in losses.items():  # 각 실험의 손실을 그린다.
        axis.semilogy(np.maximum(values, 1e-30), label=name)  # 작은 손실도 보이게 로그 축을 쓴다.
    axis.set(title=title, xlabel="Iteration", ylabel="Loss (log scale, floor 1e-30)")  # 축의 의미를 설명한다.
    axis.grid(alpha=0.2)  # 비교할 격자를 넣는다.
    axis.legend()  # 실험 이름을 표시한다.
    save_figure(figure, destination)  # 손실 그래프를 저장한다.


def plot_distributions(destination):  # 정규분포와 베르누이 분포를 비교한다.
    """두 PDF를 왼쪽에, 두 PMF를 오른쪽에 함께 표시한다."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))  # 두 분포 종류를 나란히 놓는다.
    x = np.linspace(-4, 5, 500)  # 밀도를 비교할 위치를 만든다.
    axes[0].plot(x, normal_pdf(x, 0, 1), label="N(0,1): variance=1")  # 첫 번째 밀도를 그린다.
    axes[0].plot(x, normal_pdf(x, 2, 0.5), label="N(2,0.5): variance=0.5")  # 두 번째 밀도를 그린다.
    axes[0].set(title="Normal PDF", xlabel="x", ylabel="Density")  # 확률 자체가 아닌 밀도임을 표시한다.
    axes[1].bar(np.array([0, 1]) - 0.18, bernoulli_pmf(0.3), width=0.36, label="B(0.3)")  # 첫 베르누이 분포를 그린다.
    axes[1].bar(np.array([0, 1]) + 0.18, bernoulli_pmf(0.7), width=0.36, label="B(0.7)")  # 두 번째 분포를 옆에 그린다.
    axes[1].set(title="Bernoulli PMF", xlabel="Outcome", ylabel="Probability", xticks=[0, 1], ylim=(0, 1))  # 두 결과와 확률을 표시한다.
    for axis in axes:  # 모든 그래프에 범례를 넣는다.
        axis.legend()  # 분포 이름을 보여준다.
    save_figure(figure, destination)  # 분포 그림을 저장한다.
