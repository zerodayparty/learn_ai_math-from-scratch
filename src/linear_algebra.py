"""행렬 변환, 거듭제곱 반복법, 특이값 분해를 계산한다."""

from pathlib import Path  # 이미지 파일의 위치를 다룬다.
import numpy as np  # 모든 수학 계산에 NumPy를 사용한다.
from src.common import finite_array, positive_integer, positive_number  # 입력 검사 도구를 가져온다.


def unit_circle(count=100):  # 단위 원 위의 점을 만든다.
    """열마다 한 점을 담은 (2, count) 단위 원을 반환한다."""
    positive_integer(count, "count")  # 점 개수가 양의 정수인지 확인한다.
    if count < 3:  # 다각형 면적에는 최소 세 점이 필요하다.
        raise ValueError("At least three points are required.")  # 점이 부족함을 알린다.
    theta = np.linspace(0, 2 * np.pi, count, endpoint=False)  # 원을 겹치지 않는 각도로 나눈다.
    return np.stack([np.cos(theta), np.sin(theta)])  # 가로와 세로 좌표를 쌓는다.


def rotation(theta):  # 회전 행렬을 만든다.
    """라디안 theta만큼 반시계 방향으로 회전하는 행렬을 반환한다."""
    finite_array([theta])  # 각도가 유한한지 확인한다.
    c, s = np.cos(theta), np.sin(theta)  # 각도의 코사인과 사인을 구한다.
    return np.array([[c, -s], [s, c]])  # 길이를 유지하는 회전 행렬을 만든다.


def scaling(sx=2.0, sy=0.5):  # 확대와 축소 행렬을 만든다.
    """가로 sx배, 세로 sy배로 바꾸는 행렬을 반환한다."""
    finite_array([sx, sy])  # 배율이 유한한지 확인한다.
    return np.diag([sx, sy])  # 각 축의 배율을 대각선에 넣는다.


def shear(k=1.0):  # 전단 행렬을 만든다.
    """x 좌표를 x + k*y로 바꾸는 전단 행렬을 반환한다."""
    finite_array([k])  # 기울이는 값이 유한한지 확인한다.
    return np.array([[1.0, k], [0.0, 1.0]])  # 높이에 비례하여 옆으로 민다.


def transform(matrix, points):  # 점들에 행렬을 적용한다.
    """(2, 2) 행렬로 (2, n) 점 배열을 변환한다."""
    matrix, points = finite_array(matrix, 2), finite_array(points, 2)  # 행렬과 점 배열을 검사한다.
    if matrix.shape != (2, 2) or points.shape[0] != 2:  # 두 축의 변환인지 확인한다.
        raise ValueError("Expected a (2, 2) matrix and (2, n) points.")  # 크기 오류를 알린다.
    return matrix @ points  # 각 점에 같은 행렬을 곱한다.


def polygon_area(points):  # 점들이 둘러싼 면적을 구한다.
    """신발끈 공식으로 (2, n) 다각형의 절댓값 면적을 구한다."""
    points = finite_array(points, 2)  # 좌표를 검사한다.
    if points.shape[0] != 2 or points.shape[1] < 3:  # 다각형에 필요한 모양인지 확인한다.
        raise ValueError("Expected (2, n) points with n >= 3.")  # 잘못된 점 배열을 알린다.
    x, y = points  # 가로 좌표와 세로 좌표를 나눈다.
    return float(abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))) / 2)  # 이웃한 점의 외적을 합한다.


def area_check(matrix, count=100):  # 행렬식과 실제 면적 변화를 비교한다.
    """면적비, 행렬식, 상대 오차를 반환하며 특이 행렬은 절대 오차를 쓴다."""
    circle = unit_circle(count)  # 기준 원의 점들을 만든다.
    ratio = polygon_area(transform(matrix, circle)) / polygon_area(circle)  # 변환 전후 면적의 비를 구한다.
    determinant = float(np.linalg.det(matrix))  # 행렬식으로 면적 배율을 구한다.
    expected = abs(determinant)  # 면적은 방향과 관계없이 양수다.
    error = abs(ratio - expected) / expected if expected > 0 else abs(ratio)  # 반사와 면적 0도 처리한다.
    return {"determinant": determinant, "area_ratio": ratio, "error": error}  # 비교 결과를 돌려준다.


def power_iteration(matrix, tol=1e-10, max_iter=10000, initial=None):  # 고유벡터 방향을 반복하여 찾는다.
    """실수 대칭 행렬의 절댓값이 가장 큰 고유쌍을 잔차로 검증한다.

    유일한 지배 고유값과 그 방향 성분이 있는 시작 벡터가 필요하다.
    양의 정부호 예제에서는 이 고유값이 수치상 최대 고유값이다.
    동률이나 잘못된 시작 방향은 수렴하지 않거나 다른 고유쌍을 줄 수 있다.
    """
    matrix = finite_array(matrix, 2)  # 입력 행렬을 검사한다.
    if matrix.shape[0] != matrix.shape[1] or not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12):  # 정사각 대칭 행렬만 허용한다.
        raise ValueError("A real symmetric square matrix is required.")  # 알고리즘의 학습 범위를 알린다.
    positive_number(tol, "tol")  # 잔차 허용값을 확인한다.
    positive_integer(max_iter, "max_iter")  # 최대 반복 횟수를 확인한다.
    vector = np.random.default_rng(42).normal(size=matrix.shape[0]) if initial is None else finite_array(initial, 1).copy()  # 시작 방향을 정한다.
    if vector.shape != (matrix.shape[0],) or np.linalg.norm(vector) == 0:  # 벡터 크기와 영벡터 여부를 확인한다.
        raise ValueError("A nonzero compatible initial vector is required.")  # 시작 방향 오류를 알린다.
    vector /= np.linalg.norm(vector)  # 길이를 1로 만든다.
    scale = max(1.0, float(np.linalg.norm(matrix)))  # 행렬 크기에 맞는 잔차 기준을 만든다.
    for iteration in range(1, max_iter + 1):  # 허용 횟수만큼 방향을 개선한다.
        product = matrix @ vector  # 행렬로 방향을 늘이거나 뒤집는다.
        norm = np.linalg.norm(product)  # 새 벡터 길이를 잰다.
        if norm == 0:  # 영행렬이나 영공간 시작 방향을 구별한다.
            if np.any(matrix):  # 영행렬이 아니면 지배 방향을 찾을 수 없다.
                raise ValueError("Initial vector lies in the null space.")  # 다른 시작 벡터가 필요함을 알린다.
            return 0.0, vector, iteration, 0.0  # 영행렬의 고유값 0을 돌려준다.
        vector = product / norm  # 길이를 1로 되돌린다.
        value = float(vector @ matrix @ vector)  # 레일리 몫으로 고유값을 추정한다.
        residual = float(np.linalg.norm(matrix @ vector - value * vector))  # Av = lambda*v의 오차를 구한다.
        if residual <= tol * scale:  # 충분히 정확한 고유쌍인지 확인한다.
            vector *= 1 if vector[np.argmax(abs(vector))] >= 0 else -1  # 고유벡터 부호를 일정하게 맞춘다.
            return value, vector, iteration, residual  # 값과 방향, 반복 횟수, 오차를 돌려준다.
    raise RuntimeError("Power iteration did not converge; check spectral gap and initial direction.")  # 수렴 실패를 숨기지 않는다.


def synthetic_image(size=64):  # 외부 파일 없이 예제 이미지를 만든다.
    """직접 만든 원, 물결, 그라데이션을 섞은 흑백 이미지를 반환한다."""
    positive_integer(size, "size")  # 크기가 양의 정수인지 확인한다.
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))  # 모든 픽셀의 위치를 만든다.
    image = 0.25 * (x + 1) + 0.2 * np.sin(18 * (x * x + y * y)) + 0.4 * (x * x + y * y < 0.5)  # 직접 무늬를 그린다.
    return (image - image.min()) / (image.max() - image.min())  # 밝기를 0부터 1까지로 맞춘다.


def load_grayscale(path=None, size=64):  # 이미지 읽기와 전처리를 한다.
    """이미지를 흰 배경에 합성하고 NumPy 이중선형 보간으로 size 이하로 줄인다."""
    positive_integer(size, "size")  # 크기 설정을 확인한다.
    if size > 64:  # 과제에서 정한 크기 한계를 지킨다.
        raise ValueError("The assignment image size must be <= 64.")  # 크기 제한을 알린다.
    if path is None:  # 파일이 지정되지 않으면 재현 가능한 예제를 쓴다.
        return synthetic_image(size)  # 직접 만든 이미지를 돌려준다.
    from matplotlib.image import imread  # 시각화 도구의 이미지 읽기 기능만 가져온다.
    raw = imread(Path(path))  # 사용자가 지정한 로컬 이미지를 읽는다.
    image = raw.astype(float) / 255 if raw.dtype == np.uint8 else raw.astype(float)  # PNG 밝기를 실수로 맞춘다.
    if image.ndim == 3:  # 컬러 이미지이면 흑백으로 바꾼다.
        rgb = image[..., :3]  # 빨강, 초록, 파랑을 분리한다.
        if image.shape[2] == 4:  # 투명도 채널이 있으면 흰 배경과 합친다.
            rgb = rgb * image[..., 3:4] + (1 - image[..., 3:4])  # 투명한 부분을 흰색으로 만든다.
        image = rgb @ np.array([0.2126, 0.7152, 0.0722])  # 세 색의 밝기를 가중 평균한다.
    image = finite_array(image, 2)  # 흑백 행렬을 검사한다.
    height, width = image.shape  # 기존 가로세로 크기를 확인한다.
    factor = min(1.0, size / max(height, width))  # 긴 쪽이 64 이하가 되게 배율을 정한다.
    ys = np.linspace(0, height - 1, max(1, round(height * factor)))  # 새 세로 좌표를 정한다.
    xs = np.linspace(0, width - 1, max(1, round(width * factor)))  # 새 가로 좌표를 정한다.
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)  # 가까운 왼쪽 위 픽셀을 찾는다.
    y1, x1 = np.minimum(y0 + 1, height - 1), np.minimum(x0 + 1, width - 1)  # 오른쪽 아래 픽셀을 찾는다.
    wy, wx = (ys - y0)[:, None], (xs - x0)[None, :]  # 네 픽셀을 섞을 비율을 구한다.
    top = image[np.ix_(y0, x0)] * (1 - wx) + image[np.ix_(y0, x1)] * wx  # 위쪽 두 픽셀을 섞는다.
    bottom = image[np.ix_(y1, x0)] * (1 - wx) + image[np.ix_(y1, x1)] * wx  # 아래쪽 두 픽셀을 섞는다.
    return top * (1 - wy) + bottom * wy  # 위와 아래 밝기를 섞는다.


def svd_compress(image, k):  # 중요한 특이값만 남겨 이미지를 복원한다.
    """복원 행렬과 저장 원소 수를 반환하며 k를 가능한 rank로 제한한다."""
    image = finite_array(image, 2)  # 흑백 행렬을 검사한다.
    positive_integer(k, "k")  # 남길 특이값 개수를 확인한다.
    u, s, vh = np.linalg.svd(image, full_matrices=False)  # 행렬을 세 조각으로 나눈다.
    rank = min(k, len(s))  # 행렬 크기를 넘는 요청을 실제 가능한 개수로 제한한다.
    restored = (u[:, :rank] * s[:rank]) @ vh[:rank]  # A_k = U_k*Sigma_k*V_k^T로 복원한다.
    stored = rank * (image.shape[0] + image.shape[1] + 1)  # 저장할 실수의 총 개수를 센다.
    info = {"requested_k": k, "effective_rank": rank, "stored_values": stored, "storage_ratio": stored / image.size, "mse": float(np.mean((image - restored) ** 2))}  # 화질과 저장량을 기록한다.
    return restored, info  # 복원 이미지와 설명을 돌려준다.


def plot_transformations(destination):  # 선형대수 모듈에서 변환 그림을 실행한다.
    """세 행렬 변환과 면적비를 PNG로 저장한다."""
    from src.visualization import plot_transformations as plot  # 그림 코드를 필요할 때 가져온다.
    return plot(destination)  # 공통 그림 함수에 작업을 맡긴다.


def plot_svd(image, destination):  # 선형대수 모듈에서 이미지 비교를 실행한다.
    """k=10,50,100의 이미지 복원을 PNG로 저장한다."""
    from src.visualization import plot_svd as plot  # 그림 코드를 필요할 때 가져온다.
    return plot(image, destination)  # 같은 그림 구현을 재사용한다.
