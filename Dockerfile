# 공식 Python 3.11 이미지를 사용한다.
FROM python:3.11-slim
# Python 실행 설정과 그림 캐시 위치를 정한다.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib
# 컨테이너 안에서 프로젝트 작업 위치를 정한다.
WORKDIR /app
# 의존성 목록만 먼저 복사한다.
COPY requirements.txt ./
# 공개 패키지를 버전 목록대로 설치한다.
RUN python -m pip install --no-cache-dir -r requirements.txt
# 실행에 필요한 코드만 복사한다.
COPY main.py ./
# 수학 구현 폴더만 복사한다.
COPY src/ ./src/
# 키와 원본 데이터 없이 생성 이미지로 실행한다.
CMD ["python", "main.py", "--synthetic"]
