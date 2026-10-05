# 🟩 검증 결과

## 🟢 1. 확인한 환경

2026년 10월 5일, Mac의 ARM64 환경에서 Python 3.11을 사용했다. NumPy 2.2.6, Matplotlib 3.10.3, 노트북 실행 도구는 pyproject.toml과 uv.lock에 고정했다. ARM64는 64-bit ARM architecture다.

Matplotlib 3.10.3과 새 pyparsing 조합의 제거 예정 API 경고를 피하려고 pyparsing 3.2.3을 uv 제약으로 고정했다. 수학 구현에 pyparsing을 직접 사용하지 않는다.

<br><br>

## 🟢 2. 실제 실행한 검사

| 검사 | 실행 또는 방법 | 결과 |
| --- | --- | --- |
| 단위·통합 테스트 | Python unittest discover | 26개 통과, 수정 후 재실행 통과 |
| 제공 이미지 CLI | main.py 기본 실행 | 63×64 흑백, 수치 판정 11개 통과 |
| 이미지 없는 환경의 CLI | --synthetic 별도 실행 | 64×64 생성 이미지, 판정 11개 통과 |
| 역전파 노트북 | 실제 Python 커널 | 코드 셀 4개 실행, 모든 손계산 4자리 비교 통과 |
| 확률·손실 노트북 | 실제 Python 커널 | 코드 셀 5개 실행, 분포·MLE·정보량 검사 통과 |
| 추가 수학 노트북 | 실제 Python 커널 | 코드 셀 3개 실행, 미분 오차 그림·튜닝·Newton 확인 |
| 반복 재현 | 동일 조건 main.py 재실행 | summary.json 전체 데이터 일치 |
| 이미지 선택 간 비교 | 두 실행의 JSON 비교 | 신경망·원형·타원형 최적화 결과 일치 |
| 정적 코드 검사 | AST·토큰·허용 import 검사 | Python 파일 13개와 노트북 코드 통과 |
| 보안 패턴 검사 | 작성 코드·문서·잠금 파일·저장 노트북 | 개인 경로·주요 키 형태·개인키 표식 발견 없음 |
| 요구사항 보존 | 수정 전후 SHA-256 비교 | 전체 파일 바이트의 해시 일치 |
| 시각 확인 | 생성 PNG 직접 열기 | 원·타원·전단·복원·분포·손실·미분 오차의 축과 범례 확인 |
| Docker Compose 설정 | --env-file /dev/null config --quiet | 설정 검증 통과 |
| Docker 서버 연결 | 서버 버전 읽기 시도 | 서버에 연결되지 않음 |

SHA = Secure Hash Algorithm(보안 해시 알고리즘). 원본 요구사항의 SHA-256은 다음과 같다.

```text
7d8fa95569ce6189c79f096ca333e7dd83115017cbb3b91f2ee1577abf474ee1
```

<br><br>

## 🟢 3. 중요한 수치 결과

| 항목 | 결과 | 해석 |
| --- | --- | --- |
| 지배 고유값 | 4.618033988749895 | 독립적인 2×2 특성방정식 및 NumPy 값과 일치 |
| 고유벡터 | [0.8506508083,0.5257311123] | 단위 길이, 부호를 무시해 방향 비교 |
| Power Iteration | 36회, 잔차 약 3.62e-10 | 반복 횟수는 초기 방향과 tol에 따라 달라짐 |
| f'(3) | 6.000000000039306 | 오차 약 3.93e-11 |
| BCE 손실 | 0.434958436851461 | 고정 예제 정답 1 기준 |
| 모든 매개변수 기울기 검사 | 최대 오차 약 3.70e-12 | 첫 행렬·편향·둘째 벡터·출력 편향 모두 확인 |
| 원형 GD 100회 최종 반경 | 1.44e-9 | 요구한 반경 0.1 이내 |
| 타원형 지속 진입 | GD 38회, beta=0.9 Momentum 80회 | 고정 설정에서는 Momentum이 더 느림 |
| 타원형 튜닝 Momentum | 11회부터 지속 진입 | 곡률 정보를 사용한 추가 설정 |
| Newton | 한 단계 원점 | 정확한 양의 정부호 이차 함수에 한정 |
| 제공 이미지 k=10 MSE | 약 9.30e-3 | 저장 실수 원소 수는 원본의 약 0.317배 |
| 제공 이미지 k=50 MSE | 약 7.35e-5 | 저장 원소 수는 원본의 약 1.587배 |
| 제공 이미지 k=100 | 실제 rank=63, MSE 약 5.29e-30 | 거의 완전 복원, 저장 원소 수는 2배 |

SVD 오차 감소 판정에는 절대 허용값 1e-14를 둔다. 생성 이미지처럼 이미 충분히 복원된 경우 1e-30 수준의 반올림 차이를 실패로 판단하지 않는다. 전체 rank 복원 MSE는 별도로 1e-20 미만인지 확인한다.

<br><br>

## 🟢 4. 다시 실행하는 방법

```sh
uv sync --locked
uv run --locked main.py
uv run --locked main.py --synthetic --output-dir outputs/generated
uv run --locked python -m unittest discover -s tests -v
uv run --locked python scripts/execute_notebooks.py
uv run --locked python scripts/check_project.py
```

| 요소 | 풀네임 또는 뜻 | 해설 |
| --- | --- | --- |
| uv | 공식 약자 풀네임이 없는 도구 이름 | Python 환경 관리 |
| sync | synchronize | uv.lock과 설치 환경 맞추기 |
| run | 실행 | 프로젝트 환경으로 실행 |
| --locked | 잠금 유지 | 버전 변경 방지 |
| --synthetic | 직접 생성 | 외부 이미지 없는 실행 |
| --output-dir | 출력 폴더 | 별도 결과 저장 |
| python -m | module | 모듈을 프로그램으로 실행 |
| unittest | unit test | Python 기본 검사 도구 |
| discover | 찾기 | 테스트 자동 검색 |
| -s | start directory | tests부터 검색 |
| -v | verbose | 결과 자세히 출력 |
| execute_notebooks.py | 노트북 실행 | 실제 커널에서 모든 셀 실행 |
| check_project.py | 정적 검사 | 코드 설명·도구 범위·민감정보 패턴 확인 |

<br><br>

## 🟢 5. 검증의 한계

- Dockerfile과 Compose는 작성했고 설정을 검사했다. Docker 서버 연결이 없어 실제 이미지 빌드와 컨테이너 실행은 확인하지 못했다.
- 노트북 커널은 로컬 소켓을 사용한다. 제한된 실행 환경에서는 소켓이 차단되어 실행 권한을 조정한 뒤 검증했다.
- 보안 검사는 알려진 문자열 패턴과 사용 코드 확인이다. 모든 종류의 비밀정보가 없다는 완전한 보장은 아니다. `.env`와 키 파일은 읽지 않았다.
- 사용자 제공 PNG의 출처·배포 권한은 확인하지 않았다. 원본을 Git에 추가하거나 외부로 전송하지 않았다.
- GitHub 원격 게시와 URL 제출을 수행하지 않았다. 로컬 파일과 실행 결과가 완료 범위다.
- 다른 운영체제, 다른 CPU, Python 버전 변경, 다른 이미지 입력 전체를 검증한 것은 아니다.
- Momentum·Adam의 속도 결과는 명시한 함수·설정·초기점에 대한 실험이다. 일반 신경망의 속도 우열을 입증하지 않는다.
