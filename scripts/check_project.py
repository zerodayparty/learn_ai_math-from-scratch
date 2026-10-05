"""코드 설명, 계산 라이브러리 범위, 민감정보 패턴을 검사한다."""

from pathlib import Path  # 프로젝트 파일을 찾는다.
import ast  # 코드를 실행하지 않고 구조를 분석한다.
import io  # 코드 문자열을 토큰 분석기로 전달한다.
import json  # 노트북 내용을 읽는다.
import re  # 알려진 민감정보 형태를 찾는다.
import tokenize  # 코드와 주석을 정확히 구별한다.


def check_code(source, label):  # 한 코드 문서의 품질을 확인한다.
    """Docstring, 줄별 주석, import 범위, eig 검증 위치를 검사한다."""
    errors = []  # 발견한 문제를 모은다.
    tree = ast.parse(source)  # Python 구문을 검사하고 구조를 읽는다.
    doc_lines = set()  # 설명 문자열인 줄을 모은다.
    allowed = {"src", "numpy", "matplotlib", "pathlib", "json", "argparse", "unittest", "tempfile", "sys", "nbformat", "nbclient", "IPython", "ast", "io", "re", "tokenize"}  # 계산과 실행에 허용한 도구를 정한다.
    for node in ast.walk(tree):  # 모든 코드 구조를 확인한다.
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):  # 설명이 필요한 구조를 선택한다.
            doc = ast.get_docstring(node)  # 구조의 설명을 찾는다.
            if not doc and not isinstance(node, ast.Module):  # 함수와 클래스 설명이 없는지 확인한다.
                errors.append(f"{label}:{node.lineno}: missing Docstring")  # 상대 위치로 문제를 기록한다.
            if doc:  # 실제 설명 문자열 범위를 확인한다.
                expression = node.body[0]  # 맨 앞의 설명 문자열을 찾는다.
                doc_lines.update(range(expression.lineno, expression.end_lineno + 1))  # 설명 줄은 실행 코드 주석 검사에서 뺀다.
        if isinstance(node, (ast.Import, ast.ImportFrom)):  # 불러오는 라이브러리를 검사한다.
            names = [alias.name.split('.')[0] for alias in node.names] if isinstance(node, ast.Import) else [(node.module or '').split('.')[0]]  # 최상위 도구 이름을 찾는다.
            for name in names:  # 모든 불러오기 항목을 확인한다.
                if name not in allowed:  # 허용 범위 밖의 도구인지 확인한다.
                    errors.append(f"{label}:{node.lineno}: unexpected import {name}")  # 계산 도구 범위 문제를 기록한다.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"eig", "eigh", "eigvals", "eigvalsh"}:  # 고유값 계산 호출을 찾는다.
            if label != "src/experiments.py" and not label.startswith("tests/"):  # 결과 비교 검증 위치인지 확인한다.
                errors.append(f"{label}:{node.lineno}: eigen solver outside verification")  # 구현 중 우회 계산을 기록한다.
    comments = {token.start[0] for token in tokenize.generate_tokens(io.StringIO(source).readline) if token.type == tokenize.COMMENT}  # 주석이 있는 줄을 찾는다.
    for index, line in enumerate(source.splitlines(), 1):  # 코드 한 줄씩 확인한다.
        if line.strip() and not line.lstrip().startswith('#') and index not in doc_lines and index not in comments:  # 설명 없는 코드 줄을 찾는다.
            errors.append(f"{label}:{index}: missing easy line comment")  # 줄별 주석 누락을 기록한다.
    return errors  # 발견한 문제만 돌려준다.


def main():  # 프로젝트 전체 검사를 실행한다.
    """비밀 파일을 열지 않고 제출 코드와 문서, 노트북만 검사한다."""
    root = Path(__file__).resolve().parents[1]  # 프로젝트 위치를 찾는다.
    python_files = [root / "main.py"] + sorted((root / "src").glob("*.py")) + sorted((root / "scripts").glob("*.py")) + sorted((root / "tests").glob("*.py"))  # 실제 작성한 코드만 선택한다.
    paths = python_files + [root / "README.md", root / "pyproject.toml", root / "requirements.txt", root / "uv.lock", root / "Dockerfile", root / "compose.yaml", root / ".dockerignore"]  # 제출할 설정 파일을 선택한다.
    for folder in ["docs", "_practice", "_record"]:  # 작성한 설명 문서를 추가한다.
        paths += sorted((root / folder).glob("*.md"))  # Markdown 파일만 읽는다.
    errors = []  # 모든 검사 문제를 모은다.
    texts = []  # 보안 패턴을 검사할 문서를 모은다.
    for path in paths:  # 선택한 파일을 확인한다.
        if not path.exists():  # 필수 파일 누락을 확인한다.
            errors.append(f"Missing file: {path.relative_to(root)}")  # 상대 파일명으로 기록한다.
            continue  # 다음 파일을 확인한다.
        source = path.read_text(encoding="utf-8")  # 비밀 파일을 제외한 내용을 읽는다.
        label = str(path.relative_to(root))  # 개인 경로를 출력하지 않을 이름을 만든다.
        texts.append((label, source))  # 보안 검사 대상으로 추가한다.
        if path.suffix == ".py":  # Python 코드만 구조를 검사한다.
            errors += check_code(source, label)  # 코드 품질 검사를 한다.
    for path in sorted((root / "notebooks").glob("*.ipynb")):  # 학습 노트북을 확인한다.
        notebook = json.loads(path.read_text())  # 셀과 출력을 읽는다.
        label = str(path.relative_to(root))  # 상대 이름을 만든다.
        texts.append((label, json.dumps(notebook, ensure_ascii=False)))  # 출력까지 보안 검사에 포함한다.
        for index, cell in enumerate(notebook["cells"]):  # 모든 셀을 확인한다.
            if cell["cell_type"] == "code":  # 실행 코드를 검사한다.
                errors += check_code(''.join(cell["source"]), f"{label}:cell{index}")  # 노트북 코드의 주석과 라이브러리를 검사한다.
                if not cell.get("execution_count"):  # 출력이 없는 준비 셀도 실행 횟수로 확인한다.
                    errors.append(f"{label}:cell{index}: missing executed output")  # 미실행 셀을 기록한다.
    patterns = {"personal path": r"/(?:Users|home)/[^/\s]+/(?:ppp|\.codex|\.local|Library)", "API key pattern": r"sk-[A-Za-z0-9_-]{20,}", "AWS key pattern": r"AKIA[A-Z0-9]{16}", "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----"}  # 주요 민감정보 형태를 정한다.
    for label, source in texts:  # 코드와 설명, 저장된 출력을 검사한다.
        for name, pattern in patterns.items():  # 각 보안 패턴을 확인한다.
            if re.search(pattern, source):  # 민감정보 형태가 있는지 확인한다.
                errors.append(f"{label}: {name}")  # 내용을 노출하지 않고 위치와 종류만 기록한다.
    if errors:  # 문제가 있으면 성공으로 표시하지 않는다.
        print('\n'.join(errors))  # 값 대신 상대 위치만 출력한다.
        raise SystemExit(1)  # 검사 실패를 알린다.
    print(f"PASS: {len(python_files)} Python files, three executed notebooks, Docstrings and easy line comments")  # 확인 범위를 출력한다.
    print("PASS: allowed imports, eigen solver verification location, known sensitive patterns")  # 보안과 계산 범위 검사 결과를 출력한다.
    print("Scope: pattern scan only; secret files were not opened.")  # 보안 검사의 한계를 밝힌다.


if __name__ == "__main__":  # 직접 실행한 경우만 시작한다.
    main()  # 전체 검사 함수를 호출한다.
