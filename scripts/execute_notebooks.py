"""노트북을 실제 Python 커널에서 실행하고 검증된 출력을 저장한다."""

from pathlib import Path  # 프로젝트 파일 위치를 찾는다.
import sys  # 실행 도구의 Python 환경을 사용한다.
import nbformat  # 노트북 형식을 읽고 검사한다.
from nbclient import NotebookClient  # 실제 커널에서 셀을 실행한다.


def main():  # 노트북 실행을 시작한다.
    """세 노트북의 모든 셀을 실행하며 실패하면 오류를 전파한다."""
    root = Path(__file__).resolve().parents[1]  # 사용자 경로를 출력하지 않고 루트를 찾는다.
    if sys.version_info[:2] != (3, 11):  # 프로젝트의 기준 Python 버전을 확인한다.
        raise RuntimeError("Use Python 3.11 with uv run.")  # 환경 오류를 알린다.
    for path in sorted((root / "notebooks").glob("*.ipynb")):  # 모든 학습 노트북을 선택한다.
        notebook = nbformat.read(path, as_version=4)  # 셀과 설명을 읽는다.
        for cell in notebook.cells:  # 이전 실행 결과를 초기화한다.
            if cell.cell_type == "code":  # 실행 셀만 정리한다.
                cell.outputs, cell.execution_count = [], None  # 오래된 출력을 비운다.
        client = NotebookClient(notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(root)}})  # 루트에서 실행할 실제 커널을 준비한다.
        client.execute()  # 모든 셀을 순서대로 실행하고 실패를 감지한다.
        for cell in notebook.cells:  # 실행 시간 같은 임시 정보는 제거한다.
            cell.metadata = {}  # 사용자 환경의 불필요한 메타데이터를 남기지 않는다.
        nbformat.validate(notebook)  # 저장할 파일의 Jupyter 형식을 검사한다.
        text = nbformat.writes(notebook)  # 파일에 쓸 내용을 만든다.
        if "/Users/" in text or "/home/" in text:  # 출력에 개인 절대 경로가 남았는지 검사한다.
            raise RuntimeError("Notebook contains a personal absolute path.")  # 민감한 출력 저장을 막는다.
        path.write_text(text, encoding="utf-8")  # 성공한 실행 결과를 저장한다.
        print("PASS:", path.name, "executed cells:", sum(cell.cell_type == "code" for cell in notebook.cells))  # 파일명과 검증 범위만 출력한다.


if __name__ == "__main__":  # 직접 실행할 때만 시작한다.
    main()  # 노트북 검증 함수를 호출한다.
