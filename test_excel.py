"""기존 Excel 검사 명령을 유지합니다."""
from pathlib import Path
from tests.test_excel import main

if __name__ == "__main__":
    main(Path(__file__).resolve().parent)
