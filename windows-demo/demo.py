"""Windows 셀 검토 RAG 실행. 기존 demo.py 명령을 유지합니다."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if not (ROOT / 'ragkit').is_dir():
    sys.path.insert(0, str(ROOT.parent))
from ragkit.cli import main

if __name__ == '__main__':
    main(ROOT)
