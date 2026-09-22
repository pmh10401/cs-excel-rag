"""Windows 평가 실행 진입점."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if not (ROOT / 'ragkit').is_dir():
    sys.path.insert(0, str(ROOT.parent))
from ragkit.evaluation import main

if __name__ == '__main__':
    main(ROOT)
