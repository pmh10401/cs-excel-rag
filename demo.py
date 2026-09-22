"""로컬 LM Studio RAG 실행. Windows 실행은 windows-demo/demo.py입니다."""
from pathlib import Path
from ragkit.local_cli import main

if __name__ == "__main__":
    main(Path(__file__).resolve().parent)
