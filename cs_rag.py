"""Mac의 LM Studio로 CS Excel 검토 RAG를 실행합니다. 인자가 없으면 웹 UI를 엽니다."""
from pathlib import Path
import sys
from ragkit.cli import main

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    # 양식과 가상 Excel은 기존 자료를 공유하고 모델 설정·결과는 Mac 경로를 사용합니다.
    extra = sys.argv[1:] or ['--ui']
    args = ['--schema', str(ROOT / 'windows-demo/cslist_schema.json'),
            '--config', str(ROOT / 'mac_config.json')]
    if '--output' not in extra:
        folder = 'outputs/mac-cslist-ui' if '--ui' in extra else 'outputs/mac-cslist'
        args += ['--output', str(ROOT / folder)]
    main(ROOT / 'windows-demo', args + extra)
