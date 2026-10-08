"""Windows 배포 ZIP을 만들고 압축 해제된 폴더에서 오프라인 검사를 실행합니다."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import os
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    assets = ['demo.py', 'evaluate.py', 'test_demo.py', 'sample_data.xlsx', 'realistic_data.xlsx',
              'realistic_cases.json', 'evaluation_cases.json', 'column_rules.json', 'config.json',
              'requirements.txt', 'README.txt', 'MANUAL.html', 'setup.bat', 'check.bat', 'start.bat',
              'cslist_schema.json', 'cslist_sample.xlsx', 'relation_sample.xlsx', 'cslist_cases.json', 'evaluate_search.py', 'workflow_cases.json']
    modules = ['__init__.py', 'records.py', 'llm.py', 'storage.py', 'search.py', 'report.py', 'cli.py',
               'evaluation.py', 'cslist.py', 'input_formats.py', 'menu.py', 'excel_app.py', 'web.py', 'english.py', 'source_cache.py', 'semantic.py', 'embedding_runtime.py',
               'retrieval_evaluation.py', 'relations.py', 'llm_trace.py', 'overnight.py', 'nli.py', 'claim_verification.py', 'assets/review.js', 'assets/theme.css', 'assets/app.html', 'assets/app.css', 'assets/app.js', 'assets/answer_prompt_en.txt', 'assets/tokenizer/fb374d419588a4632f3f557e76b4b70aebbca790']
    # 실제 사용자가 바꾼 서버·인증 설정을 배포 파일에 넣지 않습니다.
    files = [(ROOT / 'windows-demo' / ('config.example.json' if name == 'config.json' else name), name) for name in assets]
    files.append((ROOT / 'requirements-claim-verification.txt', 'requirements-claim-verification.txt'))
    files += [(ROOT / 'ragkit' / name, 'ragkit/' + name) for name in modules]
    files.append((ROOT / 'experiments/bilingual_normalize.py', 'experiments/bilingual_normalize.py'))
    files += [(ROOT / 'tests' / name, 'tests/' + name) for name in (
        '__init__.py', 'test_windows.py', 'test_cslist.py', 'test_excel_app.py', 'test_rag.py', 'test_headers.py',
        'test_ui.py', 'test_resume.py', 'test_quality.py', 'test_parallel.py', 'test_search_focus.py',
        'test_korean_search.py', 'test_bulk_reviews.py', 'test_english.py', 'test_semantic.py', 'test_relations.py',
        'test_source_cache.py', 'test_embedding_startup.py', 'test_workflow_auto.py', 'test_retrieval_evaluation.py',
        'test_overnight.py', 'test_multiuser.py', 'test_action_outcomes.py', 'test_incremental.py',
        'test_history_input.py', 'test_import_queue.py', 'test_detail_claims.py', 'test_nli.py', 'test_claim_verification.py', 'test_claim_verification_ui.py', 'test_relations_ui.cjs')]
    # 한국 시간 기준 생성 시각을 초 단위로 붙입니다. 파일 내용은 ZIP이며 풀기 전에 .zip으로 바꿉니다.
    timestamp = datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d-%H%M%S')
    output = ROOT / 'outputs' / f'RAG-Windows-CS128-{timestamp}.TXT'
    output.parent.mkdir(exist_ok=True)
    # 소스 파일만 명시적으로 묶습니다. DB·검토 데이터·API 응답·가상환경은 포함하지 않습니다.
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for source, name in files:
            archive.write(source, 'RAG-Windows-GPT4/' + name)
    with tempfile.TemporaryDirectory() as temporary, zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        archive.extractall(temporary)
        folder = Path(temporary) / 'RAG-Windows-GPT4'
        assert {path.name for path in folder.glob('*.bat')} == {'start.bat', 'setup.bat', 'check.bat'}
        assert b'demo.py --ui' in (folder / 'start.bat').read_bytes()
        assert b'--menu' not in (folder / 'start.bat').read_bytes()
        for batch in folder.glob('*.bat'):
            assert b'\n' not in batch.read_bytes().replace(b'\r\n', b''), batch.name
        env = dict(os.environ, PYTHONNOUSERSITE='1')
        env.pop('PYTHONPATH', None)
        for args in (['test_demo.py'], ['demo.py', '--search-only'], ['evaluate.py', '--help'],
                     ['demo.py', '--schema', 'cslist_schema.json', '--search-only'], ['demo.py', '--menu'], ['evaluate_search.py', '--help'],
                     ['-m', 'experiments.bilingual_normalize', '--config', 'config.json', '--output', 'outputs/bilingual-100', '--prepare-only'],
                     ['-m', 'ragkit.claim_verification', '--status']):
            # 2,000건 검토·번역의 DB 검사가 추가되어 전체 검사에만 더 긴 시간을 허용합니다.
            result = subprocess.run([sys.executable, *args], cwd=folder, env=env, input='0\n', capture_output=True, text=True,
                                    timeout=180 if args == ['test_demo.py'] else 60)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
        assert b'--upgrade pip' in (folder / 'setup.bat').read_bytes()
    print(f'PASS: 독립 압축 해제·오프라인 검사·배치 파일 형식\n{output}')


if __name__ == '__main__':
    main()
