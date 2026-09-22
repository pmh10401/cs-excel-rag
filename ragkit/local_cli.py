"""기존 로컬 명령을 유지하는 LM Studio 실행부."""
import argparse
from contextlib import closing
import json
import os
import sqlite3
import time
from pathlib import Path
from openai import OpenAI
from .records import read_excel as read_workbook
from .vectors import FIELDS, ingest, retrieve, answer
from .local_report import render


def read_excel(path, sheet_name='점검기록'):
    return read_workbook(path, sheet_name, required_columns=('id',) + FIELDS, allow_typed=False)

def main(root, argv=None):
    # 프로그램 시작점입니다. --self-check가 있으면 검사만 하고 정상 실행은 건너뜁니다.
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--excel', type=Path, default=root / 'outputs' / '01a083b4' / 'sample_data.xlsx', help='읽을 .xlsx 파일 경로')
    parser.add_argument('--sheet', default='점검기록', help='첫 행에 열 이름이 있는 시트')
    args = parser.parse_args(argv)
    if args.self_check:
        from tests.test_local import main as self_check
        self_check(root)
        return
    # Excel 파일의 각 행을 딕셔너리로 읽습니다. 이후 검색·저장 코드는 기존 흐름을 재사용합니다.
    rows = read_excel(args.excel, args.sheet)
    print(f'Excel 읽기 완료: {args.excel.name} / {args.sheet} / {len(rows)}건', flush=True)
    # 127.0.0.1은 이 PC입니다. OpenAI 서버가 아닌 LM Studio에 연결합니다.
    # 인증을 켰다면 LM_STUDIO_API_KEY 환경변수에 실제 토큰을 설정해야 합니다.
    # 인증이 없으면 'lm-studio'는 SDK에 전달하는 형식상의 값입니다.
    # timeout=240은 요청 대기 제한(초), max_retries=0은 실패 시 자동 재시도하지 않는 설정입니다.
    client = OpenAI(base_url='http://127.0.0.1:1234/v1', api_key=os.getenv('LM_STUDIO_API_KEY', 'lm-studio'), timeout=240, max_retries=0)
    # 화면과 결과 JSON에 공통으로 사용할 자료를 모읍니다.
    report = {'created_at': time.strftime('%Y-%m-%d %H:%M:%S %Z'), 'source_file': args.excel.name, 'source_sheet': args.sheet, 'rows': rows, 'cases': []}
    # DB 파일이 없으면 만듭니다. closing은 성공·실패와 관계없이 연결을 닫습니다.
    # 저장의 확정과 되돌리기는 ingest() 내부의 트랜잭션이 담당합니다.
    with closing(sqlite3.connect(root / 'demo.sqlite3')) as db:
        print('1/3 원본 저장 및 로컬 임베딩 생성', flush=True)
        report['embedded_rows'] = ingest(client, db, rows)
        # 차원은 벡터에 들어 있는 숫자 개수입니다. 값을 가정하지 않고 DB에서 읽습니다.
        report['dimensions'] = len(json.loads(db.execute('SELECT embedding FROM records LIMIT 1').fetchone()[0]))
        # 질문과 기대 근거 ID를 묶습니다. None은 답을 뒷받침할 정보가 없다는 뜻입니다.
        # 기대 ID는 사람이 결과를 비교하기 위한 표시이며 검색이나 LLM 입력에는 쓰지 않습니다.
        questions = [('냉각기 A의 냉각수를 보충했는데도 유량이 낮습니다. 무엇을 확인해야 하나요?', 'DEMO-002'), ('컨베이어 C에서 제품을 감지하지 못할 때 어떻게 조치하나요?', 'DEMO-004'), ('냉각기 A의 필터를 몇 개월마다 교체해야 하나요?', None)]
        # 모든 질문을 먼저 검색한 뒤 답변을 생성합니다. LM Studio가 모델을 자동 교체할 때
        # 질문마다 임베딩 모델과 대화 모델을 번갈아 로드하는 비용을 줄이기 위한 순서입니다.
        prepared = []
        for question, expected in questions:
            # monotonic은 시스템 시각 조정에 영향받지 않아 경과 시간 측정에 적합합니다.
            started = time.monotonic()
            hits = retrieve(client, db, question)
            print(f'검색: {question} -> {[h["id"] for h in hits]}', flush=True)
            prepared.append((question, expected, hits, time.monotonic()-started))
        print('3/3 검색된 근거로 로컬 답변 생성', flush=True)
        for question, expected, hits, search_seconds in prepared:
            # monotonic은 시스템 시각 조정에 영향받지 않아 경과 시간 측정에 적합합니다.
            started = time.monotonic()
            reply, context = answer(client, question, hits)
            # 시간은 검색과 답변 생성을 합한 값이며 모델을 로드하는 시간이 포함될 수 있습니다.
            report['cases'].append(dict(question=question, expected=expected, hits=hits, answer=reply, context=context, seconds=search_seconds+time.monotonic()-started))
            # 질문 하나가 끝날 때마다 결과를 저장해, 이후 요청이 실패해도 앞의 결과는 남깁니다.
            (root / 'demo_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(reply, flush=True)
        # 세 질문이 모두 완료되면 최종 HTML을 만듭니다.
        render(report, root / 'demo.html')
    print('완료: demo.html / demo_results.json / demo.sqlite3', flush=True)
