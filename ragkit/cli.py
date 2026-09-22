"""Windows 데모 실행 순서만 관리합니다. 경로와 설정은 실행 시 명시적으로 읽습니다."""
import argparse
from contextlib import closing
import json
import sqlite3
import time
from pathlib import Path
from openai import OpenAI, APIError, APITimeoutError
from .records import read_excel, load_header_aliases
from .llm import load_rules, answer
from .storage import ingest, apply_reviews, prepared_record
from .search import retrieval_check, retrieve, count_records, compare_search
from .english import build_english, bilingual_retrieve
from . import semantic
from .report import render, render_comparison
from .evaluation import answer_checks

def apply_review_file(path, root, rules, schema=None):
    report_path = root / 'demo_results.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    with closing(sqlite3.connect(root / 'demo.sqlite3')) as db:
        prepared = apply_reviews(db, json.loads(path.read_text(encoding='utf-8-sig')), rules, schema=schema)
    report['normalized_records'] = prepared
    report['rows'] = [record['original_cells'] for record in prepared]
    # 이전 답변은 새 검색 근거에 대한 답변이 아니므로 화면에서 제거합니다.
    report['cases'] = []
    report['column_rules'] = rules
    report['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    render(report, root / 'demo.html')
    print('검토 결과를 DB에 반영했습니다. API 호출 없음. 답변은 원래 실행 파일로 다시 생성하세요.')


def main(root, argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--menu', action='store_true', help='CS 작업 메뉴 열기')
    parser.add_argument('--ui', action='store_true', help='로컬 웹 UI 열기')
    parser.add_argument('--host', help='웹 UI 접속용 서버 IPv4 주소. 기본값은 로컬 전용')
    parser.add_argument('--port', type=int, default=8765, help='로컬 웹 UI 포트')
    parser.add_argument('--config', type=Path, default=root / 'config.json', help='모델 서버 설정 JSON')
    parser.add_argument('--output', type=Path, help='결과 폴더. Mac과 Windows 실행 결과를 따로 보관할 때 사용')
    parser.add_argument('--schema', type=Path, help='CS 열 양식 JSON. CS 결과는 cslist-output에 저장')
    parser.add_argument('--excel', type=Path, help='입력 .xlsx 또는 변환된 .rag.jsonl.gz 파일')
    parser.add_argument('--convert-input', type=Path, help='Excel을 .rag.jsonl.gz로 변환하고 종료. 모델 호출 없음')
    parser.add_argument('--reader', choices=['openpyxl', 'xlwings'], default='openpyxl', help='일반 파일 직접 읽기 / Windows Excel에서 열린 파일 읽기')
    parser.add_argument('--sheet', help='CS 기본값은 CSList, 일반 데모 기본값은 점검기록')
    parser.add_argument('--apply-reviews', type=Path, help='검토 JSON을 적용하고 비교 화면 갱신. API 호출 없음')
    questions_group = parser.add_mutually_exclusive_group()
    questions_group.add_argument('--cases', type=Path, help='시험 질문과 사람이 검토할 기준을 담은 JSON 파일')
    questions_group.add_argument('--question', help='JSON 파일 없이 질문 하나 입력')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--import-only', action='store_true', help='CS 원문을 DB에 누적 등록. 모델 호출 없음')
    mode.add_argument('--search-only', action='store_true', help='API 없이 원문 검색만 확인')
    mode.add_argument('--normalize-only', action='store_true', help='LLM 정리와 비교 화면까지만 생성')
    mode.add_argument('--compare-search', action='store_true', help='저장된 원문과 원문+채택문의 검색을 비교. API 호출 없음')
    mode.add_argument('--translate-english', action='store_true', help='저장된 CS 자료의 영어 검색 자료 생성 / 이어서 실행')
    mode.add_argument('--build-vectors', action='store_true', help='CS 벡터 자료 생성 / 이어서 실행. 문장 다듬기 호출 없음')
    parser.add_argument('--vector-limit', type=int, default=20, help='새로 임베딩할 고유 문서 수. 기본 20, 0은 전체')
    parser.add_argument('--search-mode', choices=['keyword', 'vector', 'hybrid'], default='keyword', help='질문 검색 방식')
    parser.add_argument('--translation-limit', type=int, default=20, help='이번에 영어로 변환할 접수 수. 기본 20, 0은 전체')
    parser.add_argument('--bilingual', action='store_true', help='원문·영어 번역 질문을 함께 사용하고 한국어로 답변')
    parser.add_argument('--saved', action='store_true', help='Excel을 읽지 않고 등록된 DB 전체 사용. --normalize-only와 함께 사용 가능')
    args = parser.parse_args(argv)
    if args.convert_input and (args.import_only or args.menu or args.ui or args.apply_reviews or args.search_only or args.normalize_only or args.compare_search or args.build_vectors or args.search_mode != 'keyword' or args.saved or args.translate_english or args.bilingual or args.question or args.cases):
        parser.error('--convert-input은 다른 작업 옵션과 함께 사용할 수 없습니다.')
    if args.ui:
        if args.import_only or args.menu or args.apply_reviews or args.search_only or args.normalize_only or args.compare_search or args.build_vectors or args.search_mode != 'keyword' or args.saved or args.translate_english or args.bilingual or args.question is not None or args.cases is not None:
            parser.error('--ui는 다른 작업 옵션과 함께 사용할 수 없습니다.')
        from .web import serve
        serve(root, config_path=args.config, output_dir=args.output or root / 'cslist-ui-output',
              schema_path=args.schema or root / 'cslist_schema.json', port=args.port, host=args.host,
              excel=args.excel, reader=args.reader)
        return
    if args.menu:
        from .menu import main as menu
        menu(root, config_path=args.config, output_dir=args.output)
        return
    schema = None
    if args.schema:
        from .cslist import load_schema
        schema, rules = load_schema(args.schema)
    else:
        rules = load_rules(root / 'column_rules.json')
    output = args.output or (root / 'cslist-output' if schema else root)
    if args.import_only and (schema is None or args.saved or args.bilingual or args.question or args.cases):
        parser.error('--import-only에는 CS 양식이 필요하며 질문·--saved와 함께 사용할 수 없습니다.')
    if args.build_vectors:
        if schema is None or args.vector_limit < 0:
            parser.error('--build-vectors에는 CS 양식과 0 이상의 --vector-limit이 필요합니다.')
        args.saved = (output / 'demo.sqlite3').exists()
    if args.search_mode != 'keyword' and (schema is None or not args.saved or args.search_only or args.build_vectors or args.compare_search or args.translate_english):
        parser.error('--search-mode는 --saved로 저장된 CS 자료에 질문할 때 사용하세요.')
    if args.translate_english:
        args.saved = True
        if args.translation_limit < 0:
            parser.error('--translation-limit은 0 또는 양의 정수여야 합니다.')
    if args.bilingual and (schema is None or args.search_only or args.compare_search or args.normalize_only or args.translate_english or args.build_vectors):
        parser.error('--bilingual은 CS 자료의 질문에 사용하세요. 질문 영어 번역은 API를 호출합니다.')
    if (args.saved or args.compare_search) and (schema is None or args.search_only):
        parser.error('--saved와 --compare-search는 저장된 CS 자료 조회에 사용하세요.')
    if args.apply_reviews:
        apply_review_file(args.apply_reviews, output, rules, schema=schema)
        return
    args.excel = args.excel or root / ('cslist_sample.xlsx' if schema else 'sample_data.xlsx')
    sheet = args.sheet or ('CSList' if schema else '점검기록')
    header_aliases = None
    if schema is not None:
        header_aliases = load_header_aliases(
            output / 'column_aliases.json', schema['columns'], schema['id_column'], schema.get('name'))
    if args.convert_input:
        from .source_cache import convert_excel
        def conversion_progress(info):
            total = info['total_rows']
            rows = f"{info['processed_rows']:,}" + (f'/{total:,}' if total is not None else '')
            percent = f" · 현재 단계 {info['percent']}%" if info['percent'] is not None else ''
            print(f"{info['stage']}{percent} · 시트 {rows}행 · 변환 {info['completed']:,}건 · {info['elapsed_seconds']}초", flush=True)
        if args.reader == 'xlwings':
            print('Excel 셀을 편집 중이면 Enter로 확정하거나 Esc로 취소하고, 열린 대화상자를 처리하세요.', flush=True)
        result = convert_excel(args.excel, args.convert_input, sheet,
            required_columns=schema['columns'] if schema else None,
            id_column=schema['id_column'] if schema else 'id', reader=args.reader,
            header_aliases=header_aliases, progress=conversion_progress)
        print(f'변환 완료: {result["rows"]:,}건 → {result["path"]}', flush=True)
        return
    if args.saved or args.compare_search:
        if not (output / 'demo.sqlite3').exists():
            raise ValueError('등록된 DB가 없습니다. --import-only로 먼저 원문을 등록하세요.')
        with closing(sqlite3.connect(output / 'demo.sqlite3')) as saved_db:
            rows = [json.loads(raw) for (raw,) in saved_db.execute('SELECT original_json FROM normalized_records ORDER BY id')]
        if not rows:
            raise ValueError('등록된 DB에 원문이 없습니다.')
    else:
        rows = read_excel(args.excel, sheet, required_columns=schema['columns'] if schema else None,
                          id_column=schema['id_column'] if schema else 'id', reader=args.reader,
                          header_aliases=header_aliases)
    config = {} if args.import_only else json.loads(args.config.read_text(encoding='utf-8-sig'))
    if args.build_vectors:
        semantic.settings(config)
    print(f'입력 읽기: {len(rows)}건', flush=True)
    questions = ['냉각기 A의 냉각수를 보충했는데도 유량이 낮습니다. 무엇을 확인해야 하나요?', '컨베이어 C에서 제품을 감지하지 못할 때 어떻게 조치하나요?', '냉각기 A의 필터를 몇 개월마다 교체해야 하나요?']
    if args.question is not None:
        if not args.question.strip():
            parser.error('질문이 비어 있습니다.')
        questions = [args.question]
    elif schema and args.cases is None and not args.normalize_only and not args.translate_english and not args.import_only:
        args.cases = root / 'cslist_cases.json'
    # 검토 기준은 모델에 보내지 않습니다. 답변 후 화면에서만 비교합니다.
    checks = json.loads(args.cases.read_text(encoding='utf-8-sig')) if args.cases else []
    if args.cases:
        if not isinstance(checks, list) or not checks or any(
                not isinstance(c, dict) or not isinstance(c.get('question'), str) or not c['question'].strip()
                or not isinstance(c.get('review'), str) or not isinstance(c.get('expected_ids'), list)
                or type(c.get('expect_no_match', False)) is not bool
                or (c.get('expect_no_match') and c['expected_ids'])
                or any(not isinstance(i, str) for i in c['expected_ids']) for c in checks):
            raise ValueError('시험 파일에 question, review, expected_ids를 올바르게 입력하세요.')
        questions = [c['question'] for c in checks]
        for check in checks:
            answer_checks('', check)  # 모델 호출 전에 평가 규칙의 형식을 검사합니다.
    if args.normalize_only or args.translate_english or args.build_vectors or args.import_only:
        questions = []
    def make_client():
        return OpenAI(base_url=config['base_url'], api_key=config['api_key'], timeout=config['timeout_seconds'], max_retries=0)
    client = None
    if not args.search_only:
        output.mkdir(parents=True, exist_ok=True)
    if (args.saved or args.compare_search) and not (output / 'demo.sqlite3').exists():
        raise ValueError('저장된 자료가 없습니다. 먼저 2번 문장 다듬기를 실행하세요.')
    db = sqlite3.connect(':memory:' if args.search_only else output / 'demo.sqlite3')
    try:
        if (args.saved and not args.normalize_only) or args.compare_search:
            stored = {identity: (json.loads(raw), json.loads(proposal)) for identity, raw, proposal in db.execute(
                'SELECT id, original_json, normalized_json FROM normalized_records')}
            prepared = [prepared_record(db, row, stored[row[schema['id_column']]][1], rules) for row in rows]
            # 이전 버전 DB도 원문과 검토 선택을 유지하며 새 문서 색인을 만듭니다.
            from .cslist import ensure_search_index
            ensure_search_index(db, rules, schema)
            if args.translate_english:
                with closing(make_client()) as translation_client:
                    result = build_english(db, translation_client, config, schema,
                                           limit=args.translation_limit, progress=lambda text: print(text, flush=True))
                print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
                if result['failures']:
                    raise ValueError('일부 영어 변환이 실패했습니다. 성공 건은 저장했으며 같은 명령으로 이어서 실행할 수 있습니다.')
                return
            if args.compare_search:
                edits = [{'id': record['id'], 'column': key, 'original': record['original_cells'][key], 'accepted': value}
                         for record in prepared for key, value in record['effective'].items() if value != record['original_cells'][key]]
                comparison = {'created_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'source_file': 'DB 등록 자료' if args.saved or args.compare_search else args.excel.name,
                              'source_sheet': sheet, 'rows': len(rows), 'edits': edits,
                              'cases': compare_search(db, checks or [{'question': q} for q in questions])}
                (output / 'search_comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding='utf-8')
                render_comparison(comparison, output / 'search_comparison.html')
                print(f'검색 비교 완료. API 호출 없음: {output / "search_comparison.html"}')
                return
        else:
            client = None if args.search_only or args.build_vectors or args.import_only else make_client()
            incoming = {}
            prepared = ingest(db, rows, rules, client, config, schema=schema, append=schema is not None, import_result=incoming)
            if incoming:
                print(f"DB 등록: 추가 {incoming['added']}건 · 갱신 {incoming['updated']}건 · 동일 {incoming['unchanged']}건 · 누적 {incoming['total']}건", flush=True)
            if args.import_only:
                return
        if args.build_vectors:
            with closing(semantic.client_for(config)) as embedding_client:
                result = semantic.build_vectors(db, embedding_client, config, limit=args.vector_limit,
                                                progress=lambda text: print(text, flush=True))
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            if result['failures']:
                raise ValueError('일부 벡터 생성이 실패했습니다. 같은 명령으로 남은 자료를 이어서 처리하세요.')
            return
        searches = []
        for question in questions:
            aggregate = count_records(db, question) if schema else None
            detail = {}
            if args.bilingual and not aggregate:
                if client is None:
                    client = make_client()
                hits = bilingual_retrieve(db, question, client, config, details=detail,
                                          limit=20 if args.search_mode == 'hybrid' else 3,
                                          search_english=args.search_mode != 'vector')
            else:
                hits = retrieve(db, question) if not aggregate and args.search_mode == 'keyword' else []
            if args.search_mode != 'keyword' and not aggregate:
                hits = semantic.search_cases(db, question, config, mode=args.search_mode,
                    keyword_hits=hits if args.bilingual else None, details=detail)
            searches.append((question, hits, aggregate, detail))
    finally:
        db.close()
    for question, hits, aggregate, detail in searches:
        print(f"검색: {question} → {[hit['id'] for hit in hits]}", flush=True)
        if detail.get('search'):
            print(json.dumps(detail['search'], ensure_ascii=False), flush=True)
    if args.search_only:
        print('검색 검사 완료. API는 호출하지 않았습니다.', flush=True)
        return
    # 서버 주소와 추론 옵션은 config.json에서 읽습니다. API 키는 결과 보고서에 넣지 않습니다.
    report = {'created_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'source_file': 'DB 등록 자료' if args.saved or args.compare_search else args.excel.name,
              'source_sheet': rows[0]['_source_sheet'], 'model': config['model'], 'base_url': config['base_url'],
              'reasoning_effort': config['reasoning_effort'], 'column_rules': rules,
              'menu_command': config.get('menu_command', 'start.bat'),
              'reader': args.reader,
              'workflow': 'cslist' if schema else 'demo',
              'search_method': '접수 문서 SQLite FTS5 / BM25' if schema else '문자 조각 키워드 검색',
              'rows': rows, 'normalized_records': prepared, 'cases': []}
    # 답변 호출이 실패해도 이번 입력의 정리 화면은 남겨 두어 검토할 수 있습니다.
    (output / 'demo_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    render(report, output / 'demo.html')
    if args.normalize_only:
        print(f'한국어 정리 완료: {output / "demo.html"}에서 원문과 비교하세요.', flush=True)
        return
    for question, hits, aggregate, detail in searches:
        started = time.monotonic()
        error = None
        try:
            if aggregate:
                reply, context = aggregate['answer'], ''
            else:
                if client is None and hits:
                    client = make_client()
                reply, context = answer(client, config, question, hits, details=detail)
        except (ValueError, RuntimeError, APIError) as failure:
            # 거절된 모델 답변을 성공으로 보여 주지 않고 실패 사실과 검색 원문을 남깁니다.
            error = ('서버 응답 시간 초과입니다. 다른 모델 요청이 끝난 뒤 다시 질문하세요.'
                     if isinstance(failure, APITimeoutError) else str(failure))
            reply, context = '답변을 생성하지 못했습니다. 아래 원문과 오류를 확인하세요.', ''
        report['cases'].append({'question': question, 'hits': hits, 'answer': reply, 'context': context,
                               'seconds': time.monotonic()-started, 'quality': detail, 'aggregate': aggregate, 'error': error})
        if checks:
            # 이것은 기대한 자료가 검색됐는지 확인하는 검사이며 답변 정확도 점수가 아닙니다.
            check = checks[len(report['cases']) - 1]
            report['cases'][-1]['review'] = check['review']
            report['cases'][-1]['expected_ids'] = check['expected_ids']
            report['cases'][-1]['retrieval_pass'] = retrieval_check(hits, check)
            report['cases'][-1]['answer_checks'] = answer_checks(reply, check)
        (output / 'demo_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        render(report, output / 'demo.html')
        print(reply, flush=True)
    render(report, output / 'demo.html')
    if any(case['error'] for case in report['cases']):
        raise ValueError('일부 답변이 검사를 통과하지 못했습니다. 결과 화면에 원문과 오류를 저장했습니다.')
    if any(case.get('retrieval_pass') is False for case in report['cases']):
        raise ValueError('기대 자료의 검색 여부가 검사 기준과 다릅니다. 결과 화면의 검색 근거를 확인하세요.')
    if any(case.get('answer_checks', {}).get('passed') is False for case in report['cases']):
        raise ValueError('지정 표현 검사에서 답변 누락 또는 금지 표현을 발견했습니다. 결과 화면의 원문과 검토 기준을 확인하세요.')
    print(f'완료: {output / "demo.html"}', flush=True)
