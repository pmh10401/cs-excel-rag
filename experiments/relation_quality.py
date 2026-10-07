"""한국어 합성 2,000상세에서 결합 검색과 원문 관계 보완을 비교합니다. 운영 DB를 읽지 않습니다."""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import time
from ragkit import cslist, relations, semantic, storage
from ragkit.input_formats import HISTORY_COLUMNS, HISTORY_SHEET
from ragkit.records import record_id
from ragkit.retrieval_evaluation import atomic_report, freeze, score

EQUIPMENT = ['세정장비', '도포장비', '현상장비', '증착장비', '식각장비',
             '연마장비', '검사장비', '이송장비', '건조장비', '노광장비']
PARTS = [('회전모듈', '회전척'), ('회전모듈', '회전베어링'), ('배수모듈', '배수밸브'), ('배수모듈', '배수호스'),
         ('진공모듈', '진공게이지'), ('진공모듈', '흡착패드'), ('냉각모듈', '냉각밸브'), ('냉각모듈', '냉각호스'),
         ('이송모듈', '이송벨트'), ('이송모듈', '이송베어링'), ('분사모듈', '분사노즐'), ('분사모듈', '분사밸브'),
         ('가열모듈', '가열판'), ('가열모듈', '온도센서'), ('정렬모듈', '정렬센서'), ('정렬모듈', '정렬척'),
         ('배기모듈', '배기밸브'), ('배기모듈', '배기필터'), ('구동모듈', '구동벨트'), ('구동모듈', '구동베어링')]
SYMPTOMS = ['진동', '소음', '누수', '위치 이탈', '동작 지연']


def fixture():
    rows = []
    for e, equipment in enumerate(EQUIPMENT):
        for p, (module, part) in enumerate(PARTS):
            for n in range(10):
                row = {column: '' for column in HISTORY_COLUMNS} | {
                    'CS NO': f'PILOT-{e:02}-{p//2:02}-{n:02}', '제목': f'{part} 점검 기록',
                    '설비': equipment, '기인 모듈': module, '기인 Unit(부품)': part,
                    '현상': f'운전 중 {part} {SYMPTOMS[n % 5]} 발생',
                    '원인': '연결부와 설치 상태 점검 필요',
                    '조치': ['연결부 조정 후 정상화', '부품 교체 후에도 재발', '부품 교체 예정',
                             '설치 상태 점검, 결과 미기재', '연결부 재조립 후 일부 개선'][n % 5],
                    '불량발생일': f'2026-09-{n+1:02}', 'SITE': '가상사업장',
                    '_id_column': 'CS NO', '_source_sheet': HISTORY_SHEET, '_history_detail': True,
                    '_cell_locations': {column: f'가상자료!{column}:{len(rows)+2}'
                                        for column in ('설비', '기인 모듈', '기인 Unit(부품)', '현상', '조치')}}
                rows.append(row)
    cases = []
    for p, (module, part) in enumerate(PARTS):
        equipment, symptom = EQUIPMENT[p % 10], SYMPTOMS[p % 5]
        expected = [record_id(r) for r in rows if r['설비'] == equipment and r['기인 Unit(부품)'] == part
                    and symptom in r['현상']]
        cases.append(dict(id=f'relation-{p}', group='equipment_part',
                          question=f'{equipment}의 {part} {symptom} 과거 조치', expected_ids=expected))
    for p in range(6):
        module, part = PARTS[p]
        symptom = SYMPTOMS[p % 5]
        expected = [record_id(r) for r in rows if r['기인 모듈'] == module and r['기인 Unit(부품)'] == part
                    and symptom in r['현상']]
        cases.append(dict(id=f'cross-{p}', group='module_part',
                          question=f'{module} {part} {symptom} 다른 설비의 과거 조치', expected_ids=expected))
    for i, symptom in enumerate(SYMPTOMS):
        cases.append(dict(id=f'symptom-{i}', group='symptom', question=f'{symptom} 과거 조치',
                          expected_ids=[record_id(r) for r in rows if symptom in r['현상']]))
    for i, question in enumerate(('세정장비 회전척 11.8 bar 진동', '회전모듈 회전척 소음이 아닌 진동',
                                 '회전모듈 회전척 정상 조건', '회전모듈 회전척 교체 예정')):
        # 조건 정확도를 가정하지 않습니다. 두 방식의 후보가 같은지를 별도로 검사합니다.
        cases.append(dict(id=f'guard-{i}', group='guard', question=question, expected_ids=[]))
    for i in range(3):
        target = rows[i * 10]
        cases.append(dict(id=f'exact-{i}', group='exact', question='CS NO=' + target['CS NO'],
                          expected_ids=[record_id(r) for r in rows if r['CS NO'] == target['CS NO']]))
    for i, question in enumerate(('세정장비 없는부품 진동', '없는장비 회전척 누수', 'CS NO=PILOT-99-99-99')):
        cases.append(dict(id=f'negative-{i}', group='negative', question=question, expected_ids=[]))
    return rows, freeze({'provenance': 'Created synthetic Korean development cases; not independent field evaluation', 'cases': cases})


def preserved(db):
    data = db.execute('SELECT * FROM normalized_records ORDER BY id').fetchall()
    reviews = db.execute('SELECT * FROM cell_reviews ORDER BY record_id,column_name').fetchall()
    return hashlib.sha256(json.dumps([data, reviews], ensure_ascii=False).encode()).hexdigest()


def run(output, config_path):
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'pilot.sqlite3').exists():
        raise ValueError('기존 실험을 덮어쓰지 않습니다. 새 출력 폴더를 지정하세요.')
    rows, (dataset, signature) = fixture()
    atomic_report(output / 'dataset.json', dict(dataset=dataset, freeze_hash=signature, rows=rows))
    config = {'embedding': json.loads(config_path.read_text())['embedding']}
    atomic_report(output / 'protocol.json', dict(date='2026-10-07', rows=len(rows), freeze_hash=signature,
        models={k: v for k, v in semantic.settings(config).items() if k != 'api_key'},
        profile=semantic.profile(config), limit=3, source='Isolated synthetic DB; operating DB not opened',
        code_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     (Path(__file__), Path('ragkit/relations.py'), Path('ragkit/semantic.py'))}))
    schema, rules = cslist.load_schema(Path('windows-demo/cslist_schema.json'))
    with closing(sqlite3.connect(output / 'pilot.sqlite3')) as db:
        storage.ingest(db, rows, rules, schema=schema)
        before = preserved(db)
        started = time.monotonic()
        relations.ensure(db)
        prepare_seconds = time.monotonic() - started
        with closing(semantic.client_for(config)) as client:
            vectors = semantic.build_vectors(db, client, config, progress=lambda s: print(s, flush=True))
            if vectors['failures'] or vectors['indexed'] != 2000:
                raise ValueError('실험 DB의 벡터 준비가 끝나지 않았습니다.')
            results = []
            source = {record_id(r): r for r in rows}
            for i, case in enumerate(dataset['cases']):
                methods = {}
                for mode in (('hybrid', 'graph') if i % 2 == 0 else ('graph', 'hybrid')):
                    details, started = {}, time.monotonic()
                    hits = semantic.search_cases(db, case['question'], config, mode=mode,
                                                  embedding_client=client, details=details)
                    methods[mode] = dict(ids=[h['record_id'] for h in hits], metrics=score(case, hits),
                                         seconds=time.monotonic()-started, details=details)
                    for hit in hits:
                        assert all(value == source[hit['record_id']][column] for section in hit['evidence']
                                   for column, value in section['cells'].items())
                if case['group'] in ('guard', 'symptom', 'exact', 'negative'):
                    assert methods['hybrid']['ids'] == methods['graph']['ids'], case['id']
                results.append(dict(case=case, methods=methods))
                atomic_report(output / 'progress.json', dict(processed=len(results), total=len(dataset['cases'])))
                print(f"비교 {i+1}/{len(dataset['cases'])}: {case['id']}", flush=True)
        summary = {}
        for mode in ('hybrid', 'graph'):
            positives = [r['methods'][mode] for r in results if r['case']['expected_ids']]
            summary[mode] = dict(positive_count=len(positives), top1=sum(r['metrics']['top1'] for r in positives),
                hit_at_3=sum(r['metrics']['hit_at_3'] for r in positives),
                precision_at_3=statistics.mean(r['metrics']['precision_at_3'] for r in positives),
                mean_seconds=statistics.mean(r['methods'][mode]['seconds'] for r in results),
                relation_queries=sum(r['methods'][mode]['details']['vector_search'].get('relations', {}).get('status') == 'used' for r in results))
        assert preserved(db) == before
        atomic_report(output / 'results.json', dict(complete=True, processed=len(results), missing=0, errors=0,
            freeze_hash=signature, source_hash_before=before, source_hash_after=preserved(db),
            relations=db.execute('SELECT COUNT(*) FROM rag_relations').fetchone()[0],
            relation_prepare_seconds=prepare_seconds, vectors=vectors, summary=summary, cases=results,
            limits=['Synthetic development pilot', 'Retrieval only; no answer/translation/NLI model calls',
                    'No threshold tuning or default mode/model promotion', 'No Windows field verification']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('mac_config.json'))
    args = parser.parse_args()
    run(args.output, args.config)
