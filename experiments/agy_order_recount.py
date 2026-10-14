"""고정 Git 저장 결과만 재집계합니다. 모델·DB·자료 생성 코드를 호출하지 않습니다."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess

COMMIT = '330d0e4d1d4ffb08bd958eef9f115b472933c751'
SOURCE = 'docs/experiments/agy-breadth-20261014-r3/'
REPO = Path(__file__).resolve().parents[1]
KINDS = {'condition': '운전 조건', 'detail': '상세·작업', 'language': '다국어·표현',
         'numeric': '숫자·단위', 'state': '조치 상태', 'symptom': '현상·조치'}
SEPARATE = {'exact': '정확 조회·우회', 'aggregate': '집계·우회',
            'followup': '후속 질문·우회', 'absent': '기대ID 없음·실제 재정렬'}


def blob(path):
    return subprocess.check_output(['git', 'show', f'{COMMIT}:{path}'], cwd=REPO)


def grade(ids, expected):
    expected = set(expected)
    return {'top1': bool(ids) and ids[0] in expected,
            'hit3': bool(expected.intersection(ids[:3])),
            'hit5': bool(expected.intersection(ids)),
            'rank': next((i for i, identity in enumerate(ids, 1) if identity in expected), None)}


def compare(query, before, ranking, full_pool):
    after = ranking['ranked_ids']
    if Counter(before) != Counter(after):
        raise ValueError('Candidate membership changed: ' + query['id'])
    primary = query['category'] in KINDS and bool(query['expected_ids'])
    if primary and ranking['status'] != 'ok':
        raise ValueError('A primary question lacks successful actual reranking: ' + query['id'])
    a, b = grade(before, query['expected_ids']), grade(after, query['expected_ids'])
    assert a['hit5'] == b['hit5']
    result = {'query_id': query['id'], 'category': query['category'], 'group_id': query['group_id'],
              'primary': primary, 'status': ranking['status'], 'question': query['question'],
              'expected_ids': query['expected_ids'], 'before_ids': before, 'after_ids': after,
              'candidate_set_same': True, 'order_identical': before == after,
              'first_id_unchanged': before[:1] == after[:1]}
    if primary:
        result.update(before_top1=a['top1'], after_top1=b['top1'], before_hit3=a['hit3'], after_hit3=b['hit3'],
                      before_expected_rank=a['rank'], after_expected_rank=b['rank'],
                      top1_delta=int(b['top1'])-int(a['top1']), hit3_delta=int(b['hit3'])-int(a['hit3']),
                      candidate_missing_k5=not a['hit5'],
                      expected_in_full_pool=bool(set(query['expected_ids']).intersection(full_pool)))
    return result


def summarize(rows, name, label):
    return {'category': name, 'label': label, 'n': len(rows),
            **{key: sum(r[key] for r in rows) for key in
               ('before_top1', 'after_top1', 'before_hit3', 'after_hit3', 'candidate_missing_k5')},
            **{f'{metric}_{label}': sum(r[f'{metric}_delta'] == value for r in rows)
               for metric in ('top1', 'hit3') for label, value in [('improved', 1), ('worsened', -1), ('same', 0)]}}


def csv_file(path, records):
    fields = list(dict.fromkeys(key for row in records for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                             for key, value in row.items()})


def main(destination):
    def read(name):
        return json.loads(blob(SOURCE + name))
    manifest = read('manifest.json')
    tracked = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', COMMIT, '--', SOURCE], cwd=REPO).decode().splitlines()
    assert set(tracked) == {SOURCE + name for name in (*manifest, 'manifest.json')}
    for name, expected in manifest.items():
        content = blob(SOURCE + name)
        assert len(content) == expected['bytes'] and hashlib.sha256(content).hexdigest() == expected['sha256'], name
    protocol = read('protocol-public.json')
    code = ('experiments/agy_breadth.py', 'ragkit/semantic.py', 'ragkit/clef.py', 'experiments/agy_publish.py')
    code_hashes = {name: hashlib.sha256(blob(name)).hexdigest() for name in code}
    for name in code[:3]:
        assert code_hashes[name] == protocol['source_hashes'][name], name
    queries = read('dataset-public.json')['queries']
    assert len(queries) == len({q['id'] for q in queries}) == 264
    pools_list = read('candidate-pools.json')
    pools = {p['query_id']: p['ids'] for p in pools_list}
    assert len(pools) == len(pools_list) == 264
    rankings_list = read('ranking-outcomes.json')
    rankings = {(r['query_id'], r['K']): r for r in rankings_list}
    assert len(rankings) == len(rankings_list) == 792
    rows = [compare(q, pools[q['id']][:5], rankings[q['id'], 5], pools[q['id']])
            for q in sorted(queries, key=lambda q: q['id']) if q['split'] == 'TEST']
    primary = [r for r in rows if r['primary']]
    assert len(rows) == 200 and len(primary) == 120
    assert Counter(r['category'] for r in primary) == Counter({kind: 20 for kind in KINDS})
    groups = sorted({r['group_id'] for r in primary})
    assert len(groups) == 20 and all(sum(r['group_id'] == g for r in primary) == 6 for g in groups)
    summaries = [summarize(primary, 'ALL', '주 비교 합계')]
    summaries += [summarize([r for r in primary if r['category'] == k], k, label) for k, label in KINDS.items()]
    separate = [r for r in rows if not r['primary']]
    assert Counter(r['category'] for r in separate) == Counter({kind: 20 for kind in SEPARATE})
    audit = {'source_commit': COMMIT, 'source_directory': SOURCE, 'K': 5,
             'analysis': '저장 결과의 사후 재집계', 'inference_calls': 0, 'new_evaluation_data': 0,
             'source_manifest_sha256': hashlib.sha256(blob(SOURCE+'manifest.json')).hexdigest(),
             'verified_manifest_files': len(manifest), 'code_hashes': code_hashes,
             'primary_questions': 120, 'test_questions_preserved': 200, 'generation_groups': 20,
             'candidate_multiset_matches': sum(r['candidate_set_same'] for r in rows),
             'primary_order_identical': sum(r['order_identical'] for r in primary),
             'primary_first_id_unchanged': sum(r['first_id_unchanged'] for r in primary),
             'missing_k5_but_in_full_pool': sum(r['candidate_missing_k5'] and r['expected_in_full_pool'] for r in primary),
             'missing_even_in_full_pool': sum(r['candidate_missing_k5'] and not r['expected_in_full_pool'] for r in primary),
             'raw_candidates_or_gpu_requests_in_fixed_commit': False,
             'retrieval_order_basis': 'Frozen retrieval/runner code, direct ordered export, and saved input/execution audits; not a fresh retrieval replay'}
    destination.mkdir(parents=True, exist_ok=False)
    csv_file(destination/'questions.csv', rows)
    csv_file(destination/'comparison.csv', summaries + [
        {'category': k, 'label': label, 'n': 20} for k, label in SEPARATE.items()])
    csv_file(destination/'groups.csv', [{'group_id': g, **{
        key: value for key, value in summarize([r for r in primary if r['group_id'] == g], g, g).items()
        if key not in ('category', 'label')}} for g in groups])
    (destination/'provenance.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    originals = {r['_record_id']: r for r in read('synthetic-source-subset.json')}
    cases = ['# 고정 규칙으로 고른 사례', '', '저장 결과의 사후 재집계 · 후보5건 · 2026-10-15 KST', '',
             '각 구분에서 query_id 사전순으로 첫 문항을 선택한다. 효과가 가장 큰 사례를 고르지 않는다. 이 선택 규칙은 이번 사후 재집계에서 정했으며 추론 전 사전등록 규칙으로 표현하지 않는다.', '']
    for title, eligible in [('개선', [r for r in primary if r['top1_delta'] == 1]),
                            ('악화', [r for r in primary if r['top1_delta'] == -1]),
                            ('유지', [r for r in primary if r['top1_delta'] == 0]),
                            ('후보 누락 실패', [r for r in primary if r['candidate_missing_k5']])]:
        cases += [f'## {title}', '']
        if not eligible:
            cases += ['해당 문항은0/120건이다. 다른K의 사례를 가져오거나 새 사례를 만들지 않았다.', '']
            continue
        r = eligible[0]
        cases += [f"`{r['query_id']}` · 생성군 `{r['group_id']}`", '', r['question'], '',
                  f"첫 기대ID 순위: {r['before_expected_rank']} → {r['after_expected_rank']} (`None`은 후보5건에 없음)", '',
                  '전 순서: ' + json.dumps(r['before_ids'], ensure_ascii=False), '',
                  '후 순서: ' + json.dumps(r['after_ids'], ensure_ascii=False), '',
                  '기대ID: ' + json.dumps(r['expected_ids'], ensure_ascii=False), '']
        for identity in dict.fromkeys(r['before_ids'][:1]+r['after_ids'][:1]+r['expected_ids']):
            source = originals[identity]
            cases += [f'`{identity}`', '', '\n'.join(f"- {key}: {source.get(key, '')}" for key in
                        ('설비', '기인 Unit(부품)', '작업처', '현상', '조치')), '']
    (destination/'cases.md').write_text('\n'.join(cases).rstrip()+'\n')
    total = summaries[0]
    text = ['# 기존 검색 순서와 Clef 순서의 전후 비교', '', '2026-10-15 KST · **저장 결과의 사후 재집계**', '',
            f'입력은 고정 커밋 `{COMMIT}`의 `{SOURCE}`만 사용했다. 해당 커밋의 생성 코드는 출처 확인을 위해 읽기만 했다. 신규 추론·검색·임베딩·평가 데이터 생성·정답 수정은0회다. 운영 DB·설정과 원래 결과를 변경하지 않았다.', '',
            '## 적용 전 순서의 확인', '',
            f'- [검색 결과 저장 코드](https://github.com/pmh10401/cs-excel-rag/blob/{COMMIT}/experiments/agy_breadth.py#L142-L149)는 `semantic.search_cases(... mode="hybrid", limit=20)`의 반환 hits를 그대로 고정한다.',
            f'- [검색 함수](https://github.com/pmh10401/cs-excel-rag/blob/{COMMIT}/ragkit/semantic.py#L385-L406)는 RRF 결합·중복 처리·기존 조치 다양성 처리를 거쳐 selected를 반환한다. 기준은 **hybrid로 설정한 기존 검색 함수의 최종 반환 순서**다. BM25 단독·벡터 단독을 별도 실행한 대조군이 아니다.',
            f'- [공개 내보내기](https://github.com/pmh10401/cs-excel-rag/blob/{COMMIT}/experiments/agy_publish.py#L33-L38)는 저장 hits를 순회해 ids를 만들고 정렬하지 않는다. Clef는 같은 hits의 앞5건을 받아 재정렬한다.',
            '- 고정 manifest25파일과 검색·실행기·Clef 소스의 고정 프로토콜 해시를 대조했다. 저장 input/execution 감사는 입력 지문 일치와 후보 불변을 기록한다. 이 코드·기록에 근거해 ids를 해당 실험의 Clef 적용 전 검색 순서로 확인했다.',
            '- 검색 함수에는 정확 조회·벡터 실패의 별도 반환 경로도 있다. 공개 후보 파일에는 질문별 검색 details가 없어 모든 문항에 RRF가 실제 적용됐는지는 독립 확인하지 않았다. 함수의 반환 순서를 기준으로 비교하며 이를 순수 RRF 순서라고 확대하지 않는다.',
            '- 고정 커밋에는 원래 candidates.jsonl·candidate-freeze.json·GPU 원시 요청은 들어 있지 않다. 원시 검색 실행을 독립 재현하거나 공개 내보내기를 원시 파일과 새로 바이트 대조한 것은 아니다. 저장 감사 기록을 새로운 독립 검증으로 표현하지 않는다.', '',
            '## 후보5건 전후 비교표', '',
            '첫 기대ID는 첫 후보가 고정 실버 기대ID 중 하나인 문항 수다. 상위3은 기대ID가 하나 이상 포함된 문항 수다. 각 개선/악화/동일은 그 지표의 이진값 변화이며 순서 전체의 동일을 뜻하지 않는다. 후보 누락도 주 분모에 포함했다.', '',
            '|구분|문항|첫 기대ID 전→후|상위3 전→후|첫 개선/악화/동일|상위3 개선/악화/동일|후보5 누락|',
            '|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        text.append(f"|{r['label']}|{r['n']}|{r['before_top1']}→{r['after_top1']}|{r['before_hit3']}→{r['after_hit3']}|{r['top1_improved']}/{r['top1_worsened']}/{r['top1_same']}|{r['hit3_improved']}/{r['hit3_worsened']}/{r['hit3_same']}|{r['candidate_missing_k5']}|")
    for label in SEPARATE.values():
        text.append(f'|별도 {label}|20|—|—|—|—|대상 아님|')
    text += ['', f"주 비교120문항은6유형×20이며 모두 실제 재정렬 status=ok다. 후보 집합/개수 불일치는0/120, fallback·우회는0/120이다. 신규 TEST 전체200문항도 후보 다중집합200/200 일치를 확인했고 CSV에 모두 남겼다. 정확 조회·집계·후속은 각각20문항 우회이며, 근거 없음20문항은 실제 재정렬됐지만 기대ID가 없어 주 순위 분모에서 분리했다. DEV40·회귀24는 이번 주 분석에서 제외했다.", '',
             f"첫 기대ID는 {total['before_top1']}/120(60.00%) → {total['after_top1']}/120(94.17%), 관찰 차이+34.17%p다. 상위3은113/120(94.17%)으로 전후 동일하다. 기대ID가 후보에 있는113문항 중 기존 첫 기대ID72건이 Clef 후113건이지만, 이 조건부113분모로 전체120분모를 대체하지 않는다.", '',
             f"후보5 누락7건은 상세2·언어5다. 그중 {audit['missing_k5_but_in_full_pool']}건은 저장 전체 후보20건 안에는 있으나 앞5건 밖에 있고, {audit['missing_even_in_full_pool']}건은 전체 저장 후보20건에도 없다. 재정렬은 후보에 없는 기대ID를 추가할 수 없다. Clef 후 후보 내 기대ID를 첫 자리에 두지 못한 실패는0건이고, 남은 첫 선택 실패7건은 모두 후보 누락이다.", '',
             f"첫 지표 동일79건은 기대ID 첫 선택 유지72·기대ID 누락 유지7건이다. 실제 후보 순서 전체가 동일한 수는{audit['primary_order_identical']}/120, 첫 후보ID가 그대로인 수는{audit['primary_first_id_unchanged']}/120이며 별도 개념이다.", '',
             '## 생성군과 한계', '',
             '120문항은20개 생성군에서 각6유형이 파생된 합성/실버 자료다. 질문별 CSV에 group_id를 유지하고 그룹별 집계20행도 제공한다. 독립 현장 표본120개나 담당자 평가로 세지 않는다. 이번 사후 표의 문항 비율을 모집단 정답률·통계적 일반화·새 독립 검증으로 해석하지 않는다. 라벨의 의미 타당성을 재판정하지 않았으며 모호한 상태 표현·직접적 설비명이 있는 합성 질문이라는 기존 한계도 그대로다.', '',
             '## 발표에서 사용할 수 있는 결론', '',
             '> 고정한 합성 평가의 실제 재정렬 대상120문항에서, 같은 후보5건의 순서만 비교한 사후 재집계 결과 첫 기대 근거 선택은72건에서113건으로 늘었다. 개선41건·악화0건·지표 동일79건이었다. 상위3 포함은113건으로 같았고, 기대 근거가 후보에 없는7건은 해결되지 않았다.', '',
             '이는 기존 혼합 검색 대비 저장 순위 변화다. 후보 수 확대 비교나 최종 답변 정확도 비교가 아니다. 기존에도 기대 근거가 상위3 안에 있었으므로 답변 모델이 받는 최대3건의 품질/최종 답변 향상을 이 수치만으로 주장하지 않는다. 범용 재정렬 모델과의 새 대조·현장 효과·Windows 성능·Clef 고유 우위는 확인하지 않았다.', '',
             '## 결과와 재현', '',
             '[전후 비교표 CSV](comparison.csv) · [질문별200행](questions.csv) · [생성군20행](groups.csv) · [고정 선택 사례](cases.md) · [출처 검증 기록](provenance.json)', '',
             '저장소 루트에서 아래 명령으로 같은 고정 커밋을 읽어 다른 새 출력 폴더에 재집계한다. 표준 라이브러리와 로컬 Git 객체만 사용하며 모델을 호출하지 않는다.', '',
             '```sh', 'python3 -m experiments.agy_order_recount --output /tmp/agy-order-recount-check', '```', '']
    (destination/'README.md').write_text('\n'.join(text))
    print(json.dumps(total, ensure_ascii=False))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output.resolve())
