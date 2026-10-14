"""공개 수치만 재집계하며 모델 실행·원문 검사·의미 판정은 하지 않습니다."""
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
read = lambda name: json.loads((ROOT / name).read_text(encoding='utf-8'))
csv_rows = lambda name: list(csv.DictReader((ROOT / name).read_text(encoding='utf-8-sig').splitlines()))
data = read('synthetic-data.json')
queries = {q['id']: q for q in data['queries']}
records = {r['_record_id']: r for r in data['records']}
ranks = read('ranking-results.json')
main = {(r['query_id'], r['arm']): r for r in ranks if r['variant'] == 'main'}
assert len(queries) == 36 and len(records) == 38 and len(ranks) == 192 and len(main) == 180
assert len({(r['query_id'], r['arm'], r['variant']) for r in ranks}) == len(ranks)
for r in ranks:
    assert set(r['ranked_ids']) <= records.keys()
    if r['arm'] in ('G', 'HC'):
        assert Counter(r['ranked_ids']) == Counter(main[(r['query_id'], 'H')]['ranked_ids'])
for row in csv_rows('summary.csv'):
    selected = [r for r in main.values() if r['arm'] == row['arm'] and queries[r['query_id']]['category'] == row['category']]
    assert len(selected) == int(row['completed'])
    for field, calc in {
        'top1_direct': lambda r, q: bool(r['ranked_ids']) and r['ranked_ids'][0] in q['expected_ids'],
        'hit_at_5': lambda r, q: bool(set(r['ranked_ids'][:5]) & set(q['expected_ids'])),
        'top1_condition_contradiction': lambda r, q: bool(r['ranked_ids']) and r['ranked_ids'][0] in q['contradicting_ids'],
        'fallback': lambda r, q: r['status'] == 'fallback',
        'empty_returns': lambda r, q: not r['ranked_ids'],
    }.items():
        assert sum(calc(r, queries[r['query_id']]) for r in selected) == int(row[field]), (row['arm'], field)
for row in csv_rows('paired_conditions.csv'):
    pairs = {q['pair_id'] for q in queries.values() if q['pair_id']}
    correct = sum(all(main[(q['id'], row['arm'])]['ranked_ids'] and main[(q['id'], row['arm'])]['ranked_ids'][0] in q['expected_ids']
                      for q in queries.values() if q['pair_id'] == pair) for pair in pairs)
    assert len(pairs) == int(row['completed_pairs']) == 12 and correct == int(row['both_correct'])
for row in csv_rows('selection_changes.csv'):
    q = queries[row['query_id']]
    before = main[(q['id'], 'H')]['ranked_ids'][0] in q['expected_ids']
    after = main[(q['id'], row['arm'])]['ranked_ids'][0] in q['expected_ids']
    expected = 'improved' if after and not before else 'worsened' if before and not after else 'tie'
    assert row['change'] == expected
for row in csv_rows('order_sensitivity.csv'):
    rotated = next(r for r in ranks if r['query_id'] == row['query_id'] and r['variant'] == row['variant'] and r['arm'] == 'HC')
    original = main[(row['query_id'], 'HC')]
    assert (rotated['ranked_ids'][:1] == original['ranked_ids'][:1]) == (row['same_top1_id'] == 'True')
answers = read('answer-outcomes.json')
assert len(answers) == 24 and sum(a['actual_answer_calls'] for a in answers) == 31
assert len({(a['query_id'], a['arm']) for a in answers}) == 24
assert {a['query_id'] for a in answers} == set(read('protocol-public.json')['A']['answer_ids'])
for row in csv_rows('answer_summary.csv'):
    chosen = [a for a in answers if a['arm'] == row['arm']]
    assert len(chosen) == int(row['completed']) == 8
    for field, value in [('machine_fact_complete', True), ('machine_fact_incomplete_or_error', False), ('unknown', None)]:
        assert sum(a['independent_fact_diagnostic']['machine_checkable_complete'] is value for a in chosen) == int(row[field])
    assert sum(a['status'] == 'ok' for a in chosen) == int(row['delivery_ok'])
    assert sum(a['actual_answer_calls'] for a in chosen) == int(row['answer_http_calls'])
    times = [a['timings']['total_wall'] for a in chosen]
    for field, value in [('full_wall_median', statistics.median(times)), ('full_wall_min', min(times)), ('full_wall_max', max(times))]:
        assert math.isclose(value, float(row[field]))
controls = read('validator-outcomes.json')
for row in csv_rows('B_summary.csv'):
    normal = [a for a in controls if a['kind'] == row['kind'] and a['variant'] == 'normal']
    errors = [a for a in controls if a['kind'] == row['kind'] and a['variant'] == 'injected']
    assert len(normal) == len(errors) == 3
    assert sum(a['ON_accept'] for a in normal) == int(row['normal_kept'])
    assert sum(not a['ON_accept'] for a in errors) == int(row['injected_blocked'])
print('PASS: public ranking counts, pairs, candidate invariance, answer diagnostics and injected-control counts agree.')
print('Scope: recount only; no model inference, source-cell validation, independent labels or field accuracy evaluation.')
