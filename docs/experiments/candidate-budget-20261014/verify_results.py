"""공개 결과의 분모·순위·시간만 재집계하며 모델과 의미 판정은 재실행하지 않습니다."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
read = lambda name: json.loads((ROOT / name).read_text(encoding='utf-8'))
csv_rows = lambda name: list(csv.DictReader((ROOT / name).read_text(encoding='utf-8-sig').splitlines()))
data = read('dataset-public.json')
queries = {q['id']: q for q in data['queries']}
pools = {r['query_id']: r['ids'] for r in read('candidate-pools.json')}
runs = read('ranking-results.json')
assert len(queries) == 24 and len(pools) == 24 and len(runs) == 72
assert len(data['queries']) == len(queries)
assert {(r['query_id'], r['K']) for r in runs} == {(q, k) for q in queries for k in (5, 10, 20)}
assert all(len(ids) == len(set(ids)) == 20 for ids in pools.values())
def grade(q, ids):
    expected = set(q['expected_ids'])
    return {'top1': bool(ids) and ids[0] in expected, 'hit3': bool(set(ids[:3]) & expected), 'hitK': bool(set(ids) & expected)}
for row in runs:
    assert row['slice'] == queries[row['query_id']]['slice']
    assert Counter(row['ranked_ids']) == Counter(pools[row['query_id']][:row['K']])
    assert row['metrics'] == grade(queries[row['query_id']], row['ranked_ids'])
    if row['status'] == 'ok':
        assert row['execution']['backend'] == 'mlx' and row['execution']['device'] == 'gpu'
        assert row['execution']['state_truncated'] is False
summary_rows = csv_rows('ranking-summary.csv')
assert len(summary_rows) == 6 and {(r['slice'], int(r['K'])) for r in summary_rows} == {(s, k) for s in ('new_synthetic', 'prior_v3_regression') for k in (5, 10, 20)}
for row in summary_rows:
    selected = [r for r in runs if r['slice'] == row['slice'] and r['K'] == int(row['K'])]
    assert len(selected) == int(row['completed']) == 12
    for field in ('top1', 'hit3', 'hitK'):
        assert sum(r['metrics'][field] for r in selected) == int(row[field])
    assert sum(r['status'] == 'fallback' for r in selected) == int(row['fallback'])
    assert math.isclose(statistics.median(r['wall_seconds'] for r in selected), float(row['rerank_wall_median']))
    executed = [r['execution'] for r in selected if r['execution']]
    for field, original in [('load_median', 'load_seconds'), ('inference_median', 'inference_seconds')]:
        assert math.isclose(statistics.median(r[original] for r in executed), float(row[field]))
    assert max(r['peak_memory_bytes'] for r in executed) == int(row['MLX_inference_peak_bytes'])
answers = read('answer-outcomes.json')
assert len(answers) == 18 and len({(r['query_id'], r['K']) for r in answers}) == 18
assert {r['query_id'] for r in answers} == set(data['answer_ids'])
assert {(r['query_id'], r['K']) for r in answers} == {(q, k) for q in data['answer_ids'] for k in (5, 10, 20)}
assert all(len(r['source_record_ids']) <= 3 for r in answers)
assert all(0 <= r['http_calls'] <= 2 for r in answers)
answer_summary = csv_rows('answer-summary.csv')
assert len(answer_summary) == 3 and {int(r['K']) for r in answer_summary} == {5, 10, 20}
for row in answer_summary:
    selected = [r for r in answers if r['K'] == int(row['K'])]
    assert len(selected) == int(row['completed']) == 6
    assert sum(r['status'] == 'ok' for r in selected) == int(row['delivered'])
    assert sum(bool(r['http_calls']) for r in selected) == int(row['reached_answer_model'])
    assert sum(not r['http_calls'] for r in selected) == int(row['pre_answer_failure'])
    assert sum(r['cleanup_failed'] for r in selected) == int(row['cleanup_failure'])
    for field, value in [('machine_fact_complete', True), ('machine_fact_incomplete', False), ('unknown', None)]:
        assert sum(r['independent_fact_diagnostic']['machine_checkable_complete'] is value for r in selected) == int(row[field])
    assert sum(r['http_calls'] for r in selected) == int(row['http_calls'])
    assert sum(len(r['call_metadata']) for r in selected) == int(row['http_calls'])
    assert sum(reason == 'length' for r in selected for c in r['call_metadata'] for reason in c['finish_reasons']) == int(row['length_finish_calls'])
    for field in ('prompt_tokens', 'completion_tokens'):
        assert sum(c['usage'].get(field, 0) for r in selected for c in r['call_metadata']) == int(row[field])
    assert math.isclose(statistics.median(r['timings']['total_wall'] for r in selected), float(row['full_wall_median']))
by_key = {(r['query_id'], r['K']): r for r in runs}
changes = csv_rows('changes.csv')
assert len(changes) == 48 and {(r['query_id'], int(r['K'])) for r in changes} == {(q, k) for q in queries for k in (10, 20)}
for row in changes:
    before, after = by_key[(row['query_id'], 5)], by_key[(row['query_id'], int(row['K']))]
    change = 'improved' if after['metrics']['top1'] and not before['metrics']['top1'] else 'worsened' if before['metrics']['top1'] and not after['metrics']['top1'] else 'tie'
    assert row['change'] == change
    missing = not grade(queries[row['query_id']], pools[row['query_id']][:5])['hitK']
    assert row['input_missing5_presentK'] == str(missing and after['metrics']['hitK'])
    assert row['outside5_rescued_top1'] == str(missing and after['metrics']['top1'])
assert sum(r['http_calls'] for r in answers) <= 36
matched = [q for q in data['answer_ids'] if all(any(r['query_id'] == q and r['K'] == k and r['http_calls'] and not r['cleanup_failed'] for r in answers) for k in (5, 10, 20))]
matched_summary = csv_rows('matched-answer-summary.csv')
assert len(matched_summary) == 3 and {int(r['K']) for r in matched_summary} == {5, 10, 20}
for row in matched_summary:
    selected = [r for r in answers if r['query_id'] in matched and r['K'] == int(row['K'])]
    assert len(selected) == int(row['matched_queries'])
    assert sum(r['status'] == 'ok' for r in selected) == int(row['delivered'])
    assert sum(r['http_calls'] for r in selected) == int(row['http_calls'])
    assert math.isclose(statistics.median(r['timings']['total_wall'] for r in selected), float(row['wall_median']))
cost = read('cost-summary.json')
assert cost['answer_calls'] == sum(r['http_calls'] for r in answers)
assert cost['matched_answer_queries'] == matched
assert cost['answer_model_reached_flows'] == sum(bool(r['http_calls']) for r in answers)
assert math.isclose(cost['answer_flow_wall_sum'], sum(r['timings']['total_wall'] for r in answers))
assert math.isclose(cost['standalone_ranking_wall_sum'], sum(r['wall_seconds'] for r in runs))
manifest = read('manifest.json')
assert set(manifest) == {p.name for p in ROOT.iterdir() if p.is_file() and p.name != 'manifest.json'}
for name, value in manifest.items():
    path = ROOT / name
    assert path.stat().st_size == value['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == value['sha256']
print('PASS:72 rankings,18 answer attempts, actual model/paired counts, table denominators/times and public hashes')
print('Scope: recount only; not inference, independent source labels, semantic correctness or Windows/field validation')
