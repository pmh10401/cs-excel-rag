"""공개 확대 평가를 재집계합니다. 모델 추론·의미 판정은 실행하지 않습니다."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys


def grade(q, ids):
    gold=set(q['expected_ids'])
    return {'top1':bool(ids) and ids[0] in gold,'hit3':bool(gold&set(ids[:3])),
            'hitK':bool(gold&set(ids)),'contradiction_top1':bool(ids) and ids[0] in q.get('contradicting_ids',[])}


def check_rank(row, query, pool):
    before=pool[:row['K']]
    assert Counter(before)==Counter(row['ranked_ids']), 'Candidate membership changed'
    assert grade(query,before)['hitK']==grade(query,row['ranked_ids'])['hitK']


def check_manifest(root):
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    assert set(manifest)=={p.name for p in root.iterdir() if p.is_file() and p.name!='manifest.json'}, 'Missing/extra public files'
    for name, expected in manifest.items():
        p=root/name
        assert p.stat().st_size==expected['bytes'],name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==expected['sha256'],name


def interval(pairs):
    groups={g:[v for group,v in pairs if group==g] for g,_ in pairs}
    keys=sorted(groups);rng=random.Random(20261014);values=[]
    for _ in range(2000):
        sample=[v for _ in keys for v in groups[rng.choice(keys)]]
        values.append(sum(sample)/len(sample))
    values.sort()
    return {'paired_delta':sum(v for _,v in pairs)/len(pairs),'low':values[int(.025*1999)],'high':values[int(.975*1999)],
            'clusters':len(keys),'questions':len(pairs),'bootstrap_draws':2000,'seed':20261014}


def verify(root):
    def read(name):return json.loads((root/name).read_text(encoding='utf-8'))
    def table(name):return list(csv.DictReader((root/name).read_text(encoding='utf-8-sig').splitlines()))
    check_manifest(root)
    data=read('dataset-public.json');queries={q['id']:q for q in data['queries']}
    assert len(queries)==len(data['queries'])==264
    assert Counter(q['split'] for q in queries.values())=={'DEV':40,'TEST':200,'REGRESSION':24}
    for split, count in [('DEV',4),('TEST',20)]:
        counts=Counter(q['category'] for q in queries.values() if q['split']==split)
        assert len(counts)==10 and set(counts.values())=={count}
        assert len({q['group_id'] for q in queries.values() if q['split']==split})==count
    assert {q['group_id'] for q in queries.values() if q['split']=='DEV'}.isdisjoint(q['group_id'] for q in queries.values() if q['split']=='TEST')
    assert len(set(data['answer_ids']))==80
    assert set(Counter(queries[q]['category'] for q in data['answer_ids']).values())=={8}
    pools=read('candidate-pools.json');poolmap={p['query_id']:p['ids'] for p in pools}
    assert len(pools)==len(poolmap)==264 and set(poolmap)==set(queries)
    assert all(len(p['ids'])==len(p['original_section_sha256']) and len(set(p['ids']))==len(p['ids']) for p in pools)
    ranks=read('ranking-outcomes.json');bykey={(r['query_id'],r['K']):r for r in ranks}
    assert len(ranks)==len(bykey)<=792
    for r in ranks:
        assert r['K'] in (5,10,20) and r['status'] in ('ok','bypass','fallback')
        check_rank(r,queries[r['query_id']],poolmap[r['query_id']])
    summary=table('ranking-by-type.csv')
    expected_keys={(s,c,k) for s in ('DEV','TEST','REGRESSION') for c in {q['category'] for q in queries.values() if q['split']==s} for k in (5,10,20)}
    assert {(r['split'],r['category'],int(r['K'])) for r in summary}==expected_keys
    assert len(summary)==len(expected_keys)
    for row in summary:
        ids={q['id'] for q in queries.values() if q['split']==row['split'] and q['category']==row['category']}
        selected=[r for r in ranks if r['query_id'] in ids and r['K']==int(row['K'])]
        positive=[r for r in selected if queries[r['query_id']]['expected_ids'] and row['category'] not in ('aggregate','followup')]
        assert int(row['planned'])==len(ids) and int(row['checkpointed'])==len(selected)
        assert int(row['positive_scored'])==len(positive)
        for name in ('top1','hit3','hitK','contradiction_top1'):
            assert int(row[name])==sum(grade(queries[r['query_id']],r['ranked_ids'])[name] for r in positive)
        for name,status in [('gpu_used','ok'),('bypass','bypass'),('fallback','fallback')]:
            assert int(row[name])==sum(r['status']==status for r in selected)
    answers=read('answer-outcomes.json');keys={(r['query_id'],r['K']) for r in answers}
    planned={(q['id'],5) for q in queries.values() if q['split']=='DEV'}|{(q,k) for q in data['answer_ids'] for k in (5,10,20)}
    assert len(answers)==len(keys)<=280 and keys<=planned
    for r in answers:
        assert r['split']==queries[r['query_id']]['split'] and r['category']==queries[r['query_id']]['category']
        assert len(r['source_ids'])<=3 and len(r['calls'])==r['call_count']<=2
        assert all(c['model'] in (None,'gemini-3.8-flash-high') for c in r['calls'])
    answer_rows=table('answers.csv')
    assert len(answer_rows)==4 and {(r['split'],int(r['K'])) for r in answer_rows}=={('DEV',5),('TEST',5),('TEST',10),('TEST',20)}
    for row in answer_rows:
        rr=[r for r in answers if r['split']==row['split'] and r['K']==int(row['K'])]
        assert int(row['planned'])==(40 if row['split']=='DEV' else 80)
        assert int(row['checkpointed'])==len(rr)
        for name, status in [('provided','ok'),('insufficient','insufficient_evidence'),('direct_sql','aggregate'),('clarification','clarification'),('error','error')]:
            assert int(row[name])==sum(r['status']==status for r in rr)
        assert int(row['model_reached'])==sum(bool(r['call_count']) for r in rr)
        assert int(row['literal_complete'])==sum(r['diagnostic']['literal_complete'] is True for r in rr)
        assert int(row['unknown'])==sum(r['diagnostic']['literal_complete'] is None for r in rr)
        assert int(row['calls'])==sum(r['call_count'] for r in rr)
        if rr:assert math.isclose(float(row['median_cli_flow_seconds']),statistics.median(r['seconds'] for r in rr))
    typed=table('answers-by-type.csv')
    assert len(typed)==30 and {(r['category'],int(r['K'])) for r in typed}=={(q['category'],k) for q in queries.values() if q['split']=='TEST' for k in (5,10,20)}
    for row in typed:
        rr=[r for r in answers if r['split']=='TEST' and r['category']==row['category'] and r['K']==int(row['K'])]
        assert int(row['planned'])==8 and int(row['checkpointed'])==len(rr)
        for name,status_value in [('provided','ok'),('error','error')]:assert int(row[name])==sum(r['status']==status_value for r in rr)
        for name,key in [('absence_abstained','abstained'),('sql_boundary_matched','procedural_match'),('literal_complete','literal_complete')]:
            assert int(row[name])==sum(r['diagnostic'].get(key) is True for r in rr)
        assert int(row['literal_unknown'])==sum(r['diagnostic']['literal_complete'] is None for r in rr)
    scopes=read('scope-outcomes.json')
    assert len(scopes)==len({r['query_id'] for r in scopes})<=24
    for r in scopes:
        q=queries[r['query_id']];assert q['category']=='followup'
        assert r['matches_expected_scope']==(set(r['actual_source_ids'])==set(q['expected_ids']))
    changes=table('changes.csv');change_keys={(r['query_id'],int(r['K'])) for r in changes}
    expected_change_keys={(q['id'],k) for q in queries.values() if q['split']=='TEST' and q['expected_ids'] and q['category']!='followup' for k in (10,20) if (q['id'],5) in bykey and (q['id'],k) in bykey}
    assert len(changes)==len(change_keys) and change_keys==expected_change_keys
    for r in changes:
        q=queries[r['query_id']];k=int(r['K'])
        a=grade(q,bykey[(q['id'],5)]['ranked_ids']);b=grade(q,bykey[(q['id'],k)]['ranked_ids'])
        assert q['split']=='TEST' and r['group_id']==q['group_id'] and r['category']==q['category']
        assert int(r['delta'])==int(b['top1'])-int(a['top1'])
        assert r['outside5_rescued']==str(not a['hitK'] and b['top1'])
    expected_intervals=[]
    positive=sum(q['split']=='TEST' and bool(q['expected_ids']) and q['category']!='followup' for q in queries.values())
    for k in (10,20):
        selected=[r for r in changes if int(r['K'])==k]
        if len(selected)==positive:expected_intervals.append({'K':k,**interval([(r['group_id'],int(r['delta'])) for r in selected])})
    assert read('paired-deltas.json')==expected_intervals, 'Cluster interval/count mismatch'
    status=read('status.json');cost=read('cost.json')
    assert cost['attempts_without_completed_flow_row']>=0 and cost['linked_calls_without_metadata']>=0
    assert cost['answer_and_scope_cli_attempts']==sum(r['call_count'] for r in answers)+sum(r['calls'] for r in scopes)+cost['attempts_without_completed_flow_row']
    assert cost['provider_measured_calls']+cost['provider_unmeasured_calls']==cost['answer_and_scope_cli_attempts']
    if status['complete']:
        assert len(ranks)==792 and keys==planned and len(scopes)==24
    print(f"PASS: {len(ranks)}/792 ranking rows, {len(answers)}/280 answer attempts, {len(scopes)}/24 scope resolutions; public hashes/counts/candidates")
    print('Scope: recount only; not independent labels, inference, semantic correctness or Windows/field proof')


if __name__=='__main__':verify(Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent)
