"""고정한 확대 평가의 독립 집계. 모델 호출·운영 DB 쓰기·원시 답변 공개는 하지 않습니다."""
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines()] if Path(path).exists() else []
def compact(text):return ''.join(c for c in text.casefold() if c.isalnum())
def fingerprint(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def public_seconds(value):return value if type(value) in (int,float) and math.isfinite(value) and value>=0 else None
def public_usage(value):
    if not isinstance(value,dict):return None
    return {k:value[k] for k in ('input_tokens','output_tokens','thinking_tokens','cache_read_tokens','total_tokens') if type(value.get(k)) is int and value[k]>=0}
def error_type(message):
    name=(message or '').split(':',1)[0]
    return name if name in ('RuntimeError','ValueError','TimeoutError','FileExistsError','PermissionError','OSError','JSONDecodeError','KeyboardInterrupt','AssertionError','MemoryError','TypeError','KeyError','SystemExit') else 'unknown' if message else ''


def rank_grade(q,ids):
    gold=set(q['expected_ids'])
    return {'top1':bool(ids) and ids[0] in gold,'hit3':bool(gold&set(ids[:3])),'hitK':bool(gold&set(ids)),
            'contradiction_top1':bool(ids) and ids[0] in set(q.get('contradicting_ids',[]))}


def answer_diagnostic(q,row,originals):
    if q['category']=='aggregate':
        got=row.get('details',{}).get('aggregate',{}).get('count')
        return {'task':'SQL_or_supported_boundary','procedural_match':got==q['expected_count'] and row['status'] in ('aggregate','clarification'),
                'literal_complete':None,'scope':'procedural, not generated-answer semantic quality'}
    if q['category']=='absent':
        return {'task':'absence','abstained':row['status'] in ('insufficient_evidence','no_match'),
                'literal_complete':None,'scope':'explicit lack-of-evidence response; not calibration or universal hallucination detection'}
    claims=row.get('details',{}).get('claims',[])
    by_source=defaultdict(list);actions=set();cited=set()
    for claim in claims:
        for ev in claim.get('evidence',[]):
            identity=ev['id'].rsplit('/',1)[0];cited.add(identity);by_source[identity].append(claim['text'])
            if ev.get('column') in ('조치','조치내용(1)','조치내용(2)'):actions.add(identity)
    gold=set(q['expected_ids'])
    found=gold<=cited if q['category']=='exact' else bool(gold&cited)
    literal=[i for i in gold if i in actions and compact(originals[i].get('조치','')) and compact(originals[i].get('조치','')) in compact(' '.join(by_source[i]))]
    complete=bool(literal) and found
    if q['category']=='exact':complete=gold<=set(literal)
    return {'task':q['category'],'target_source_cited':found,'out_of_scope_source_cited':bool(cited&set(q.get('contradicting_ids',[]))),
            'literal_complete':True if row['status']=='ok' and complete else False if row['status']=='error' else None,
            'scope':'source/action-address plus exact literal action; paraphrase=unknown; no full semantic or field validation'}


def delta_interval(pairs,draws=2000):
    grouped=defaultdict(list)
    for group,delta in pairs:grouped[group].append(delta)
    keys=sorted(grouped);rng=random.Random(20261014)
    if not keys:return None
    values=[]
    for _ in range(draws):
        sample=[v for _ in keys for v in grouped[rng.choice(keys)]]
        values.append(sum(sample)/len(sample))
    values.sort()
    return {'paired_delta':sum(v for _,v in pairs)/len(pairs),'low':values[int(.025*(draws-1))],'high':values[int(.975*(draws-1))],
            'clusters':len(keys),'questions':len(pairs),'bootstrap_draws':draws,'seed':20261014}


def paired_inputs(outcomes):
    bykey={(r['query_id'],r['K']):r for r in outcomes if r['split']=='TEST' and r['calls']}
    result=[]
    for k in (10,20):
        pairs=[(r,bykey[(qid,k)]) for (qid,number),r in bykey.items() if number==5 and (qid,k) in bykey]
        same=sum(a['calls'][0]['rag_messages_sha256']==b['calls'][0]['rag_messages_sha256'] for a,b in pairs)
        result.append({'K':k,'paired_cli_attempts':len(pairs),'identical_initial_rag_messages':same,'different_initial_rag_messages':len(pairs)-same})
    return result


def write_csv(path,data,fieldnames=None):
    if not data and not fieldnames:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames or list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)


def cohort_calls(root,flows=()):
    """결과 행을 저장하기 전 통신 실패도 실제 호출 비용에 포함합니다."""
    paths=sorted(p for p in (root/'agy-traces').glob('*.metadata.json') if p.name.startswith(('answer-','scope-')))
    calls={p.name.removesuffix('.metadata.json'):read(p) for p in paths}
    for flow in flows:
        prefix=f"answer-{flow['query_id']}-{flow['K']}" if 'K' in flow else f"scope-{flow['query_id']}"
        for index,call in enumerate(flow['calls']):
            key=f'{prefix}-{index}'
            if key not in calls:calls[key]=call|{'metadata_missing':True}
            else:
                for field in ('wall_seconds','provider_duration','usage','confirmed_model'):
                    if calls[key].get(field) is not None and call.get(field) is not None:
                        assert calls[key][field]==call[field],f'Conflicting measurement for {key}/{field}'
                    if call.get(field) is not None:calls[key][field]=call[field]
    return list(calls.values())


def analyze(root):
    p=read(root/'protocol.json');data=read(root/'dataset.json');status=read(root/'status.json')
    if status['stage'] not in ('model_experiments_finished','model_experiments_finished_with_errors','interrupted'):raise ValueError('Wait until model timing ends')
    assert hashlib.sha256((root/'dataset.json').read_bytes()).hexdigest()==p['dataset_sha256']
    query={q['id']:q for q in data['queries']};originals={r['_record_id']:r for r in data['records']}
    candidates={r['query_id']:r for r in rows(root/'candidates.jsonl')}
    ranks=rows(root/'rankings.jsonl');answers=rows(root/'answers.jsonl');scopes=rows(root/'scopes.jsonl')
    if status['complete']:
        planned={(q['id'],5) for q in data['queries'] if q['split']=='DEV'}|{(q,k) for q in data['answer_ids'] for k in (5,10,20)}
        assert len(ranks)==792 and {(r['query_id'],r['K']) for r in ranks}=={(q,k) for q in query for k in (5,10,20)}
        assert len(answers)==280 and {(r['query_id'],r['K']) for r in answers}==planned
        assert len(scopes)==24 and {r['query_id'] for r in scopes}=={q['id'] for q in data['queries'] if q['category']=='followup'}
    assert len({(r['query_id'],r['K']) for r in ranks})==len(ranks)
    assert len({(r['query_id'],r['K']) for r in answers})==len(answers)
    for r in ranks:
        before=[h['record_id'] for h in candidates[r['query_id']]['hits'][:r['K']]]
        assert Counter(before)==Counter(r['ranked_ids'])
        assert rank_grade(query[r['query_id']],before)['hitK']==rank_grade(query[r['query_id']],r['ranked_ids'])['hitK']
    summaries=[]
    for split in ('DEV','TEST','REGRESSION'):
        categories=sorted({q['category'] for q in data['queries'] if q['split']==split})
        for category in categories:
            qs=[q for q in data['queries'] if q['split']==split and q['category']==category]
            for k in (5,10,20):
                rr=[r for r in ranks if r['K']==k and r['query_id'] in {q['id'] for q in qs}]
                pos=[r for r in rr if query[r['query_id']]['expected_ids'] and category not in ('aggregate','followup')]
                scores=[rank_grade(query[r['query_id']],r['ranked_ids']) for r in pos]
                summaries.append({'split':split,'category':category,'K':k,'planned':len(qs),'checkpointed':len(rr),'positive_scored':len(pos),
                    'top1':sum(g['top1'] for g in scores),'hit3':sum(g['hit3'] for g in scores),'hitK':sum(g['hitK'] for g in scores),
                    'contradiction_top1':sum(g['contradiction_top1'] for g in scores),'gpu_used':sum(r['status']=='ok' for r in rr),
                    'bypass':sum(r['status']=='bypass' for r in rr),'fallback':sum(r['status']=='fallback' for r in rr)})
    by_key={(r['query_id'],r['K']):r for r in ranks};paired=[];changes=[]
    for k in (10,20):
        pairs=[]
        for q in data['queries']:
            if q['split']!='TEST' or not q['expected_ids'] or q['category']=='followup':continue
            if (q['id'],5) not in by_key or (q['id'],k) not in by_key:continue
            a=rank_grade(q,by_key[q['id'],5]['ranked_ids']);b=rank_grade(q,by_key[q['id'],k]['ranked_ids'])
            delta=int(b['top1'])-int(a['top1']);pairs.append((q['group_id'],delta))
            changes.append({'query_id':q['id'],'category':q['category'],'group_id':q['group_id'],'K':k,'delta':delta,
                'outside5_rescued':not a['hitK'] and b['top1']})
        planned=sum(q['split']=='TEST' and bool(q['expected_ids']) and q['category']!='followup' for q in data['queries'])
        result=delta_interval(pairs) if len(pairs)==planned else None
        if result:paired.append({'K':k,**result})
    outcome=[]
    for r in answers:
        q=query[r['query_id']];diag=answer_diagnostic(q,r,originals)
        outcome.append({'query_id':r['query_id'],'split':q['split'],'category':q['category'],'K':r['K'],'status':r['status'],
            'seconds':r['seconds'],'source_ids':r.get('source_ids',[]),'error_type':error_type(r.get('error')),
            'diagnostic':diag,'call_count':len(r['calls']),
            'calls':[{'wall_seconds':c['wall_seconds'],'provider_duration':public_seconds(c.get('provider_duration')),'usage':public_usage(c.get('usage')),
                'rag_messages_sha256':fingerprint(c['request']['messages']),'schema_sha256':fingerprint(c['schema']),
                'model':c.get('confirmed_model'),'error_type':error_type(c.get('error')),'cli_timeout_seconds':c.get('cli_timeout_seconds'),'unmapped_options':c.get('unmapped_options',{})} for c in r['calls']],
            'validation_categories':[e['category'] for e in r.get('details',{}).get('validation_events',[])]})
    answer_summary=[]
    for split in ('DEV','TEST'):
        for k in ((5,) if split=='DEV' else (5,10,20)):
            rr=[r for r in outcome if r['split']==split and r['K']==k]
            attempted_times=sorted(r['seconds'] for r in rr if r['call_count'])
            answer_summary.append({'split':split,'K':k,'planned':40 if split=='DEV' else 80,'checkpointed':len(rr),
                'cli_attempted_flows':sum(bool(r['call_count']) for r in rr),'provided':sum(r['status']=='ok' for r in rr),
                'insufficient':sum(r['status']=='insufficient_evidence' for r in rr),'direct_sql':sum(r['status']=='aggregate' for r in rr),
                'clarification':sum(r['status']=='clarification' for r in rr),'error':sum(r['status']=='error' for r in rr),
                'literal_complete':sum(r['diagnostic']['literal_complete'] is True for r in rr),
                'unknown':sum(r['diagnostic']['literal_complete'] is None for r in rr),
                'calls':sum(r['call_count'] for r in rr),
                'median_answer_stage_seconds_all_paths':statistics.median(r['seconds'] for r in rr) if rr else None,
                'cli_attempted_answer_stage_samples':len(attempted_times),
                'median_cli_attempted_answer_stage_seconds':statistics.median(attempted_times) if attempted_times else None,
                'p95_cli_attempted_answer_stage_seconds':attempted_times[int(.95*(len(attempted_times)-1))] if attempted_times else None})
    follow=[]
    for r in scopes:
        q=query[r['query_id']];resolved=r.get('resolved') or {};ids={i.rsplit('/',1)[0] for i in resolved.get('source_ids',[])}
        follow.append({'query_id':q['id'],'split':q['split'],'actual_source_ids':sorted(ids),'matches_expected_scope':ids==set(q['expected_ids']),
            'clarification':'clarification' in resolved,'error':bool(r.get('error')),'calls':len(r['calls']),
            'cli_wall_seconds':sum(c['wall_seconds'] for c in r['calls'])})
    costs=[{k:r[k] for k in ('wall_seconds','parent_sampled_child_peak_rss','exit','results','prototype_resident') if k in r} for r in rows(root/'gpu-costs.jsonl')]
    all_calls=cohort_calls(root,answers+scopes)
    linked_calls=sum(len(r['calls']) for r in answers+scopes)
    provider=[v for c in all_calls if (v:=public_seconds(c.get('provider_duration'))) is not None]
    tokens={key:[usage[key] for c in all_calls if (usage:=public_usage(c.get('usage'))) is not None and key in usage] for key in ('input_tokens','thinking_tokens','output_tokens')}
    cost={'gpu_execution_attempts':costs,'answer_and_scope_cli_attempts':len(all_calls),
          'attempts_without_completed_flow_row':len(all_calls)-linked_calls,
          'linked_calls_without_metadata':sum(bool(c.get('metadata_missing')) for c in all_calls),
          'request_without_final_metadata':len([p for p in (root/'agy-traces').glob('*.request.json') if p.name.startswith(('answer-','scope-')) and not p.with_name(p.name.replace('.request.json','.metadata.json')).exists()]),
          'cli_wall_sum':sum(c['wall_seconds'] for c in all_calls),'cli_wall_is_not_parallel_elapsed':True,
          'provider_seconds_sum':sum(provider) if provider else None,
          'provider_measured_calls':len(provider),'provider_unmeasured_calls':len(all_calls)-len(provider),
          **{key:sum(values) if values else None for key,values in tokens.items()},
          'token_measured_calls':{key:len(values) for key,values in tokens.items()},
          'embedding_preparation':read(root/'embedding-cost.json') if (root/'embedding-cost.json').exists() else None,
          'remote_gpu_memory':None,'user_browser_end_to_end_latency':None,'resident_prototype':True}
    public=root/'public';public.mkdir(exist_ok=True)
    public_status={k:status[k] for k in ('stage','time','complete','rankings','answer_attempts','wall_seconds','transport_failures','scope_errors','planned_attempts_recorded','partial_preserved') if k in status}
    public_status['error_type']=error_type(status.get('error'))
    for name,value in [('ranking-outcomes.json',ranks),('answer-outcomes.json',outcome),('scope-outcomes.json',follow),('cost.json',cost),('paired-deltas.json',paired),('status.json',public_status)]:
        if name=='ranking-outcomes.json':
            value=[{k:r[k] for k in ('query_id','K','status','ranked_ids','seconds')}|{'load_seconds':r.get('load_seconds'),'gpu_peak':r.get('gpu_peak')} for r in ranks]
        (public/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    write_csv(public/'ranking-by-type.csv',summaries);write_csv(public/'answers.csv',answer_summary)
    write_csv(public/'changes.csv',changes,('query_id','category','group_id','K','delta','outside5_rescued'))
    input_pairs=paired_inputs(outcome);write_csv(public/'matched-inputs.csv',input_pairs)
    report=['# AGY 임시 모델 · 질문 유형 확대 평가','',f"기준2026-10-14 KST · 상태 {status['stage']} · 전체 모델 실험 완료 여부 {status['complete']}",'',
       'DEV40·TEST200·회귀24를 분리했다. 신규 TEST는 하드웨어/공통 생성 계열20군이며, 독립 현장 사례200개가 아니다. 전부 창작 합성/실버 라벨이다.', '',
       '## 신규 TEST 순위','', '|유형|K|저장/계획|양성 분모|첫 기대ID|상위3|입력 내 포함|GPU|우회|fallback|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        if r['split']=='TEST':report.append(f"|{r['category']}|{r['K']}|{r['checkpointed']}/{r['planned']}|{r['positive_scored']}|{r['top1']}|{r['hit3']}|{r['hitK']}|{r['gpu_used']}|{r['bypass']}|{r['fallback']}|")
    report+=['','기대ID는 고정한 합성 실버 라벨이며 사람이나 현장의 독립 정답이 아니다. 근거 없음은 양성 순위 분모에 넣지 않는다. 집계와 후속 질문도 별도 절차 지표다. 정확 조회 우회는 Clef의 성과가 아니다. 입력 후보가 같은 K의 재정렬 전후 Hit@K 불변과 후보 다중집합을 대조했다. 미실행은 오답으로 바꾸지 않고 저장/계획을 표시한다. 아래 짝지은 양성 전체에는 정확 조회 우회도 들어 있으므로 유형별 표와 함께 읽는다.','','## 짝지은 차이와 불확실성','']
    for r in paired:report.append(f"- K{r['K']} 대 K5: 첫 적합 차이 {r['paired_delta']*100:.2f}%p, 군집 bootstrap95% 구간 [{r['low']*100:.2f}, {r['high']*100:.2f}]%p; {r['questions']}문항/{r['clusters']}군집.")
    report+=['','## 답변 시도','', '|구간|K|저장/계획|CLI 시도|답변 제공|근거 부족|직접 SQL|대상 재확인|오류|제한적 문자 충족|unknown|호출|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in answer_summary:report.append(f"|{r['split']}|{r['K']}|{r['checkpointed']}/{r['planned']}|{r['cli_attempted_flows']}|{r['provided']}|{r['insufficient']}|{r['direct_sql']}|{r['clarification']}|{r['error']}|{r['literal_complete']}|{r['unknown']}|{r['calls']}|")
    report+=['','CLI 시도에는 모델 생성 전에 실패한 실행도 포함한다. 모델의 실제 추론 진입이나 완료를 보장하는 수가 아니다.','']
    report+=['','문자 진단은 정답 주소의 조치 셀을 인용했고 그 원문의 조치·결과 문구가 답변에 직접 있는지만 본다. 의역·번역은 unknown이며 의미 오류 차단이나 최종 정답률이 아니다. SQL 경계·근거 부족 응답은 해당 과제 지표로 분리한다. TEST 답변은 유형별8개, 총80질문×3구성이고 DEV는40질문×K5이다. 응답 실패와 미실행을 제외해 성공률을 키우지 않는다.','','## 비용과 해석','',
       f"실제 답변·후속 CLI 시도 {cost['answer_and_scope_cli_attempts']}, 전체 CLI wall 합 {cost['cli_wall_sum']:.3f}초. 완료 행에 연결되지 못한 호출 {cost['attempts_without_completed_flow_row']}회, 종료 메타데이터가 없는 요청 {cost['request_without_final_metadata']}회도 구분한다. 동시 실행 상한 {p['workers']}개이므로 wall 합은 전체 경과시간이 아니다. 제공자 처리 측정합 {cost['provider_seconds_sum']}초(측정 {cost['provider_measured_calls']}회·미측정 {cost['provider_unmeasured_calls']}회)는 전체가 아니다. 미측정을 0초로 바꾸지 않는다. 토큰은 AGY 내부 시스템 문맥도 포함하며 측정 건수를 별도 기록한다.",'',
       'GPU는 엄격한 MLX8bit 상주 프로토타입으로 실행했다. 적재를 요청마다 다시 하는 운영 성능이나 Windows/120B 속도로 발표하지 않는다. 원격 AGY 장치 메모리·브라우저 전체 지연은 미측정이다. 사전 연결 시험은 이 코호트 밖이며 별도 로그와 비용으로 보존한다. AGY의 temperature·출력/문맥 한도를 동일하게 강제했다고 주장하지 않는다.','',
       '첫 CLI 시도의 RAG 메시지 지문이 같은 쌍은 동일한 선택 근거·지시문을 받은 경우다. 그래도 결과가 다르면 후보 수만의 인과 효과로 돌릴 수 없다. 지문은 제공자의 숨은 내부 문맥까지 동일함을 증명하지 않는다. 실패도 포함한 공통 CLI 진입 쌍만 세며 미실행은 별도다.','',
       '발표 가능: 고정한 합성 기술 과제에서 질문 유형별 관찰 결과·실패·범위·실측 실행 비용. 발표 불가: 현장 정답률·사람 공수 절감·고장 해결 실적·Clef 고유 우위·임상/양산 성과. 운영 코드·개인 설정·DB를 교체하지 않았다.']
    for r in input_pairs:report.append(f"\n- K{r['K']} 대 K5: 공통 초기 CLI 시도 {r['paired_cli_attempts']}쌍 중 같은 RAG 메시지 {r['identical_initial_rag_messages']}쌍.")
    (public/'report.md').write_text('\n'.join(report)+'\n')
    print('RECOUNT',len(ranks),'rankings',len(answers),'answer attempts',len(scopes),'followup outcome rows')

if __name__=='__main__':analyze(Path(sys.argv[1]).resolve())
