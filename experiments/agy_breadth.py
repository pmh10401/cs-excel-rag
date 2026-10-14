"""임시 AGY 답변 모델로 10유형을 평가하는 격리 실행기. 운영 DB·설정을 읽지 않습니다."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import selectors
import signal
import sqlite3
import subprocess
import sys
import time
from unittest.mock import patch

from experiments.agy_client import AgyClient, FOLLOW_SCHEMA, MODEL
from experiments.agy_cases import build, KINDS
from ragkit import clef, cslist, llm, search, semantic, storage

REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO/'outputs/ai-track-v2-20261014-125025-v3/runtime/bin/python'
WORKER = REPO/'experiments/agy_clef_worker.py'
CLEF_PATH = Path('/Volumes/MacStorage/AIModels/clef-rag-test-20261008/hub-mlx/models--mlx-community--clef-flash-8bit/snapshots/5a195dc147926b63b1d139691ed4480ca9f92c0e')
LMS = '/Users/mac/.lmstudio/bin/lms'
KS = (5,10,20)
ANSWER_CFG = {'model':MODEL,'reasoning_effort':'high','answer_instruction_language':'ko','claim_guard_mode':'off','max_context_chars':24000}


def read(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines()] if Path(path).exists() else []
def append(path,value):
    with Path(path).open('a') as stream:stream.write(json.dumps(value,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
def db_read(root):return closing(sqlite3.connect('file:'+str(root/'experiment.sqlite3')+'?mode=ro',uri=True))
def state(root,stage,**extra):save(root/'status.json',{'stage':stage,'time':time.time(),'complete':False,**extra})


def prepare(root):
    if (root/'protocol.json').exists():raise ValueError('Use a new output; frozen protocol cannot be overwritten')
    data=build(REPO);save(root/'dataset.json',data)
    schema,rules=cslist.load_schema(REPO/'windows-demo/cslist_schema.json')
    with closing(sqlite3.connect(root/'experiment.sqlite3')) as db:
        storage.ingest(db,data['records'],rules,schema=schema)
        assert db.execute('select count(*) from normalized_records').fetchone()[0]==2998
        assert db.execute('pragma integrity_check').fetchone()[0]=='ok'
    original=read(REPO/'outputs/candidate-budget-20261014-151500/protocol.initial.json')
    files=[Path(__file__),REPO/'experiments/agy_client.py',REPO/'experiments/agy_cases.py',REPO/'tests/test_agy_breadth.py',WORKER,*list((REPO/'ragkit').glob('*.py')),REPO/'experiments/clef_mlx_server.py']
    protocol={'id':root.name,'created_at':time.time(),'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
      'model_stop_at':original['model_stop_at'],'delivery_deadline':original['delivery_deadline'],'deadline_policy':'Original24h/14h boundaries retained; no reset',
      'source_hashes':{str(p.relative_to(REPO)):sha(p) for p in files},'dirty_diff_sha256':hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=REPO)).hexdigest(),'new_sources_uncommitted_at_freeze':True,'dataset_sha256':sha(root/'dataset.json'),
      'personal_config_sha256':sha(REPO/'mac_config.json'),'model':MODEL,'effort':'high','K':list(KS),'references':3,'workers':2,
      'dev':40,'test':200,'regression':24,'test_clusters':20,'dev_clusters':4,'test_answer_ids':data['answer_ids'],
      'answer_plan':'40DEV atK5;80TEST balanced8pertype at3K;followup resolve once per question;SQL/count no answer LLM',
      'maximum_answer_attempts':280,'maximum_generation_calls':560,'maximum_followup_calls':24,'transport_retries':0,'validation_repair':1,
      'controls':'AGY supports model/effort/schema; native temperature/output/context limits not exposed. Not comparable with localGPTOSS defaults.',
      'gpu_mode':'Existing Clef MLX8bit resident worker for quality; prototype, not current request-per-load deployment latency',
      'gpu_order':'question index rotates5/10/20;DEV precedesTEST;original wire instructions/fields/checks;no truncation',
      'stopping':'Two transport failures in a wave pause pending requests, not counted as completed; resume retains exact completed request keys',
      'statistics':'TEST-only paired cluster bootstrap2000samples seed20261014;20hardware/template groups;descriptive,synthetic/silver,no field/human claim',
      'metrics':'top1,hit3,hitK for gold-positive only;none/SQL/exact/followup separately;answer delivery,limited literal support,unknown separate',
      'privacy':'Only authored synthetic source/messages supplied;OS sandbox deniesDocuments,CloudStorage,Codex reads;no TEST labels in request',
      'old_results':'Previous24regression not merged withTEST;no previous output overwritten'}
    assert time.time()<protocol['model_stop_at']
    save(root/'protocol.json',protocol);(root/'protocol.sha256').write_text(sha(root/'protocol.json')+'\n')
    state(root,'prepared',questions=264,records=2998)
    print('FROZEN:DEV40 TEST200 REGRESSION24;20TESTclusters;80answerTESTquestions',flush=True)


def verify(root):
    p=read(root/'protocol.json');assert sha(root/'protocol.json')==(root/'protocol.sha256').read_text().strip()
    assert sha(root/'dataset.json')==p['dataset_sha256']
    for name,value in p['source_hashes'].items():assert sha(REPO/name)==value,name
    assert sha(REPO/'mac_config.json')==p['personal_config_sha256']
    return p


def cli(root,*args):
    p=subprocess.run([LMS,*args],capture_output=True,text=True,timeout=180)
    append(root/'runtime.jsonl',{'args':args,'stdout':p.stdout,'stderr':p.stderr,'exit':p.returncode,'time':time.time()})
    if p.returncode:raise RuntimeError('LM Studio CLI failed; retained raw output')
    return p.stdout


def loaded(root):
    for attempt in range(3):
        try:
            data=json.loads(cli(root,'ps','--json'))
            if not isinstance(data,list) or any(not isinstance(m,dict) or not isinstance(m.get('identifier'),str) for m in data):raise ValueError('Invalid model inventory')
            return data
        except (ValueError,json.JSONDecodeError):
            if attempt==2:raise
            time.sleep(1)


def wait_gpu(root,p):
    import psutil
    while time.time()<p['model_stop_at']:
        models=loaded(root)
        active=[]
        for proc in psutil.process_iter(['pid','cmdline']):
            try:
                if proc.pid!=os.getpid() and any(str(a).endswith('code/model_worker.py') or str(a).endswith('experiments/agy_clef_worker.py') or a in ('experiments.clm_rerank','experiments.lev_rerank','mlx_lm.server') for a in (proc.info['cmdline'] or [])):active.append(proc.pid)
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        if not models and not active and psutil.virtual_memory().available>=10*2**30:return
        state(root,'resource_wait',foreign_model_count=len(models),other_worker_pids=active)
        time.sleep(20)
    raise TimeoutError('Original model deadline reached')


def freeze_candidates(root,p,data):
    file=root/'candidates.jsonl'
    if file.exists():
        assert sha(file)==read(root/'candidate-freeze.json')['sha256'];return rows(file)
    wait_gpu(root,p)
    alias='cs-agy-breadth-embedding-20261014'
    cfg={'embedding':{'base_url':'http://127.0.0.1:1234/v1','api_key':'EMPTY','model':alias,
      'document_prefix':'title: none | text: ','query_prefix':'task: search result | query: ',
      'revision':root.name,'batch_size':16,'min_score':0.3,'auto_index':False,'auto_start':False}}
    started=time.perf_counter();owned=False
    try:
        cli(root,'load','text-embedding-embeddinggemma-2','--identifier',alias,'--gpu','max');owned=True
        from openai import OpenAI
        with OpenAI(base_url='http://127.0.0.1:1234/v1',api_key='EMPTY',max_retries=0,timeout=180) as client:
            with closing(sqlite3.connect(root/'experiment.sqlite3')) as db:index=semantic.build_vectors(db,client,cfg)
            assert index['indexed']==2998 and not index['failures']
            vectors=semantic.embed(client,[search.focused_query(q['question']) for q in data['queries']],semantic.settings(cfg),query=True)
        originals={r['_record_id']:{k:v for k,v in r.items() if not k.startswith('_')} for r in data['records']}
        frozen=[]
        with db_read(root) as db:
            for q,vector in zip(data['queries'],vectors):
                t=time.perf_counter();details={}
                aggregate=search.count_records(db,q['question']) if q['category']=='aggregate' else None
                if q['category']=='aggregate':assert aggregate is not None and aggregate['count']==q['expected_count']
                ids=q.get('history_record_ids')
                hits=[] if aggregate else semantic.search_cases(db,q['question'],cfg,mode='hybrid',limit=20,query_vector=vector[None,:],record_ids=ids,details=details)
                for hit in hits:
                    assert all(v==originals[hit['record_id']][k] for s in hit['evidence'] for k,v in s['cells'].items())
                frozen.append({'query_id':q['id'],'hits':hits,'aggregate':aggregate,'retrieval_seconds':time.perf_counter()-t,'details':details})
        temporary=file.with_suffix('.tmp')
        for r in frozen:append(temporary,r)
        temporary.replace(file)
        save(root/'candidate-freeze.json',{'sha256':sha(file),'protocol_sha256':sha(root/'protocol.json'),'no_answer_or_rerank_started':True})
        save(root/'embedding-cost.json',{'seconds':time.perf_counter()-started,'index':index})
        return frozen
    finally:
        if owned:cli(root,'unload',alias)


class Captured(Exception):pass

def wire(root,q,hits,k):
    captured={};details={}
    def send(request,timeout):captured.update(json.loads(request.data));raise Captured()
    with db_read(root) as db,patch.object(clef,'prepare',lambda *a,**kw:None),patch.object(clef,'urlopen',send):
        try:clef.rerank(db,q['question'],hits[:k],{'model':'clef:flash','candidates':k},details=details)
        except Captured:pass
    return captured,details


def replay(root,q,hits,k,response):
    details={}
    with db_read(root) as db,patch.object(clef,'prepare',lambda *a,**kw:None),patch.object(clef,'urlopen',lambda *a,**kw:io.StringIO(json.dumps(response))):
        result=clef.rerank(db,q['question'],hits[:k],{'model':'clef:flash','candidates':k},details=details)
    if details.get('clef_search',{}).get('status')!='used':raise ValueError('Existing Clef response validation failed')
    assert Counter(h['record_id'] for h in result)==Counter(h['record_id'] for h in hits[:k])
    return result,details



def record_rank(root,q,hits,k,raw,request):
    assert raw['request_sha256']==digest(request),'Cached GPU input differs'
    try:
        ranked,details=replay(root,q,hits,k,raw);status='ok';error=None
    except Exception as e:
        ranked=hits[:k];details={};status='fallback';error=str(e)
    append(root/'rankings.jsonl',{'query_id':q['id'],'K':k,'status':status,'ranked_ids':[h['record_id'] for h in ranked],
        'details':details,'error':error,'seconds':raw.get('compute_seconds'),'load_seconds':raw.get('load_seconds'),
        'gpu_peak':raw.get('device_peak_bytes'),'rss':raw.get('peak_rss_bytes'),'raw_key':raw['key']})


def rank_all(root,p,data,frozen):
    by_id={r['query_id']:r for r in frozen};qs={q['id']:q for q in data['queries']}
    output=root/'rankings.jsonl';done={(r['query_id'],r['K']) for r in rows(output)};pending=[];all_requests=[]
    cache={r['key']:r for r in rows(root/'gpu-results.jsonl')}
    for i,q in enumerate(data['queries']):
        for k in KS[i%3:]+KS[:i%3]:
            hits=by_id[q['id']]['hits']
            request,details=({}, {}) if q['category'] in ('aggregate','followup') else wire(root,q,hits,k)
            if request:
                item={'key':q['id']+'::'+str(k),'request':request};all_requests.append(item)
                if (q['id'],k) not in done:
                    if item['key'] in cache:record_rank(root,q,hits,k,cache[item['key']],request)
                    else:pending.append(item)
            elif (q['id'],k) not in done:append(output,{'query_id':q['id'],'K':k,'status':'bypass','ranked_ids':[h['record_id'] for h in hits[:k]],'details':details,'seconds':0})
    freeze=root/'ranking-request-freeze.json'
    if freeze.exists():assert read(freeze)['sha256']==digest(all_requests)
    else:save(freeze,{'requests':all_requests,'sha256':digest(all_requests),'protocol_sha256':sha(root/'protocol.json')})
    if not pending:return
    wait_gpu(root,p)
    import psutil
    spec={'kind':'clef-resident','path':str(CLEF_PATH),'requests':pending}
    env=dict(os.environ,PYTHONPATH=str(REPO),PYTORCH_ENABLE_MPS_FALLBACK='0',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    start=time.perf_counter();count=0;low=0;peak=0
    with (root/'gpu.stderr.log').open('a') as stderr:
        child=subprocess.Popen([str(PYTHON),'-B','-u',str(WORKER)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,env=env,cwd=REPO)
        try:
            child.stdin.write(json.dumps(spec,ensure_ascii=False));child.stdin.close()
            selector=selectors.DefaultSelector();selector.register(child.stdout,selectors.EVENT_READ);last=time.monotonic();inventory_checked=last
            eof=False
            while True:
                if time.time()>=p['model_stop_at'] or time.monotonic()-last>300:raise TimeoutError('GPU deadline/no-result boundary reached')
                if time.monotonic()-inventory_checked>15:
                    if loaded(root):raise RuntimeError('Foreign LM Studio model appeared; only owned GPU child is stopped')
                    inventory_checked=time.monotonic()
                try:peak=max(peak,psutil.Process(child.pid).memory_info().rss)
                except psutil.NoSuchProcess:pass
                low=low+1 if psutil.virtual_memory().available<3*2**30 else 0
                if low>=15:raise RuntimeError('Memory pressure; only owned GPU child is stopped')
                for _,_ in selector.select(.2):
                    line=child.stdout.readline()
                    if not line:
                        eof=True
                        break
                    with (root/'gpu.stdout.log').open('a') as stream:stream.write(line)
                    if not line.startswith('RESULT:'):continue
                    raw=json.loads(line[7:]);qid,number=raw['key'].rsplit('::',1);k=int(number);q=qs[qid]
                    append(root/'gpu-results.jsonl',raw)
                    record_rank(root,q,by_id[qid]['hits'],k,raw,next(item['request'] for item in pending if item['key']==raw['key']))
                    count+=1;last=time.monotonic();state(root,'rankings',resident_done=count,resident_planned=len(pending),total_rows=len(rows(output)))
                if child.poll() is not None and eof:
                    if child.returncode:raise RuntimeError('GPU worker failed; partial results preserved')
                    break
            selector.close()
            assert count==len(pending),'Do not treat worker termination as completion'
        finally:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            child.stdout.close()
            append(root/'gpu-costs.jsonl',{'wall_seconds':time.perf_counter()-start,'parent_sampled_child_peak_rss':peak,'exit':child.returncode,'results':count,'prototype_resident':True})


def history_for(q,frozen):
    by_id={h['record_id']:h for h in frozen['hits']}
    claims=[]
    for identity in q['history_record_ids']:
        hit=by_id[identity];s=hit['evidence'][0]
        claims.append({'text':s['cells'].get('현상',''),'sources':[{'id':s['id'],'receipt':s['cells']['CS NO'],'cells':s['cells']}]})
    return [{'question':'앞서 확인한 두 부품의 사례를 설명해 주세요.','answer_claims':claims}]


def resolve_scope(root,p,q,frozen):
    client=AgyClient(root,'scope-'+q['id'],schema=FOLLOW_SCHEMA,timeout=p.get('cli_timeout_seconds',180),deadline=p['model_stop_at'])
    try:
        result=llm.resolve_follow_up(client,ANSWER_CFG,q['question'],history_for(q,frozen))
        return {'query_id':q['id'],'resolved':result,'calls':client.calls}
    except Exception as error:
        return {'query_id':q['id'],'resolved':None,'error':f'{type(error).__name__}: {error}','calls':client.calls,
            'transport_failure':not client.calls or any(c.get('error') for c in client.calls)}


def answer_one(root,p,q,k,frozen,ranking,scope):
    start=time.perf_counter();details={};calls=[];error=None;status='error';hits=[];client=None;stage='scope'
    try:
        if time.time()>=p['model_stop_at']-20:raise TimeoutError('Original model stop reached')
        if frozen['aggregate']:
            status='aggregate' if frozen['aggregate']['count'] is not None else 'clarification'
            details={'aggregate':frozen['aggregate']}
        else:
            by_id={h['record_id']:h for h in frozen['hits']}
            hits=[by_id[identity] for identity in ranking['ranked_ids'][:3]]
            question=q['question']
            if q['category']=='followup':
                if scope.get('error'):raise ValueError('Follow-up response validation failed: '+scope['error'])
                resolved=scope['resolved']
                if 'clarification' in resolved:return {'query_id':q['id'],'K':k,'status':'clarification','details':resolved,'calls':[],'seconds':time.perf_counter()-start,'source_ids':[]}
                question=resolved['question']
                if resolved.get('source_ids'):
                    selected={identity.rsplit('/',1)[0] for identity in resolved['source_ids']}
                    hits=[h for h in frozen['hits'] if h['record_id'] in selected][:3]
                else:
                    raise ValueError('Follow-up did not identify a fixed previous case; no silent gold scope substitution')
            if not hits:status='no_match'
            else:
                stage='model'
                client=AgyClient(root,'answer-'+q['id']+'-'+str(k),timeout=p.get('cli_timeout_seconds',180),deadline=p['model_stop_at'])
                llm.answer(client,ANSWER_CFG,question,hits,details=details)
                status=details.get('answer_status','error')
    except Exception as e:error=f'{type(e).__name__}: {e}'
    if client:calls=client.calls
    assert len(calls)<=2
    return {'query_id':q['id'],'K':k,'status':status,'error':error,'transport_failure':bool(error) and stage=='model' and (not calls or any(c.get('error') for c in calls)),'details':details,'calls':calls,
            'seconds':time.perf_counter()-start,'source_ids':[h['record_id'] for h in hits]}


def run_answers(root,p,data,frozen):
    output=root/'answers.jsonl';done={(r['query_id'],r['K']) for r in rows(output)}
    by_id={r['query_id']:r for r in frozen};ranked={(r['query_id'],r['K']):r for r in rows(root/'rankings.jsonl')}
    scopes={r['query_id']:r for r in rows(root/'scopes.jsonl')}
    workers=p.get('workers',2)
    if type(workers) is not int or not 1<=workers<=6:raise ValueError('Frozen workers must be 1..6')
    # 후속 해석 실패도 실제 결과로 저장합니다. 다른 질문 유형의 실행을 막지 않습니다.
    for split in ('DEV','TEST'):
        pending=[q for q in data['queries'] if q['category']=='followup' and q['split']==split and q['id'] not in scopes]
        for offset in range(0,len(pending),workers):
            if time.time()>=p['model_stop_at']-20:raise TimeoutError('Original model deadline before scope request')
            with ThreadPoolExecutor(max_workers=workers) as pool:
                tasks=[pool.submit(resolve_scope,root,p,q,by_id[q['id']]) for q in pending[offset:offset+workers]]
                for task in as_completed(tasks):
                    scope=task.result();append(root/'scopes.jsonl',scope);scopes[scope['query_id']]=scope
                    state(root,'followup_resolution',completed=len(scopes),planned=24)
    plan=[]
    for q in data['queries']:
        if q['split']=='DEV':ks=(5,)
        elif q['id'] in data['answer_ids']:ks=KS
        else:continue
        for k in ks:
            if (q['id'],k) not in done:plan.append((q,k))
    failures=0
    for offset in range(0,len(plan),workers):
        if time.time()>=p['model_stop_at']-20:raise TimeoutError('Original model deadline; pending answers preserved')
        wave=plan[offset:offset+workers];failures=0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            tasks=[pool.submit(answer_one,root,p,q,k,by_id[q['id']],ranked[q['id'],k],scopes.get(q['id'])) for q,k in wave]
            for task in as_completed(tasks):
                result=task.result();append(output,result)
                transport=result.get('transport_failure',False)
                failures+=int(transport)
                state(root,'answers',completed=len(rows(output)),planned=280)
        if failures>=2:raise RuntimeError('Repeated transport failures: stopped with pending requests, no success-until retry')


def run(root):
    with (root/'execution.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=verify(root);data=read(root/'dataset.json');started=time.perf_counter()
        try:
            frozen=freeze_candidates(root,p,data)
            rank_all(root,p,data,frozen)
            assert len(rows(root/'rankings.jsonl'))==792
            run_answers(root,p,data,frozen)
            verify(root)
            answers=rows(root/'answers.jsonl');assert len(answers)==280
            failures=sum(bool(r.get('transport_failure')) for r in answers)
            scope_errors=sum(bool(r.get('error')) for r in rows(root/'scopes.jsonl'))
            save(root/'status.json',{'stage':'model_experiments_finished' if not failures and not scope_errors else 'model_experiments_finished_with_errors',
                'complete':not failures and not scope_errors,'transport_failures':failures,'scope_errors':scope_errors,'planned_attempts_recorded':True,'time':time.time(),
                'rankings':792,'answer_attempts':280,'wall_seconds':time.perf_counter()-started})
        except BaseException as error:
            state(root,'interrupted',error=f'{type(error).__name__}: {error}',partial_preserved=True);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run'));parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve()
    if root==REPO or REPO not in root.parents or root.parent!=REPO/'outputs':raise ValueError('Use a separate direct outputs child')
    root.mkdir(exist_ok=True)
    (prepare if args.action=='prepare' else run)(root)

if __name__=='__main__':main()
