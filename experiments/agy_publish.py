"""종료된 합성 실험의 공개 묶음만 만듭니다. 원시 모델 입출력은 복사하지 않습니다."""
import hashlib
import csv
import json
from pathlib import Path
import shutil
import sys

from experiments.agy_report import error_type, public_seconds, public_usage, read, rows, write_csv
from experiments.agy_verify import verify


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def publish(root, destination):
    status=read(root/'status.json')
    if status['stage'] not in ('interrupted','model_experiments_finished','model_experiments_finished_with_errors'):
        raise ValueError('Do not publish an active evaluation as a final result')
    destination.mkdir(parents=True, exist_ok=False)
    public=root/'public'
    # 명시한 파생 결과만 복사합니다. 재귀 복사는 요청·응답 로그를 노출할 수 있습니다.
    names=('ranking-outcomes.json','answer-outcomes.json','scope-outcomes.json','cost.json',
           'paired-deltas.json','status.json','ranking-by-type.csv','answers.csv','changes.csv','matched-inputs.csv','report.md')
    for name in names:
        if (public/name).exists():shutil.copyfile(public/name,destination/name)
    data=read(root/'dataset.json');protocol=read(root/'protocol.json')
    save(destination/'dataset-public.json',{
        'queries':data['queries'],'answer_ids':data['answer_ids'],'provenance':data['provenance'],
        'corpus_records':len(data['records']), 'corpus_generator':'experiments.agy_cases.build',
        'full_source_sha256':protocol['dataset_sha256']})
    pools=[];needed={i for q in data['queries'] for i in q['expected_ids']}
    for row in rows(root/'candidates.jsonl'):
        ids=[h['record_id'] for h in row['hits']];needed.update(ids)
        pools.append({'query_id':row['query_id'],'ids':ids,'original_section_sha256':[
            hashlib.sha256(json.dumps(h['evidence'],ensure_ascii=False,sort_keys=True).encode()).hexdigest() for h in row['hits']]})
    save(destination/'candidate-pools.json',pools)
    save(destination/'synthetic-source-subset.json',[r for r in data['records'] if r['_record_id'] in needed])
    allowed=('id','head','created_at','model_stop_at','delivery_deadline','deadline_policy','source_hashes',
        'dataset_sha256','model','effort','K','references','workers','dev','test','regression','test_clusters','dev_clusters',
        'test_answer_ids','answer_plan','maximum_answer_attempts','maximum_generation_calls','maximum_followup_calls',
        'transport_retries','validation_repair','controls','gpu_mode','gpu_order','stopping','statistics','metrics','privacy','old_results',
        'resume_source_run','resume_source_protocol_sha256','execution_revision','cli_timeout_seconds','budget_calculation','new_sources_uncommitted_at_freeze','dirty_diff_sha256',
        'operational_revision_only','test_rankings_seen_before_this_revision','previous_failed_scope_attempts_retained','remote_cancel_verified')
    save(destination/'protocol-public.json',{k:protocol[k] for k in allowed if k in protocol})
    for name in ('inventory-during-run.json','input-audit.json','execution-audit.json'):
        if (root/name).exists():shutil.copyfile(root/name,destination/name)
    # 연결 시험은 코호트 밖이며 본문·대화 식별자는 공개하지 않습니다.
    probes=[]
    for path in sorted((root/'agy-traces').glob('dev-*.metadata.json')):
        r=read(path)
        probes.append({'probe_key':path.stem.removesuffix('.metadata'),'model':r.get('confirmed_model'),
            'wall_seconds':r['wall_seconds'],'provider_seconds':public_seconds(r.get('provider_duration')),'usage':public_usage(r.get('usage')),
            'error_type':error_type(r.get('error')), 'exit':r.get('exit')})
    for name in ('probe.stdout.json','structured-probe.stdout.jsonl'):
        path=root/name
        if not path.exists():continue
        if name.endswith('.jsonl'):
            results=[e['result'] for line in path.read_text().splitlines() if (e:=json.loads(line)).get('event')=='result']
        else:results=[read(path)]
        for r in results:
            probes.append({'probe_key':name,'wall_seconds':None,'provider_seconds':public_seconds(r.get('duration_seconds')),
                'usage':public_usage(r.get('usage')),'status':r.get('status'),'scope':'standalone preflight; exact CLI wall unavailable in saved metadata'})
    for name in ('gpu-probe.stdout.log','gpu-wrapper-probe.stdout.log'):
        path=root/name
        if not path.exists():continue
        for line in path.read_text().splitlines():
            if not line.startswith('RESULT:'):continue
            r=json.loads(line[7:])
            probes.append({'probe_key':name,**{k:r.get(k) for k in ('load_seconds','compute_seconds','device_peak_bytes','peak_rss_bytes','backend','device')},
                'scope':'GPU preflight outside cohort; not a question in TEST'})
    save(destination/'preflight-costs.json',probes)
    outcomes=read(public/'answer-outcomes.json')
    answers_by_type=[]
    for category in sorted({q['category'] for q in data['queries'] if q['split']=='TEST'}):
        for k in (5,10,20):
            selected=[r for r in outcomes if r['split']=='TEST' and r['category']==category and r['K']==k]
            answers_by_type.append({'category':category,'K':k,'planned':8,'checkpointed':len(selected),
                'provided':sum(r['status']=='ok' for r in selected),'error':sum(r['status']=='error' for r in selected),
                'absence_abstained':sum(r['diagnostic'].get('abstained') is True for r in selected),
                'sql_boundary_matched':sum(r['diagnostic'].get('procedural_match') is True for r in selected),
                'literal_complete':sum(r['diagnostic']['literal_complete'] is True for r in selected),
                'literal_unknown':sum(r['diagnostic']['literal_complete'] is None for r in selected)})
    write_csv(destination/'answers-by-type.csv',answers_by_type)
    with (destination/'report.md').open('a',encoding='utf-8') as report:
        report.write('\n## TEST 답변 유형별 분모\n\n제공은 정답을 뜻하지 않는다. 근거 없는 질문의 명시적 거절과 SQL/미지원 조건의 절차 일치를 별도로 센다. 문자 진단의 unknown은 의미 오답으로 바꾸지 않는다.\n\n|유형|K|기록/계획|제공|오류|근거 없음 거절|SQL 경계 일치|문자 충족|문자 unknown|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        for r in answers_by_type:
            report.write(f"|{r['category']}|{r['K']}|{r['checkpointed']}/{r['planned']}|{r['provided']}|{r['error']}|{r['absence_abstained']}|{r['sql_boundary_matched']}|{r['literal_complete']}|{r['literal_unknown']}|\n")
        scopes=read(public/'scope-outcomes.json')
        report.write('\n후속 질문의 고정 대화 대상 해석: ')
        for split,planned in [('DEV',4),('TEST',20)]:
            selected=[r for r in scopes if r['split']==split]
            report.write(f"{split} 기대 주소 일치 {sum(r['matches_expected_scope'] for r in selected)}/{len(selected)}기록(계획{planned}); ")
        report.write('실제 사용자의 연속 대화 성과가 아니다.\n')
    # 성과를 보고 가장 유리한 사례를 고르지 않도록 첫 ID 규칙을 사용합니다.
    changes=list(csv.DictReader((public/'changes.csv').read_text(encoding='utf-8-sig').splitlines()))
    questions={q['id']:q for q in data['queries']}
    rankings={(r['query_id'],r['K']):r for r in read(public/'ranking-outcomes.json')}
    originals={r['_record_id']:r for r in data['records']}
    cases=['# 고정 규칙으로 고른 설명 사례','',
        '각 개선·악화·유지에서 (질문 ID, K) 순으로 첫 사례를 선택합니다. 표본 전체 결과를 대체하지 않습니다. 원시 답변은 공개하지 않습니다.','']
    for label,delta in [('개선',1),('악화',-1),('유지',0)]:
        selected=sorted((r for r in changes if int(r['delta'])==delta and questions[r['query_id']]['category']!='exact'),key=lambda r:(r['query_id'],int(r['K'])))
        cases.extend([f'## {label}',''])
        if not selected:
            cases.extend(['해당 사례가 없거나 비교할 결과가 아직 없습니다.','']);continue
        row=selected[0];qid=row['query_id'];k=int(row['K']);q=questions[qid]
        first=lambda n:next(iter(rankings[qid,n]['ranked_ids']),'없음')
        cases.extend([f"질문 `{qid}`: {q['question']}",'',f"후보 5건의 첫 ID: `{first(5)}`",'',
            f"후보 {k}건의 첫 ID: `{first(k)}`",'',f"실버 기대 ID: {', '.join('`'+i+'`' for i in q['expected_ids'])}",''])
        for identity in dict.fromkeys(rankings[qid,5]['ranked_ids'][:1]+rankings[qid,k]['ranked_ids'][:1]+q['expected_ids']):
            r=originals[identity];cases.append(f"- `{identity}`: "+' · '.join(f'{c}: {r.get(c,"")}' for c in ('설비','기인 Unit(부품)','작업처','현상','조치')))
        cases.append('')
    (destination/'cases.md').write_text('\n'.join(cases).rstrip()+'\n',encoding='utf-8')
    shutil.copyfile(Path(__file__).resolve().parents[1]/'experiments/agy_verify.py',destination/'verify_results.py')
    (destination/'README.md').write_text('''# AGY 임시 모델 · 질문 유형 확대 평가

기준 2026-10-14 KST. [전체 결과](report.md)의 완료·실패·미실행 분모를 먼저 확인하세요.

신규 TEST 200문항(10유형×20), DEV 40문항, 이전 회귀 24문항을 분리했습니다. TEST는 하드웨어·생성 규칙 20군이며 독립 현장 사건 200개가 아닙니다. 전부 창작 합성 원문·실버 라벨이고 사람 평가를 수행하지 않았습니다.

```mermaid
flowchart LR
    Q[질문·라벨·원문 고정] --> H[공통 검색 후보20건]
    H --> C[일반 질문 · Clef MLX GPU 후보5 / 10 / 20]
    H --> P[정확 조회 · 후속 주소 한정]
    H --> SQL[SQL 집계 · 미지원 안내]
    P --> A
    SQL --> R
    C --> A[일반 최대3근거 · 임시 AGY Gemini3.8]
    A --> V[기존 원문 검사 · 보정 최대1회]
    V --> R[실패·unknown·비용을 포함한 집계]
```

- [질문·실버 라벨](dataset-public.json) · [후보 ID 목록](candidate-pools.json) · [창작 원문 부분집합](synthetic-source-subset.json)
- [유형별 순위](ranking-by-type.csv) · [답변 시도](answers.csv) · [질문별 변화](changes.csv) · [짝지은 차이](paired-deltas.json)
- [TEST 답변 유형별 과제·실패·unknown](answers-by-type.csv)
- [같은 초기 RAG 입력의 공통 CLI 시도](matched-inputs.csv) · 지문은 원시 입력이나 제공자 내부 문맥이 아닙니다.
- [순위 결과](ranking-outcomes.json) · [답변 진단](answer-outcomes.json) · [후속 대상 해석](scope-outcomes.json)
- [고정 선택 규칙의 개선·악화·유지 사례](cases.md)
- [실행 비용](cost.json) · [코호트 외 연결 시험 비용](preflight-costs.json) · [고정 조건](protocol-public.json) · [파일 해시](manifest.json)

이 문서가 있는 결과 폴더에서 표준 라이브러리만으로 저장 수치·분모·후보 불변·파일 해시를 다시 계산합니다.

```sh
python verify_results.py
```

이는 모델 추론이나 라벨의 현장 타당성을 다시 검증하는 명령이 아닙니다. 전체 corpus는 `experiments.agy_cases.build`로 생성하며 공개 원문 파일은 후보·기대 근거에 해당하는 부분집합입니다. 모델 실행기는 Mac 절대 경로·이미 설치된 모델·AGY 계정에 의존합니다. 이식 가능한 Windows 재현 패키지로 제시하지 않습니다.

합성 원문을 직접 등록·색인했고 LLM 다듬기·영어 보완 자료 생성은 실행하지 않았습니다. 다국어 결과는 영어·중국어 원문 직접 검색 조건입니다. 운영의 선택적 번역 보완까지 적용한 품질이나 실제 Excel/DRM 입력 처리를 검증한 것이 아닙니다.

일반 답변은 근거 최대3건, 특정 후속 질문은 해석된 원문 주소만 사용하며 집계는 LLM을 호출하지 않습니다. 공개 결과의 `source_ids`는 선택 목록이므로 CLI 진입이 없는 행을 실제 전송 근거로 해석하지 않습니다. 실제 요청과 원문 지문은 로컬 증거로 보존합니다.

Clef는 별도 MLX 8bit 상주 프로토타입입니다. 운영의 매 요청 적재·교대 지연과 다릅니다. AGY의 실제 모델 ID와 high effort를 확인했지만 temperature·최대 출력·문맥 길이는 CLI로 강제하지 못합니다. 로컬 GPT-OSS와 동등한 배포 조건의 성능 비교가 아닙니다. 후속 질문은 창작한 고정 대화를 사용했습니다.

RAG의 역할별 메시지를 CLI 프롬프트의 JSON으로 감쌌으며 제공자의 native 역할 API와 다릅니다. `finish_reason=stop`은 CLI 성공·구조화 결과에 따른 어댑터 계약으로, 원래 제공자의 종료 신호가 아닙니다. CLI 내부 모델 호출 횟수도 직접 통제하지 못합니다. CLI 실행 수와 제공자 사용량·시간을 구분하며 원시 API 한 번의 성능으로 발표하지 않습니다.

합성 원문에는 2026-10-14 이후의 가상 날짜와 어색한 상태 표현이 있습니다. 상태의 예정 문구에 과거형 동사가 함께 있는 경우도 있어 해당 실버 라벨을 독립적인 의미 정답으로 보지 않습니다. 물리적으로 검증한 고장·측정 자료나 시점 검증 자료가 아닙니다. 실행 중 자료를 고쳐 유리한 결과를 만들지 않았습니다.

원시 LLM 요청·응답·추론·대화 ID·개인 설정·운영 DB·실제 CS·인증 정보·가중치·가상환경은 포함하지 않습니다. 답변 진단은 원문 주소와 제한적 문자 일치이며 의역은 unknown입니다. 검사 통과를 의미 정답률로 바꾸지 않습니다. 현장 효과·Windows/120B·사람 공수 절감·Clef 고유 우위는 이번 평가로 확인할 수 없습니다.

모든 답변 구성은 `claim_guard_mode=off`이며 규칙 기반 원문·인용·수치 검사를 사용합니다. 선택적 의미 검증 모델을 실행한 결과가 아닙니다.

연결 시험의 알려진 제공자 시간·GPU 적재/계산 비용을 별도 표에 남겼습니다. 초기 단독 CLI 시험 일부는 정확한 wall 측정값이 저장되지 않아 `null`이며, 전체 비용을 빠짐없이 실측했다거나 이를 0초로 간주하지 않습니다.

CLI 시간 제한은 원격 턴의 부분 결과를 반환할 수 있습니다. 원격 턴 취소·남은 처리·전체 과금 비용은 확인하지 못했습니다. CLI 종료를 원격 모델 계산 종료로 해석하지 않습니다. 최초 기준 HEAD에는 새 실험 파일이 아직 커밋되지 않았으므로 당시 새 코드의 해시는 프로토콜에 따로 기록했습니다.
''',encoding='utf-8')
    save(destination/'manifest.json',{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(destination.iterdir()) if p.is_file()})
    verify(destination)


if __name__=='__main__':publish(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
