"""합성 CS 평가 10유형. DEV/TEST는 사건·하드웨어·질문 문구 계열을 분리합니다."""
import json
from pathlib import Path
from ragkit.input_formats import HISTORY_COLUMNS, HISTORY_SHEET
from ragkit.records import record_id
from experiments.relation_quality import fixture

KINDS = ('symptom', 'condition', 'detail', 'state', 'numeric', 'language', 'absent', 'exact', 'aggregate', 'followup')
NAMES = ('현상과 조치', '운전 조건과 부정', '상세와 작업 구분', '조치 상태와 재발', '숫자와 단위', '표현과 언어', '근거 없음', '정확 조회', '집계와 지원 경계', '후속 질문')
# 각 하드웨어 계열은 한 split에만 둡니다. 반복되는 과제를 독립 사건으로 세지 않습니다.
DOMAINS = [
 ('식각장비','진공펌프','진공계통','배기 시간이 길어졌다','배기관 가스켓을 교체했다','목표 진공 도달 시간이 정상화되었다','pump down slow'),
 ('세정장비','순환펌프','냉각계통','순환수 유량이 낮아졌다','흡입 스트레이너를 세척했다','순환수 유량이 회복되었다','low coolant flow'),
 ('노광장비','냉각팬','광원냉각','가동 중 팬 소음이 커졌다','냉각팬 어셈블리를 교체했다','이상음이 사라졌다','fan noise'),
 ('도포장비','토출밸브','약액공급','밸브를 닫아도 액이 떨어졌다','토출밸브 시트를 교체했다','닫힘 후 잔류 토출이 멈췄다','valve dripping'),
 ('현상장비','분사노즐','분사계통','분사 패턴이 한쪽으로 치우쳤다','분사노즐을 세척하고 다시 장착했다','분사 패턴이 균일해졌다','uneven spray'),
 ('증착장비','배기댐퍼','배기계통','댐퍼 개방 동작이 늦어졌다','댐퍼 구동부 연결을 조정했다','개방 동작이 정상화되었다','damper delay'),
 ('연마장비','스핀들베어링','회전계통','회전 중 진동이 커졌다','스핀들 베어링을 교체했다','진동이 줄어 정상 범위가 되었다','spindle vibration'),
 ('검사장비','정렬센서','정렬계통','마크 검출이 간헐적으로 누락됐다','정렬센서 창을 세척했다','마크 검출이 정상화되었다','mark detection miss'),
 ('이송장비','승강브레이크','승강계통','정지 후 승강축이 처졌다','승강 브레이크를 교체했다','정지 후 처짐이 발생하지 않았다','axis sag'),
 ('건조장비','히터접촉기','가열계통','가열 출력이 간헐적으로 끊겼다','히터 접촉기를 교체했다','가열 출력이 안정되었다','heater intermittent'),
 ('열처리장비','도어실링','챔버도어','도어 주변에서 누설이 확인됐다','도어 실링을 교체했다','도어 주변 누설이 사라졌다','door seal leak'),
 ('세정반송장비','그리퍼패드','파지계통','기판 파지 중 미끄러짐이 발생했다','그리퍼 패드를 교체했다','파지 중 미끄러짐이 사라졌다','gripper slip'),
 ('약액공급장비','공급호스','약액라인','연결부에 액이 맺혔다','공급호스 연결부를 재조립했다','연결부 누수가 멈췄다','hose leak'),
 ('배기처리장비','차압센서','필터계통','차압 표시가 흔들렸다','차압센서 감지관을 세척했다','차압 표시가 안정되었다','pressure fluctuation'),
 ('반입장비','도킹핀','도킹계통','도킹 위치가 어긋났다','도킹핀 위치를 조정했다','도킹 위치가 정상화되었다','docking offset'),
 ('진공반송장비','슬릿밸브','진공이송','밸브 닫힘 신호가 늦게 들어왔다','슬릿밸브 위치센서를 조정했다','닫힘 신호가 정상적으로 확인됐다','close signal delay'),
 ('광학검사장비','조명전원','조명계통','검사 조명이 깜빡였다','조명 전원 모듈을 교체했다','검사 조명이 안정되었다','light flicker'),
 ('기판세척장비','배수밸브','배수계통','배수 완료가 늦어졌다','배수밸브 내부 이물을 제거했다','배수 시간이 정상화되었다','drain delay'),
 ('웨이퍼정렬장비','척구동벨트','척구동','원점 복귀 위치가 반복해서 어긋났다','척 구동벨트 장력을 조정했다','원점 복귀 위치가 안정되었다','home position error'),
 ('검사반송장비','엔코더케이블','위치계통','이동 중 위치 신호가 끊겼다','엔코더 케이블을 교체했다','이동 중 위치 신호가 유지되었다','encoder dropout'),
 # DEV 하드웨어는 TEST에 재사용하지 않습니다.
 ('검증로봇','감속기','구동부','회전 시 유격이 커졌다','감속기를 교체했다','회전 유격이 정상 범위가 되었다','gear backlash'),
 ('검증칠러','냉매필터','냉매부','냉매 공급이 원활하지 않았다','냉매필터를 교체했다','냉매 공급이 회복되었다','refrigerant restriction'),
 ('검증오븐','온도조절기','온도제어','설정 온도까지 올라가지 않았다','온도조절기를 교체했다','설정 온도에 도달했다','temperature low'),
 ('검증공급기','레귤레이터','압력제어','공급 압력이 흔들렸다','레귤레이터를 교체했다','공급 압력이 안정되었다','unstable pressure'),
]
CONDITIONS = [('기동 직후', '가동을 시작한 바로 뒤', '연속 가동 중'), ('자동 운전 중', '자동 모드로 운전할 때', '수동 운전 중'),
              ('정지 후', '설비를 멈춘 다음', '이동 중'), ('재가동 직후', '다시 가동하자마자', '가동 전 점검 중')]


def build(repo):
    background, _ = fixture()
    legacy = json.loads((Path(repo)/'docs/experiments/architecture-v3-20261014/synthetic-data.json').read_text())['records']
    for row in legacy:
        row.update(_id_column='CS NO', _source_sheet=HISTORY_SHEET, _history_detail=True)
        assert record_id(row) == row['_record_id']
    rows, queries = background + legacy, []
    for split, domains in [('DEV', range(20, 24)), ('TEST', range(20))]:
        for local, domain in enumerate(domains):
            equipment, part, module, symptom, action, outcome, english = DOMAINS[domain]
            base_equipment = ('가상평가' if split=='TEST' else '가상개발') + equipment
            for type_index, kind in enumerate(KINDS):
                equipment = ('가람','나래','다온','라온','마루','바른','새롬','아람','여울','하늘')[type_index] + base_equipment
                cs = f'AGY-{split}-{local:02}-{type_index:02}'
                condition, paraphrase, opposite = CONDITIONS[local % len(CONDITIONS)]
                originals = []
                for detail in range(4):
                    row = {k:'' for k in HISTORY_COLUMNS} | {'CS NO':cs if detail<3 else cs+'-OTHER',
                        '제목':part+' 점검 이력', '설비':equipment if detail<3 else '가상비교'+DOMAINS[(domain+1)%20][0],
                        '기인 모듈':module, '기인 Unit(부품)':part if detail!=2 else part+' 보조부품',
                        '작업처':'후속 정비' if detail==0 else '초도 정비' if detail==1 else '부속 작업',
                        '현상':condition+' '+symptom+'.' if detail!=1 else opposite+' '+symptom+'.',
                        '원인':'점검 기록에서 연결 상태 이상이 확인되었다.',
                        '조치':action+' '+outcome+'.' if detail!=1 else '동일 조치를 예정했으며 실제 작업은 실시하지 않았다.',
                        '불량발생일':f'2026-10-{local+1:02}', 'SITE':'창작평가사업장',
                        '_id_column':'CS NO','_source_sheet':HISTORY_SHEET,'_history_detail':True}
                    if detail==2:
                        row['현상']='보조부품 체결 상태를 점검했다.';row['조치']='보조부품 체결을 조정했다. 결과는 기록하지 않았다.'
                    originals.append(row)
                a,b,c,other=originals
                question=f'{equipment}의 {part}에서 {symptom}고 보고된 기록에서 실제로 수행한 조치와 확인된 결과를 알려줘.'
                if kind=='condition':
                    question=f'{equipment} {part}가 {paraphrase} 문제가 났던 이력의 조치와 결과는?'
                elif kind=='detail':
                    question=f'{equipment}의 {part} 후속 정비에서 기록된 조치와 결과만 알려줘. 초도 정비나 보조부품 작업과 섞지 말아줘.'
                elif kind=='state':
                    state=local%5
                    state_text=[action+' '+outcome+'.', action+' 작업은 예정이며 아직 실시하지 않았다.',
                                '조치를 실시하지 않았다. 결과는 확인하지 못했다.',action+' 잠시 정상화되었으나 동일 증상이 재발했다.',
                                action+' 조치 후 관찰 결과는 기록하지 않았다.'][state]
                    a['조치']=state_text
                    request=['실제로 조치하고 정상화가 기록된','조치가 예정 상태였던','조치를 실시하지 않았던','잠시 좋아졌다가 재발한','조치는 했지만 결과가 미기재인'][state]
                    question=f'{equipment} {part}에서 {request} 사례의 조치 상태와 결과를 설명해줘.'
                    b['조치']=action+' '+outcome+'.' if state else '조치 예정이며 미실시 상태이다.'
                elif kind=='numeric':
                    value,contrast=[('-0.4 mm','+0.4 mm'),('11.8 kPa','11.8 MPa'),('40 ℃','50 ℃'),('0.5 mm','5 mm')][local%4]
                    a['현상']+=' 점검 기록의 측정값은 '+value+'.';b['현상']+=' 점검 기록의 측정값은 '+contrast+'.'
                    question=f'{equipment} {part} 점검에서 {value}로 기록된 경우의 조치와 결과는?'
                elif kind=='language':
                    if local%4==0:a['현상']='Observed during operation: '+english+'.'
                    elif local%4==1:a['조치']='The connection was inspected and reassembled. The fault did not recur during the same-condition retest.'
                    elif local%4==2:a['조치']='已检查连接部并重新装配，复测后现象消失。'
                    question=f'{equipment} {part}에서 {english} 문제가 났던 이력의 실제 시행 조치와 확인된 결과를 한국어로 알려줘.' if local%2 else f'{equipment} {part}, 증상은 「{symptom}」입니다. 예전엔 실제로 어케 조치했고 확인된 결과는 어땠나요?'
                elif kind=='absent':
                    missing=['정전 후 레시피 메모리가 유실된','인증서 만료로 서버 접속이 차단된','백업 매체 손상으로 이력을 복구하지 못한','운영체제 부팅 영역이 손상된'][local%4]
                    question=f'{equipment} {part}에서 {missing} 경우에 성공이 확인된 복구 조치를 알려줘.'
                elif kind=='exact':
                    question=f'CS NO={cs}에 등록된 모든 상세의 현상과 조치·결과를 각각 보여줘.'
                elif kind=='aggregate':
                    question=f'CS NO="{cs if local%4==0 else cs+"-MISSING"}" 등록된 접수는 총 몇 건인가요?' if local%2==0 else f'{equipment}에서 2026년 10월에 재발한 접수는 몇 건인가요?'
                elif kind=='followup':
                    question='첫 번째 사례의 해당 부품에서 기록된 조치와 결과만 다시 알려줘.'
                if split=='DEV':
                    # 개발용 문구는 별도로 고정합니다. 공통 합성 생성 규칙 자체는 한계로 보고합니다.
                    alternatives={
                        'symptom':f'{equipment} 점검 대상 {part}: 원문에서 확인된 문제와 실제 수행 작업 및 확인된 경과를 정리하세요.',
                        'condition':f'{equipment} 점검 시 {condition}에 해당했던 {part} 기록을 찾아 처리 경과를 설명하세요.',
                        'detail':f'{equipment}의 {part} 정비 중 나중에 수행한 작업을 골라 그 작업 내용과 확인 결과만 제시하세요.',
                        'state':f'{equipment}의 {part} 이력 중 {request} 기록에 한정해 작업 상태와 확인 결과를 구분하세요.' if kind=='state' else '',
                        'numeric':f'{equipment} {part} 기록 중 측정값 {value}에 해당하는 처리 내역을 설명하세요.' if kind=='numeric' else '',
                        'language':f'{equipment} {part} 관련 {english} 이력의 실제 수행 작업과 확인된 경과를 한국어로 요약해 주세요.',
                        'absent':f'{equipment} {part}에서 {missing} 이력이 실제로 있는지, 있다면 근거와 복구 결과를 제시하세요.' if kind=='absent' else '',
                        'exact':f'접수번호 {cs}의 각 상세에 적힌 이상 현상과 처리 경과를 나누어 제시하세요.',
                        'aggregate':f'CS NO="{cs if local%4==0 else cs+"-MISSING"}" 접수 건수 알려주세요' if local%2==0 else f'{equipment} 최근 한 달 완료 접수는 몇 건인가요?',
                        'followup':'앞서 첫 번째로 설명한 부품의 처리 내용과 관찰 결과를 다시 설명하세요.'}
                    question=alternatives[kind]
                for row in originals:
                    row['_record_id']=record_id(row)
                rows.extend(originals)
                expected=[] if kind in ('absent','aggregate') else [a['_record_id'],b['_record_id'],c['_record_id']] if kind=='exact' else [a['_record_id']]
                q={'id':f'{split}-{kind}-{local:02}','split':split,'category':kind,'group_id':f'{split}-hardware-{local:02}',
                   'question':question,'expected_ids':expected,'contradicting_ids':[] if kind=='exact' else [b['_record_id'],c['_record_id'],other['_record_id']],
                   'required_action':a['조치'],'required_quote':a['현상'],'required_column':'현상',
                   'source_ids':[r['_record_id'] for r in originals], 'label_source':'authored synthetic/silver; not human or field labels'}
                if kind=='aggregate':q['expected_count']=(1 if local%4==0 else 0) if local%2==0 else None
                if kind=='followup':q['history_record_ids']=[a['_record_id'],c['_record_id']]
                queries.append(q)
    for row in rows:
        row['_record_id']=record_id(row)
    regression=json.loads((Path(repo)/'docs/experiments/candidate-budget-20261014/dataset-public.json').read_text())['queries']
    for q in regression:
        queries.append(q | {'split':'REGRESSION','category':'regression','group_id':'REGRESSION-'+q['group_id']})
    assert len(rows)==2998 and len({r['_record_id'] for r in rows})==2998
    assert len(queries)==264 and len({q['id'] for q in queries})==264
    available={r['_record_id'] for r in rows}
    assert all(set(q['expected_ids'])<=available for q in queries)
    assert len({q['group_id'] for q in queries if q['split']=='TEST'})==20
    return {'records':rows,'queries':queries,'answer_ids':[q['id'] for q in queries if q['split']=='TEST' and int(q['id'].rsplit('-',1)[1]) in {(KINDS.index(q['category'])*2+j*3)%20 for j in range(8)}],
            'provenance':'200 new TEST queries in20 hardware/template clusters;40DEV in4 disjoint clusters;24old regression only;all authored synthetic'}
