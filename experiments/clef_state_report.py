"""상태 의미 진단의 종료 기록만 집계합니다. 추론·답변 생성은 하지 않습니다."""
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
import sys

from experiments.clef_state_diagnostic import checked, rows, sha, verify


def metrics(selected, baseline_accept):
    labels = Counter(r['label'] for r in selected)
    return {'n': len(selected), 'supported_n': labels['supported'], 'contradicted_n': labels['contradicted'],
        'insufficient_n': labels['insufficient'],
        'contradiction_held': sum(r['label'] == 'contradicted' and not r['accept'] and not r['failed'] for r in selected),
        'baseline_missed_contradictions': sum(r['label'] == 'contradicted' and baseline_accept[r['case_id']] for r in selected),
        'additional_contradiction_detection': sum(r['label'] == 'contradicted' and baseline_accept[r['case_id']] and not r['accept'] and not r['failed'] for r in selected),
        'additional_insufficient_hold': sum(r['label'] == 'insufficient' and baseline_accept[r['case_id']] and not r['accept'] and not r['failed'] for r in selected),
        'normal_false_hold': sum(r['label'] == 'supported' and not r['accept'] and not r['failed'] for r in selected),
        'normal_hold_including_check_failure': sum(r['label'] == 'supported' and not r['accept'] for r in selected),
        'insufficient_wrong_pass': sum(r['label'] == 'insufficient' and r['accept'] for r in selected),
        'check_failure': sum(r['failed'] for r in selected),
        'semantic_unclassified': sum(r['semantic'] is None for r in selected),
        'semantic_confusion': {gold: {pred: sum(r['label'] == gold and r['semantic'] == pred for r in selected)
            for pred in ('supported', 'contradicted', 'insufficient')} for gold in ('supported', 'contradicted', 'insufficient')}}


def table(path, data):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(data)


def seconds(value):
    return f'{value:.3f}' if value is not None else '미측정'


def publish(root, destination):
    protocol = verify(root); status = json.loads((root/'status.json').read_text())
    assert status['phase'] in ('finished', 'finished_with_errors'), 'Wait for the owned run to finish'
    results = rows(root/'results.jsonl'); cases = json.loads((root/'cases.json').read_text())
    assert len(results) == len({(r['case_id'], r['arm']) for r in results}) == 153
    assert {r['case_id'] for r in results} == {c['id'] for c in cases}
    for r in results:
        case = next(c for c in cases if c['id'] == r['case_id'])
        assert all(r[k] == case[k] for k in ('split', 'group_id', 'label'))
    base = {r['case_id']: r['accept'] for r in results if r['arm'] == 'rules'}
    summaries = [{'split': split, 'arm': arm, **metrics([r for r in results if r['split'] == split and r['arm'] == arm], base)}
        for split in ('DEV', 'TEST', 'REGRESSION') for arm in ('rules', 'state_rules', 'clef')]
    raw = [json.loads(s[7:]) for s in (root/'resident-stdout.log').read_text().splitlines() if s.startswith('RESULT:')]
    costs = rows(root/'worker-costs.jsonl'); handoff = rows(root/'handoff-costs.jsonl')
    for index, item in enumerate(handoff):
        cold = next((c for c in costs if c['key'] == f'cold-{index}'), None)
        item['clef_load_compute_cleanup_wall'] = cold['wall_seconds'] if cold else None
        path = root/f'cold-{index}-stdout.log'
        saved = [json.loads(s[7:]) for s in path.read_text().splitlines() if s.startswith('RESULT:')] if path.exists() else []
        try:
            assert len(saved) == 1
            item['classification'] = checked(saved[0]); item['classification_failed'] = False
        except (AssertionError, ValueError, KeyError, TypeError):
            item['classification'] = None; item['classification_failed'] = True
        values = [item['answer_release'], item['answer_restore'], item['clef_load_compute_cleanup_wall']]
        item['incremental_switch_seconds'] = sum(values) if all(v is not None for v in values) else None
        ready = [item['answer_release'], item['clef_load_compute_cleanup_wall']]
        item['inspection_ready_seconds'] = sum(ready) if all(v is not None for v in ready) else None
        warm = next(r for r in results if r['case_id'] == item['case_id'] and r['arm'] == 'clef')
        item['same_decision_as_resident'] = item['classification'] == warm['semantic'] if not item['classification_failed'] else None
    timings = {}
    for split in ('DEV', 'TEST', 'REGRESSION'):
        ids = {c['id'] for c in cases if c['split'] == split}
        values = sorted(r['compute_seconds'] for r in raw if r['key'] in ids and 'compute_seconds' in r)
        timings[split] = {'samples': len(values), 'median': statistics.median(values) if values else None,
            'p95': values[int(.95*(len(values)-1))] if values else None}
    all_lms = rows(root/'lms-costs.jsonl')
    cost = {'resident': next(c for c in costs if c['key'] == 'resident'), 'worker_attempts': costs,
        'inference_by_split': timings, 'resident_load_seconds': sum(r.get('load_seconds', 0) for r in raw),
        'resident_compute_sum': sum(r.get('compute_seconds', 0) for r in raw),
        'resident_model_calls': len(raw), 'handoff': handoff,
        'all_measured_worker_wall': sum(c['wall_seconds'] for c in costs),
        'all_measured_lms_control_wall': sum(c['seconds'] for c in all_lms),
        'gpu_allocated_peak_bytes': max((r.get('device_peak_bytes', 0) for r in raw), default=None),
        'process_rss_peak_bytes': max((r.get('peak_rss_bytes', 0) for r in raw), default=None),
        'capability_probe': json.loads((root/'capability-cost.json').read_text()),
        'run_wall_seconds': status['wall_seconds'], 'actual_answer_generation_calls': 0,
        'early_cli_discovery_delay': None, 'production_browser_latency': None}
    destination.mkdir(parents=True, exist_ok=False)
    safe_rows = [{key: r.get(key) for key in ('case_id', 'split', 'group_id', 'label', 'arm', 'accept', 'semantic', 'failed', 'reason', 'compute_seconds')}
                 | {'rule_seconds': r['seconds'], 'classifier_attempted': r['arm'] == 'clef' and base[r['case_id']]} for r in results]
    table(destination/'decisions.csv', safe_rows)
    table(destination/'metrics.csv', [{k: v for k, v in r.items() if k != 'semantic_confusion'} for r in summaries])
    for name, value in [('metrics.json', summaries), ('cost.json', cost), ('cases-public.json', cases),
                        ('protocol-public.json', protocol), ('status.json', status)]:
        (destination/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    text = ['# Clef 답변 상태 의미 검사 · 소규모 진단', '', '2026-10-15 KST · 운영 미연결 · 합성/실버 라벨', '',
        '기존 운영 코드·설정·발표 파일은 변경하지 않았다. 답변 문장과 그 문장이 실제 인용한 원문 셀만 입력했다. 다른 검색 후보·질문·영어 번역·외부 지식으로 답변을 정당화하지 않았고, 답변 재생성·자동 수정·운영 채택은0회다.', '',
        '```mermaid', 'flowchart LR',
        '    A[고정 답변 문장과 실제 인용 셀] --> B[기존 규칙 검사]',
        '    B --> O[기존 규칙 구성의 통과·거절 기록]',
        '    B -->|통과| R[간단한 상태 규칙 보강 구성]',
        '    B -->|통과| C[Clef 세 범주 구성 · MLX GPU]',
        '    C --> S[원문이 뒷받침함 · 허용]',
        '    C --> X[원문과 충돌함 · 보류]',
        '    C --> U[근거 부족 또는 모호함 · 보류]',
        '    C --> E[검사 실패 · 별도 기록 후 보류]', '```', '',
        '## 입력 지원과 사전 고정', '',
        '이미 설치된 Clef-flash MLX8bit 로더의 해시를 확인했다. 현재 로더의 native type은noul/choice/score이며, classification이라는 새 type을 흉내 내지 않고 choice의 세 범주를 사용했다. 비채점3입력에서 세 범주 구조·MLX GPU·잘림 없음·출력 토큰0을 확인했다. 이3개를 평가 정확도 표본으로 더하지 않았다. 모델 설치·다운로드는 없었다.', '',
        f"고정 프로토콜 SHA256 `{sha(root/'protocol.json')}`, 자료 SHA256 `{protocol['cases_sha256']}`. DEV4원문묶음/12문장과 TEST12원문묶음/36문장을 분리하고 모든 사실·라벨·지시문·규칙·argmax·처리 기준을 평가 전에 고정했다. DEV 이후 지시문·임계값·규칙을 고치지 않았다. 저자가 만든 실버 라벨이며 독립 담당자 판정이 아니다. 같은 원문에서 파생한3문장을 독립 현장 사건3개로 세지 않는다.", '',
        '원문이 직접 뒷받침하면supported, 명시적 반대 사실이면contradicted, 정보 부족·모호함이면insufficient이다. 계획만으로 시행/미시행을 단정하지 않고, 결과 미기재를 성공/실패로 바꾸지 않는다. 확률은 세 선택지 안의 상대값이며 보정된 진실 확률이 아니다.', '',
        '## 비교군과 TEST 결과', '',
        '기존 규칙은 현재 validate_claims를 수정 없이 호출하되, 비교 공정성을 위해 실제 인용 셀만 전달했다. 운영의 전체 후보 묶음 입력을 재현한 것은 아니다. 규칙 통과에는 의미 삼분류가 없으므로semantic=null을supported로 바꾸지 않았다. 상태 규칙 보강은 제한된 상태 단어 패턴으로 명시적 충돌/부족을 보류하고 나머지는 기존 통과를 유지한다. 기존 규칙+Clef는 기존 통과 건에만 모델을 호출하고supported만 허용한다. 검사 실패는 별도 보류이며 오류 탐지 성공으로 세지 않는다.', '',
        '|구성|충돌 차단/12|기존 누락 중 추가 충돌 탐지|정상 오차단/12|근거 부족의 잘못된 통과/12|검사 실패/36|',
        '|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        if r['split'] == 'TEST':
            name = {'rules': '기존 규칙', 'state_rules': '상태 규칙 보강', 'clef': '기존 규칙 + Clef'}[r['arm']]
            text.append(f"|{name}|{r['contradiction_held']}|{r['additional_contradiction_detection']}/{r['baseline_missed_contradictions']}|{r['normal_false_hold']}|{r['insufficient_wrong_pass']}|{r['check_failure']}|")
    text += ['', '정상 오차단과 충돌 차단은 검사 실패를 제외한 실버 라벨 대조다. 정상 보류에 검사 실패까지 포함한 수와 삼분류 혼동 행렬은metrics.json에 있다. 규칙의 미판정은 의미상 근거 부족 판정과 다르다. 유형별 표현과 실제 모델 판단·gate 결과는decisions.csv와cases-public.json에서 함께 확인할 수 있다.', '',
        '기존 규칙을 통과했던 근거 부족10건 중 상태 규칙은4건, Clef는10건을 추가 보류했다. 이를 명시적 충돌 탐지와 합쳐 동일 의미 판정으로 발표하지 않는다.', '',
        '## 삼분류와 통과/보류의 차이', '',
        '위11건은충돌 라벨 문장을 비허용한 수이며, 모두를정확히contradicted로 분류했다는 뜻이 아니다. 기존 규칙이근거 부족2건을 먼저 차단해 TEST모델 호출은34회다. 모델이 실제로 내린 세 범주를 분리하면 다음과 같다.', '',
        '|실버 라벨|supported 판정|contradicted 판정|insufficient 판정|모델 미호출|검사 실패|',
        '|---|---:|---:|---:|---:|---:|']
    combined = next(r for r in summaries if r['split'] == 'TEST' and r['arm'] == 'clef')
    for gold in ('supported', 'contradicted', 'insufficient'):
        counts = combined['semantic_confusion'][gold]
        selected = [r for r in results if r['split'] == 'TEST' and r['arm'] == 'clef' and r['label'] == gold]
        skipped = sum(not base[r['case_id']] for r in selected)
        failed = sum(base[r['case_id']] and r['failed'] for r in selected)
        text.append(f"|{gold}|{counts['supported']}|{counts['contradicted']}|{counts['insufficient']}|{skipped}|{failed}|")
    text += ['', '충돌을근거 부족으로판정한1건과근거 부족을충돌로판정한2건도오분류다. 부족근거 오통과0건을 삼분류가모두맞았다는결과로 바꾸지 않는다. 정상 오차단0/12도 일반적 무오류 보장이 아니다.', '',
        '## 고정 선택 규칙의 설명 사례', '',
        '각 조건에 해당하는 TEST 문장 중 case_id 사전순 첫 사례다. 없는 사례는 만들지 않는다. 선택은 사후 설명용이며 새로운 성능 표본이 아니다.', '']
    by_id = {c['id']: c for c in cases}
    for name, condition in [('Clef 추가 충돌 보류', lambda r: r['arm'] == 'clef' and r['label'] == 'contradicted' and base[r['case_id']] and not r['accept'] and not r['failed']),
                            ('Clef 충돌 미탐지·오통과', lambda r: r['arm'] == 'clef' and r['label'] == 'contradicted' and r['accept']),
                            ('Clef 근거 부족을 충돌로 오분류', lambda r: r['arm'] == 'clef' and r['label'] == 'insufficient' and r['semantic'] == 'contradicted'),
                            ('Clef 정상 오차단', lambda r: r['arm'] == 'clef' and r['label'] == 'supported' and not r['accept'] and not r['failed']),
                            ('Clef 근거 부족 오통과', lambda r: r['arm'] == 'clef' and r['label'] == 'insufficient' and r['accept']),
                            ('상태 규칙 정상 오차단', lambda r: r['arm'] == 'state_rules' and r['label'] == 'supported' and not r['accept'] and not r['failed'])]:
        selected = sorted((r for r in results if r['split'] == 'TEST' and condition(r)), key=lambda r: r['case_id'])
        if not selected:
            text.append(f'- {name}: 해당 사례 없음.')
        else:
            r = selected[0]; c = by_id[r['case_id']]
            text.append(f"- {name} `{c['id']}`: 답변 「{c['claim']}」 / 실제 인용 셀 「" + ' | '.join(v['text'] for v in c['cited_cells']) + f"」 / 라벨 {c['label']} / 판정 {r['semantic']}.")
    text += ['', '## DEV와 기존3건 회귀 · 주 결과와 분리', '']
    for r in summaries:
        if r['split'] != 'TEST':
            normal = f"{r['normal_false_hold']}/{r['supported_n']}" if r['supported_n'] else '정상 표본 없음'
            text.append(f"- {r['split']} / {r['arm']}: n={r['n']}, 충돌 차단{r['contradiction_held']}/{r['contradicted_n']}, 정상 오차단{normal}, 근거 부족 오통과{r['insufficient_wrong_pass']}/{r['insufficient_n']}, 검사 실패{r['check_failure']}.")
    text += ['', '기존 B09·B10·B11은 이전 실험이 주입한 상태 오류이며 실제 모델이 자연 발생시킨 오류3건이 아니다. 원래 답변 문장과 실제 인용 셀을 보존하고 다른 상세/열은 제외했다. 계획→시행과 미기재→정상은insufficient, 명시적 미실시→실시는contradicted로 사전 구분했다. 신규 TEST 분모나 개선 증거에 합치지 않았다.', '',
        '## 순수 추론과 적재·교대 비용', '',
        f"상주 프로토타입에서 실제 모델 호출{len(raw)}회. 기존 규칙이 거절한 건은 모델을 호출하지 않았다. 적재{cost['resident_load_seconds']:.3f}초, 계산 합{cost['resident_compute_sum']:.3f}초, 자식 시작부터 종료까지wall {cost['resident']['wall_seconds']:.3f}초다. 적재·계산은wall에 포함되며 다시 합산하지 않는다.", '']
    for split, t in timings.items():
        text.append(f"- {split}: 실제 분류{t['samples']}회, 순수 계산 중앙{seconds(t['median'])}초, p95 {seconds(t['p95'])}초. 적은 표본의 p95를 안정된 서비스 지연으로 해석하지 않는다.")
    text += ['', '다음은 고정한DEV의 동일3입력을 사용한 별도 비용 측정이다. 소유한 진단 전용 GPT-OSS20B 별칭만16K·parallel1로 적재/해제했다. 실제 답변 생성은 없었다. 기존 사용자 모델을 중지하거나 서비스 설정을 바꾸지 않았다. cold는Clef별도 프로세스의 시작·적재·분류·종료 전체다. 추가 교대 비용은답변 모델 해제+cold+답변 모델 재적재이며, 최초 답변 모델 적재와 최종 별칭 해제는 준비/정리 비용으로 따로 보존했다.', '']
    for h in handoff:
        text.append(f"- {h['case_id']}: 최초 적재{seconds(h['setup_load'])}초 / 해제{seconds(h['answer_release'])}초 / cold {seconds(h['clef_load_compute_cleanup_wall'])}초 / 재적재{seconds(h['answer_restore'])}초 / 최종 정리{seconds(h['cleanup_unload'])}초 / 검사 준비 완료까지{seconds(h['inspection_ready_seconds'])}초 / 답변 모델 복원을 포함한 교대 합{seconds(h['incremental_switch_seconds'])}초 / 오류{h['error']} / 분류 실패{h['classification_failed']} / 상주 판정과 동일{h['same_decision_as_resident']}.")
    text += ['', f"실행기 전체wall {status['wall_seconds']:.3f}초(분류+비용 실험+상태 확인 포함). worker wall 합{cost['all_measured_worker_wall']:.3f}초, LM Studio 제어 합{cost['all_measured_lms_control_wall']:.3f}초. 이 부분합들을 전체wall에 또 더하지 않는다. 비채점 API 탐침 비용{cost['capability_probe']['wall_seconds']:.3f}초는 별도다. 초기 읽기 전용CLI 환경 탐색의 연결 지연은 정밀 계측하지 않았으며 모델 추론 비용이나0초로 간주하지 않는다.", '',
        f"추론 구간의 MLX GPU allocated 피크{cost['gpu_allocated_peak_bytes']}bytes, 분류 자식RSS 피크{cost['process_rss_peak_bytes']}bytes. 적재 도중의 GPU 메모리 피크는 별도 계측하지 않았다. unified memory를 중복 합산하지 않는다. 답변 모델은 적재/해제만 측정했으며 답변 생성 지연·GPU피크·실제 브라우저 전체 응답 시간은 미측정이다. 현재 서비스에 연결한 운영 실측이 아닌 격리된 교대 진단이다.", '',
        '## 해석 범위', '',
        '이 진단은 현재 로컬 Clef가 인용 셀에 근거한 상태 삼분류를 실제로 수행할 수 있는지와, 규칙 대비 추가 탐지·오차단·부족 근거 통과·비용을 관찰한다. 작은 균형 합성자료·실버 라벨이며 현장 정확도나 자동 채택 근거가 아니다. 미탐지·오차단·실패를 모두 보존하고, 이 TEST를 고쳐 새로운 독립 성과로 발표하지 않는다. 종료 뒤 보고서의 혼동 행렬과 설명 사례만 정리했으며 모델 입력·규칙·라벨·처리 기준은 바꾸지 않았다.', '',
        '모델 원시 요청/응답·추론 로그·실제 CS·DB·설정·가중치·가상환경은 공개하지 않는다. 공개 cases는창작 자료이며decisions는판정 집계용 파생 결과다. 운영 코드나 발표 파일은 수정하지 않았다.', '',
        '추가 교대 합은 계측한 구성요소의 합으로 제어 전 상태 조회는 포함하지 않는다. 답변 모델 복원까지 기다려야 검사 결과를 화면에 제공할 수 있는지는 운영에 연결하지 않아 확인하지 않았다. 검사 준비 완료까지의 해제+cold 비용과 완전 교대 비용을 구분한다. 전체 실행기wall을 따로 보존했다. CSV의rule_seconds는규칙 시간이고compute_seconds는추가 GPU 계산 시간이며 둘을 전체 사용자 지연으로 해석하지 않는다. LM Studio의 답변 모델 배치는 자동 설정이며, 이를 별도 검증한 전층 GPU 추론 성능으로 설명하지 않는다.', '',
        '[문장별 판정153행](decisions.csv) · [집계](metrics.csv) · [삼분류 혼동](metrics.json) · [고정 창작 원문과 라벨](cases-public.json) · [사전 프로토콜](protocol-public.json) · [비용](cost.json) · [검증 범위](audit.md) · [종료 후 환경 관측](inventory.json)', '']
    (destination/'README.md').write_text('\n'.join(text))
    print('PUBLISHED', destination)


if __name__ == '__main__':
    publish(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
