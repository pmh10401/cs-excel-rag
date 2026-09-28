"""접수 문서·채택문 비교·원문 근거 검사를 실제 SQLite에서 확인합니다."""
import json
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ragkit import cslist, records, storage, search, llm
from ragkit.evaluation import answer_checks


def main(root):
    # 서버 모델 별칭을 정확히 대조하고, 이론상 최대치 대신 실제 로드 한도를 사용합니다.
    cfg = {'model': 'alias', 'base_url': 'http://127.0.0.1:1234/v1', 'context_window': 131072}
    payload = {'models': [{'type': 'llm', 'key': 'openai/gpt-oss-120b', 'architecture': 'gpt-oss',
        'max_context_length': 131072, 'loaded_instances': [
            {'id': 'alias', 'config': {'context_length': 8192}}]}]}
    connection = Mock()
    connection.getresponse.return_value.status = 200
    connection.getresponse.return_value.read.return_value = json.dumps(payload).encode()
    with patch.object(llm, 'HTTPConnection', return_value=connection):
        effective, info = llm.answer_context_config(cfg)
        assert effective['context_window'] == 8192 and info['context_source'] == 'server'
        assert effective['answer_tokenizer'] == 'o200k_harmony'
        assert connection.request.call_args.args[:2] == ('GET', '/api/v1/models')
        assert llm.answer_context_config(cfg | {'context_window': 4096})[1]['context_window'] == 4096
        assert llm.answer_context_config(cfg | {'model': 'other'})[1]['context_source'] == 'config'
        connection.getresponse.return_value.status = 404
        assert llm.answer_context_config({'model': 'alias', 'base_url': cfg['base_url']})[1]['context_window'] is None
    with patch.object(llm, 'HTTPConnection', side_effect=TimeoutError):
        assert llm.answer_context_config(cfg)[1]['context_source'] == 'config'
    # 동봉 토크나이저는 네트워크 없이 로드됩니다. 특수 토큰처럼 보이는 원문도 평문입니다.
    encoder = llm._answer_encoder()
    assert encoder is not None and encoder.name == 'o200k_harmony'
    messages = [{'role': 'user', 'content': '압력 11.8 STEP <|endoftext|>'}]
    tokens, method = llm.answer_input_reservation(messages, {'answer_tokenizer': 'o200k_harmony'})
    assert method == 'o200k_harmony' and tokens == len(encoder.encode_ordinary(messages[0]['content'])) + 1024
    with patch.object(llm, '_answer_encoder', return_value=None):
        assert llm.answer_input_reservation(messages, {'answer_tokenizer': 'o200k_harmony'})[1] == 'utf8_bytes'
    # 자동 선택은 순위·셀 전체를 보존하며 경계 밖의 사례나 모델 응답은 잘라 넣지 않습니다.
    candidates = [{'id': str(i), 'evidence': [{'id': f'{i}/접수', 'kind': '접수',
        'cells': {'현상': '압력 11.8 STEP <|endoftext|> ' * 50, 'MTTR': ''}}]} for i in range(6)]
    automatic = {'model': 'alias', 'answer_tokenizer': 'o200k_harmony', 'max_context_chars': 100000}
    _, two_messages = llm.case_answer_messages('현상과 MTTR은?', candidates[:2])
    two_tokens = llm.answer_input_reservation(two_messages, automatic)[0]
    details = {}
    selected, budget_config = llm.select_answer_cases(automatic | {'context_window': two_tokens + 500,
        'answer_max_tokens': 500}, '현상과 MTTR은?', candidates, details=details)
    assert selected == candidates[:2] and details['references']['limited_by'] == 'tokens'
    assert llm.generation_options(budget_config, for_answer=True, messages=two_messages)['max_tokens'] == 500
    context, sent_messages = llm.case_answer_messages('현상과 MTTR은?', selected)
    assert json.loads(context) == [h['evidence'][0] for h in selected]
    assert sent_messages == two_messages
    char_limit = sum(len(m['content']) for m in two_messages)
    selected, _ = llm.select_answer_cases(automatic | {'max_context_chars': char_limit},
        '현상과 MTTR은?', candidates, details=details)
    assert selected == candidates[:2] and details['references']['context_source'] == 'unknown'
    assert details['references']['limited_by'] == 'characters'
    try:
        llm.select_answer_cases(automatic | {'max_context_chars': 10}, '현상은?', candidates, details=details)
    except ValueError as error:
        assert '한 건도' in str(error) and details['references']['selected'] == 0
    else:
        raise AssertionError('최상위 사례가 한도를 넘는데 일부 셀만 사용했습니다.')
    # 실패 응답이 길어도 원문을 자르거나 출력 여유를 줄이지 않고 두 번째 호출까지 진행합니다.
    retry_hits = [{'evidence': [{'id': 'RETRY/접수', 'kind': '접수',
                               'cells': {'조치': '배선을 재체결했습니다.'}}]}]
    retry_question = '스핀 조치는?'
    good_retry = json.dumps({'claims': [{'text': '배선을 재체결했습니다.',
        'evidence': [{'id': 'RETRY/접수', 'column': '조치'}]}]}, ensure_ascii=False)
    long_bad = json.dumps({'claims': [{'text': '999분 ' + '확인했습니다. ' * 2000,
        'evidence': [{'id': 'RETRY/접수', 'column': '조치'}]}]}, ensure_ascii=False)
    def model_response(content, reason='stop'):
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason=reason, message=SimpleNamespace(content=content))])
    for language in ('ko', 'en'):
        cfg = {'model': 'mock', 'reasoning_effort': 'high', 'answer_instruction_language': language}
        _, original_messages = llm.case_answer_messages(retry_question, retry_hits, instruction_language=language)
        original_chars = sum(len(m['content']) for m in original_messages)
        for padding, mode in ((900, 'without_response'), (0, 'regenerate')):
            create = Mock(side_effect=[model_response(long_bad), model_response(good_retry)])
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            details = {}
            llm.answer(client, cfg | {'max_context_chars': original_chars + padding}, retry_question, retry_hits, details=details)
            first, second = [call.kwargs['messages'] for call in create.call_args_list]
            assert second[:2] == first == original_messages
            assert sum(len(m['content']) for m in second) <= original_chars + padding
            assert details['validation_events'][0]['retry_context'] == mode
            assert details['attempts'] == 2 and details['instruction_language'] == language
            if padding:
                assert second[-1]['content'].endswith(llm.ANSWER_REPAIR[language])
        # 출력 한도 재시도도 요청이 꽉 찬 경계에서 작동합니다.
        create.side_effect = [model_response('잘린 응답', 'length'), model_response(good_retry)]
        llm.answer(client, cfg | {'max_context_chars': original_chars}, retry_question, retry_hits, details=details)
        assert details['retried_length'] and details['attempts'] == 2
        token_cfg = cfg | {'answer_tokenizer': 'o200k_harmony', 'answer_max_tokens': 500}
        token_cfg['context_window'] = llm.answer_input_reservation(original_messages, token_cfg)[0] + 500
        create.reset_mock()
        create.side_effect = [model_response(long_bad), model_response(good_retry)]
        llm.answer(client, token_cfg, retry_question, retry_hits, details=details)
        assert all(call.kwargs['max_tokens'] == 500 for call in create.call_args_list)
        assert create.call_args_list[1].kwargs['messages'] == original_messages
    # 후속 질문 정리는 모호한 대상을 추측하지 않고 확인 질문을 반환할 수 있습니다.
    create.side_effect = [model_response('{"clarification":"어떤 사례를 뜻하나요?"}'), model_response('{"question":""}')]
    history = [{'question': '스핀 조치는?', 'answer_claims': [{'text': '배선 재체결', 'receipts': ['RETRY']}]}]
    assert 'clarification' in llm.resolve_follow_up(client, cfg, '그것은?', history)
    assert create.call_args.kwargs['messages'][0]['content'] == llm.FOLLOW_UP_PROMPTS['en']
    try:
        llm.resolve_follow_up(client, cfg, '그것은?', history)
    except ValueError:
        pass
    else:
        raise AssertionError('빈 후속 검색 질문을 허용했습니다.')
    case_rules = json.loads((root / 'cslist_cases.json').read_text(encoding='utf-8'))
    for index, wrong in [(0, '관찰 시간을 확인할 수 없습니다.'), (3, '씰 교체를 완료했습니다.'),
                         (6, '온도는 86도이며 금액을 확인할 수 없습니다.'), (8, '밸브 점검을 완료했습니다.')]:
        assert answer_checks(wrong, case_rules[index])['passed'] is False, '알려진 답변 오류를 회귀 검사하지 않았습니다.'
    schema, rules = cslist.load_schema(root / 'cslist_schema.json')
    rows = records.read_excel(root / 'cslist_sample.xlsx', 'CSList',
                              required_columns=schema['columns'], id_column=schema['id_column'])
    with sqlite3.connect(':memory:') as db:
        storage.ingest(db, rows, rules, schema=schema)
        hit = search.retrieve(db, 'CS-DEMO-001의 마지막 조치는?')[0]
        assert hit['id'] == 'CS-DEMO-001/문서', '운영 검색이 접수 전체 문서를 반환해야 합니다.'
        assert {'CS-DEMO-001/접수', 'CS-DEMO-001/작업1', 'CS-DEMO-001/작업2'} == set(hit['evidence_ids'])
        assert '영점 조정함' in hit['content'] and '배기 튜브 교체' in hit['content']
        assert search.retrieve(db, 'CS-DEMO-999의 조치는?') == []
        assert search.retrieve(db, 'CS-DEMO-999 DEMO-ETCH-01의 조치는?') == []
        assert search.retrieve(db, 'CS-DEMO-001 DEMO-ETCH-99의 조치는?') == []
        entry = dict(source_key=rows[0]['_source_key'], id='CS-DEMO-001',
                     revision=records.row_revision(rows[0]), column='원인(2)', action='edit',
                     value='배기 튜브에서 균열이 확인되었습니다. qzxjvkwp.')
        storage.apply_reviews(db, {'format': 'cell-reviews-v1', 'entries': [entry]}, rules, schema=schema)
        comparison = search.compare_search(db, [{'question': 'qzxjvkwp', 'expected_ids': ['CS-DEMO-001/작업2'], 'review': '원문 의미는 사람이 대조'}])
        assert not comparison[0]['original_pass'] and comparison[0]['accepted_pass']
        assert comparison[0]['changed_cells'] == 1
        assert all('qzxjvkwp' not in h['content'] for h in comparison[0]['original_hits'])
        assert db.execute('SELECT COUNT(*) FROM cell_reviews').fetchone()[0] == 1
        # 검토 전 자료를 조용히 채택문 성능으로 보고하지 않습니다.
        storage.apply_reviews(db, {'format': 'cell-reviews-v1', 'entries': [entry | {'action': 'original'}]}, rules, schema=schema)
        assert search.compare_search(db, [{'question': '튜브', 'expected_ids': [], 'review': ''}])[0]['changed_cells'] == 0
        assert search.compare_search(db, [{'question': '튜브', 'expected_ids': [], 'review': ''}])[0]['original_pass'] is None
        # 건수는 검색 상위 3개가 아니라 전체 원본에서 계산합니다.
        assert search.count_records(db, '전체 접수는 총 몇 건인가요?')['count'] == 8
        assert search.count_records(db, '설비코드="DEMO-ETCH-01" 접수는 총 몇 건인가요?')['count'] == 1
        assert search.count_records(db, '설비코드="없는 설비" 접수는 총 몇 건인가요?')['count'] == 0
        assert search.count_records(db, '재발한 건은 총 몇 건인가요?')['count'] is None
        assert search.count_records(db, '2026년 9월 접수는 몇 건인가요?')['count'] is None
        assert search.count_records(db, '설비코드="A"와 설비코드="B"의 접수는 몇 건인가요?')['count'] is None
        assert search.count_records(db, '설비코드="x\' OR 1=1 --" 접수는 몇 건인가요?')['count'] == 0
        assert search.count_records(db, '몇 회 동작을 확인했나요?') is None
        # DB에 저장된 원문 셀로 검증하므로 잘못된 작업 인용과 수치를 잡습니다.
        hits = search.retrieve(db, 'CS-DEMO-001의 관찰 시간은?')
        good = {'claims': [{'text': '배기 튜브 교체 후 60분 동안 알람이 없었습니다.',
                            'evidence': [{'id': 'CS-DEMO-001/작업2', 'column': '조치내용(2)',
                                          'quote': rows[0]['조치내용(2)']}]}]}
        reply = llm.validate_claims(good, hits)
        assert '60분' in reply and '[CS-DEMO-001/작업2]' in reply
        # 실제 로봇 사례: 원인·조치를 인용했지만 함께 옮겨 쓴 증상 셀을 빠뜨렸습니다.
        robot_hits = [{'evidence': [
            {'id': 'CS-ROBOT/접수', 'cells': {'현상': '원점 복귀 실패.\n반복 기동 중 정지.'}},
            {'id': 'CS-ROBOT/작업1', 'cells': {'원인(1)': '엔코더 케이블 단선 확인',
                                              '조치내용(1)': '케이블 교체 후 원점 복귀 50회 정상.'}},
        ]}]
        robot_claim = {'text': '원점 복귀 실패·반복 기동 중 정지 현상에 대해 엔코더 케이블 단선을 확인하고 케이블 교체 후 원점 복귀 50회 정상이었습니다.',
                       'evidence': [
                           {'id': 'CS-ROBOT/작업1', 'column': '원인(1)', 'quote': '엔코더 케이블 단선 확인'},
                           {'id': 'CS-ROBOT/작업1', 'column': '조치내용(1)', 'quote': '케이블 교체 후 원점 복귀 50회 정상.'},
                       ]}
        try:
            llm.validate_claims({'claims': [robot_claim]}, robot_hits)
        except ValueError as error:
            assert '현상' in str(error)
        else:
            raise AssertionError('인용하지 않은 증상 셀의 문장을 답변에 옮겼습니다.')
        robot_claim['evidence'].append({'id': 'CS-ROBOT/접수', 'column': '현상', 'quote': '원점 복귀 실패.\n반복 기동 중 정지.'})
        assert '50회' in llm.validate_claims({'claims': [robot_claim]}, robot_hits)
        plan_quote = '다음 점검 때 상태 확인 예정. 주기는 미정.'
        plan_hits = [{'evidence': [{'id': 'CS-PLAN/접수', 'cells': {'재발방지대책': plan_quote}}]}]
        planned = {'evidence': [{'id': 'CS-PLAN/접수', 'column': '재발방지대책', 'quote': plan_quote}],
                   'text': '다음 점검 때 상태 확인 예정이며 주기는 미정입니다.'}
        assert '예정' in llm.validate_claims({'claims': [planned]}, plan_hits)
        for wording in ('현재 시점에서 상태 확인이 아직 남아 있음을 보여줍니다.', '점검은 아직 완료되지 않았습니다.'):
            try:
                llm.validate_claims({'claims': [planned | {'text': wording}]}, plan_hits)
            except ValueError as error:
                assert '예정' in str(error)
            else:
                raise AssertionError('다음 점검 예정만으로 현재 미완료 상태를 단정했습니다.')
        explicit = '조정기 교체 예정. 생산 중이라 미실시.'
        plan_hits[0]['evidence'][0]['cells']['재발방지대책'] = explicit
        stated = {'text': '조정기 교체는 아직 미실시입니다.', 'evidence': [
            {'id': 'CS-PLAN/접수', 'column': '재발방지대책', 'quote': explicit}]}
        assert '미실시' in llm.validate_claims({'claims': [stated]}, plan_hits), '원문에 명시된 미실시를 거절했습니다.'
        # ISO 날짜의 하이픈은 음수 부호가 아닙니다. 날짜 전체가 같을 때만 표기 변경을 허용합니다.
        dated = {'claims': [{'text': '2026년 9월 1일 접수되었습니다.', 'evidence': [
            {'id': 'CS-DEMO-001/접수', 'column': '접수일시', 'quote': rows[0]['접수일시']}]}]}
        assert '2026년 9월 1일' in llm.validate_claims(dated, hits)
        # 날짜의 구분자·0 채움은 바뀌어도 년월일 전체가 같으면 허용합니다.
        for formatted in ('2026.09.01', '2026/9/1', '2026. 9. 1.', '2026-9-1'):
            payload = {'claims': [dated['claims'][0] | {'text': f'{formatted} 접수되었습니다.'}]}
            assert formatted in llm.validate_claims(payload, hits)
            assert llm.claim_dates(formatted)[0] == llm.claim_dates('2026년 9월 1일')[0]
        flow_hits = search.retrieve(db, 'CS-DEMO-002의 유량은?')
        flow = {'claims': [{'text': '세척 후 유량은 11.8 L/min입니다.', 'evidence': [
            {'id': 'CS-DEMO-002/작업1', 'column': '조치내용(1)', 'quote': rows[1]['조치내용(1)']}]}]}
        assert '11.8 L/min' in llm.validate_claims(flow, flow_hits)
        for wrong, source in [
            ({'claims': [dated['claims'][0] | {'text': '2026-01-09 접수되었습니다.'}]}, hits),
            ({'claims': [dated['claims'][0] | {'text': '2026년 9월 31일 접수되었습니다.'}]}, hits),
            ({'claims': [dated['claims'][0] | {'text': '2026.01.09 접수되었습니다.'}]}, hits),
            ({'claims': [dated['claims'][0] | {'text': '2026/9/31 접수되었습니다.'}]}, hits),
            ({'claims': [good['claims'][0] | {'text': '-60분 관찰했습니다.'}]}, hits),
            ({'claims': [flow['claims'][0] | {'text': '유량이 11.8 L/min으로 회복되었습니다.'}]}, flow_hits),
            ({'claims': [flow['claims'][0] | {'text': '유량이 11.8 L/min으로 정상화되었습니다.'}]}, flow_hits),
        ]:
            try:
                llm.validate_claims(wrong, source)
            except ValueError:
                pass
            else:
                raise AssertionError('날짜·부호 변경 또는 원문에 없는 회복·정상화 해석을 허용했습니다.')
        for wrong in (
            {'claims': [good['claims'][0] | {'text': '90분 동안 알람이 없었습니다.'}]},
            {'claims': [good['claims'][0] | {'text': '60시간 동안 알람이 없었습니다.'}]},
            {'claims': [{'text': '0분 관찰했습니다.', 'evidence': [{'id': 'CS-DEMO-001/작업2',
                'column': '조치내용(2)', 'quote': '0분'}]}]},
            {'claims': [good['claims'][0] | {'evidence': [{'id': 'CS-DEMO-001/작업1', 'column': '조치내용(2)', 'quote': rows[0]['조치내용(2)']}]}]},
            {'claims': [good['claims'][0] | {'evidence': [{'id': 'CS-DEMO-001/작업2', 'column': '조치내용(2)', 'quote': '영구 해결 완료'}]}]},
            {'claims': [good['claims'][0] | {'evidence': []}]},
        ):
            try:
                llm.validate_claims(wrong, hits)
            except ValueError:
                pass
            else:
                raise AssertionError('수치 변경·잘못된 작업·가짜 구절·근거 누락을 허용했습니다.')
        empty_hits = search.retrieve(db, 'CS-DEMO-006의 MTTR은?')
        blank = {'claims': [{'text': 'MTTR은 미기재되어 확인할 수 없습니다.', 'evidence': [
            {'id': 'CS-DEMO-006/접수', 'column': 'MTTR', 'quote': ''}]}]}
        assert '미기재' in llm.validate_claims(blank, empty_hits)
        named_column = {'claims': [{'text': '시작시간(1)은 미기재되어 확인할 수 없습니다.', 'evidence': [
            {'id': 'CS-DEMO-006/작업1', 'column': '시작시간(1)', 'quote': ''}]}]}
        assert '미기재' in llm.validate_claims(named_column, empty_hits), '열 이름의 작업 번호를 관찰 수치로 오인했습니다.'
        try:
            llm.validate_claims({'claims': [blank['claims'][0] | {'text': 'MTTR은 0입니다.'}]}, empty_hits)
        except ValueError:
            pass
        else:
            raise AssertionError('빈칸을 0으로 바꾼 답변을 허용했습니다.')
        # 요청 크기 초과는 근거 일부를 말없이 자르지 않고 API 호출 전에 알립니다.
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock())))
        config = {'model': 'gpt-4', 'reasoning_effort': 'high', 'max_context_chars': 100}
        try:
            llm.answer(client, config, '관찰 시간은?', hits)
        except ValueError as error:
            assert '분량' in str(error)
        else:
            raise AssertionError('근거 분량 제한을 무시했습니다.')
        assert client.chat.completions.create.call_count == 0
        assert '확인할 수 없' in llm.answer(client, config, '없는 접수', [])[0]
        # 조치 셀을 인용만 하고 본문에서는 생략한 실제 실패를 축소 재현합니다.
        symptom = rows[0]['현상']
        omitted = {'claims': [{'text': symptom, 'evidence': [
            {'id': 'CS-DEMO-001/접수', 'column': '현상'},
            {'id': 'CS-DEMO-001/작업2', 'column': '조치내용(2)'}]}]}
        complete_create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(
            finish_reason='stop', message=SimpleNamespace(content=json.dumps(omitted, ensure_ascii=False)))]))
        complete_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete_create)))
        complete_details = {}
        complete_reply, _ = llm.answer(complete_client, config | {'max_context_chars': 24000},
                                      '진공 알람의 과거 조치와 결과는?', hits, details=complete_details)
        assert rows[0]['조치내용(1)'] in complete_reply and rows[0]['조치내용(2)'] in complete_reply
        assert complete_reply.index('작업1 조치·결과 원문') < complete_reply.index('작업2 조치·결과 원문')
        assert complete_details['source_supplements'] and complete_client.chat.completions.create.call_count == 1
        assert len(complete_details['claims']) == 1  # 모델 답변과 프로그램이 붙인 원문을 별도로 보관합니다.
        assert '원문 보완' not in complete_details['answer_text']
        from ragkit.web import pack_answer
        packed = pack_answer('진공 알람의 과거 조치와 결과는?', complete_reply, hits, None, complete_details, 1)
        assert packed['answer'] == complete_details['answer_text'] and '원문 보완' not in packed['answer']
        assert len(packed['claims']) == 1
        cited = {(e['id'], e['column']) for c in packed['claims'] for e in c['evidence']}
        additional = [e for c in packed['source_supplements'] for e in c['evidence']]
        assert any(e['column'] == '조치내용(1)' for e in additional)
        assert all((e['id'], e['column']) not in cited and e['quote'] == rows[0][e['column']] for e in additional)
        assert len(additional) == len({(e['id'], e['column']) for e in additional})
        failed = pack_answer('질문', '검사 실패', hits, None, complete_details | {'claims': [], 'source_supplements': []}, 1, code='error')
        assert failed['answer'] == '검사 실패', '이전 정상 답변이 오류를 가렸습니다.'
        supplemented = complete_details['source_supplements'][-1]
        assert {e['column'] for e in supplemented['evidence']} >= {'조치내용(1)', '조치내용(2)'}
        for ref in supplemented['evidence']:
            assert ref['quote'] == rows[0][ref['column']]
        # 직접 물은 작업·마지막 조치 범위를 유지하고 다른 접수의 조치는 섞지 않습니다.
        work1 = [{'text': rows[0]['조치내용(1)'], 'evidence': [
            {'id': 'CS-DEMO-001/작업1', 'column': '조치내용(1)'}]}]
        for question in ('조치내용(1)은?', '작업1의 조치는?'):
            result = llm.case_source_supplements(work1, hits, question)
            assert {e['column'] for e in result[0]['evidence']} == {'조치내용(1)'}
        result = llm.case_source_supplements(omitted['claims'], hits, '마지막 조치는?')
        assert {e['column'] for e in result[0]['evidence']} == {'조치내용(2)'}
        for question in ('초기화 조치 사례는?', '과거 조치와 후속 결과는?', '작업 12시간 후 유량 저하 사례는?'):
            result = llm.case_source_supplements(omitted['claims'], hits, question)
            assert {e['column'] for e in result[0]['evidence']} >= {'조치내용(1)', '조치내용(2)'}
        for question in ('조치시간만 알려줘', '과거 수리비는?', '미해결 여부는?', '온도 표시 값은?'):
            assert llm.case_source_supplements(omitted['claims'], hits, question) == []
        unrelated = search.retrieve(db, 'CS-DEMO-002의 조치는?')
        result = llm.case_source_supplements(omitted['claims'], hits + unrelated, '과거 사례는?')
        assert len(result) == 1 and all(e['id'].startswith('CS-DEMO-001/') for e in result[0]['evidence'])
        mixed_hits = json.loads(json.dumps(hits).replace('CS-DEMO-001', 'CS-DEMO-002'))
        mixed = work1 + [{'text': '조치', 'evidence': [{'id': 'CS-DEMO-002/작업2', 'column': '조치내용(2)'}]}]
        result = llm.case_source_supplements(mixed, hits + mixed_hits, 'CS-DEMO-001 작업1과 CS-DEMO-002 작업2의 조치는?')
        assert [{e['column'] for e in item['evidence']} for item in result] == [{'조치내용(1)'}, {'조치내용(2)'}]
        assert llm.case_source_supplements(omitted['claims'], hits, '현상만 알려주세요.') == []
        assert llm.case_source_supplements(omitted['claims'], hits, '과거 사례의 원인만 알려주세요.') == []
        assert llm.case_source_supplements(blank['claims'], empty_hits, 'MTTR은?') == []
        action_only = [{'text': rows[0]['조치내용(2)'], 'evidence': [
            {'id': 'CS-DEMO-001/작업2', 'column': '조치내용(2)'}]}]
        result = llm.case_source_supplements(action_only, hits, '목표 진공에 도달하는 시간이 계속 늦어집니다.')
        assert {e['column'] for e in result[0]['evidence']} >= {'조치내용(1)', '조치내용(2)'}
        # 빈 작업에 조치·결과를 만들어 넣거나, 빈 값을 0·미실시로 해석하지 않습니다.
        missing_hits = [{'evidence': [{'id': 'CS-EMPTY/접수', 'cells': {'현상': '유량 저하'}}]}]
        missing_claims = [{'text': '유량 저하', 'evidence': [{'id': 'CS-EMPTY/접수', 'column': '현상'}]}]
        result = llm.case_source_supplements(missing_claims, missing_hits, '과거 조치는?')
        assert '조치내용 기록이 없습니다' in result[0]['text'] and '미실시를 뜻하지' in result[0]['text']
        missing_hits[0]['evidence'].append({'id': 'CS-EMPTY/작업1', 'cells': {'조치내용(1)': ''}})
        result = llm.case_source_supplements(missing_claims, missing_hits, '과거 조치는?')
        assert '작업1 조치·결과 원문: 미기재' in result[0]['text'] and result[0]['evidence'][-1]['quote'] == ''
        missing_hits[0]['evidence'][1]['cells']['조치내용(1)'] = '교체 후 30분 재발 없음. 장기 관찰 미실시.'
        missing_hits[0]['evidence'].append({'id': 'CS-EMPTY/작업2', 'cells': {'조치내용(2)': ''}})
        assert llm.case_source_supplements(missing_claims, missing_hits, '마지막 조치는?') == []
        missing_claims[0]['evidence'] = [{'id': 'CS-EMPTY/작업1', 'column': '조치내용(1)'}]
        result = llm.case_source_supplements(missing_claims, missing_hits, '마지막 조치는?')
        assert result[0]['evidence'][0]['column'] == '조치내용(1)'
        # 근거 부족은 명시된 응답만 허용합니다. 빈 응답·검사 실패를 정상 거절로 숨기지 않습니다.
        refusal = {'status': 'insufficient_evidence', 'claims': []}
        def response(payload):
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(
                content=json.dumps(payload, ensure_ascii=False)))])
        guard_create = Mock(return_value=response(refusal))
        guard_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=guard_create)))
        guard_details = {}
        reply, _ = llm.answer(guard_client, config | {'max_context_chars': 24000},
                             '자료와 무관한 장비의 고장', hits, details=guard_details)
        assert guard_details['answer_status'] == 'insufficient_evidence' and guard_details['claims'] == []
        assert '검색된 후보' in reply and '부족' in reply and guard_create.call_count == 1
        for invalid in ({'claims': []}, refusal | {'claims': good['claims']}, refusal | {'reason': '임의 설명'}):
            guard_create.reset_mock()
            guard_create.return_value = response(invalid)
            try:
                llm.answer(guard_client, config | {'max_context_chars': 24000}, '관찰 시간은?', hits)
            except ValueError:
                assert guard_create.call_count == 2
            else:
                raise AssertionError('불완전하거나 모순된 응답을 근거 부족으로 처리했습니다.')
        guard_create.side_effect = [response({'claims': []}), response(refusal)]
        try:
            llm.answer(guard_client, config | {'max_context_chars': 24000}, '관찰 시간은?', hits)
        except ValueError as error:
            assert '검사' in str(error)
        else:
            raise AssertionError('검사 보정 실패를 근거 부족으로 바꿨습니다.')
        guard_create.side_effect = [SimpleNamespace(choices=[SimpleNamespace(
            finish_reason='length', message=SimpleNamespace(content='잘린 응답'))]), response(refusal)]
        reply, _ = llm.answer(guard_client, config | {'max_context_chars': 24000},
                             '자료와 무관한 장비의 고장', hits, details=guard_details)
        assert guard_details['answer_status'] == 'insufficient_evidence'
        assert guard_details['retried_length'] and guard_details['attempts'] == 2
        # 실제 외부 경계만 대역으로 바꿉니다. 실패 응답이 한 번 보정되고 결과에는 근거가 남아야 합니다.
        responses = [json.dumps({'claims': [good['claims'][0] | {'text': '90분 관찰했습니다.'}]}, ensure_ascii=False),
                     json.dumps(good, ensure_ascii=False)]
        client.chat.completions.create.side_effect = [SimpleNamespace(choices=[SimpleNamespace(
            finish_reason='stop', message=SimpleNamespace(content=text))]) for text in responses]
        details = {}
        reply, _ = llm.answer(client, config | {'max_context_chars': 24000}, '관찰 시간은?', hits, details=details)
        assert '60분' in reply and details['attempts'] == 2 and len(details['claims']) == 1
        assert client.chat.completions.create.call_count == 2
        # 셀 주소만 반환해도 원문 전체가 UI·보고서에 전달되어야 합니다.
        # 앞뒤 공백·개행은 모델이 재현할 필요가 없으며 임의의 주소는 여전히 거절합니다.
        source = {'id': 'CS-SPACES/작업1', 'cells': {'조치내용(1)': '  재부팅 후 45분 정상 / 영구 해결 여부 모름.\n'}}
        addressed = {'claims': [{'text': '45분 정상이며 영구 해결 여부는 모릅니다.',
                               'evidence': [{'id': source['id'], 'column': '조치내용(1)'}]}]}
        assert '45분' in llm.validate_claims(addressed, [{'evidence': [source]}])
        assert addressed['claims'][0]['evidence'][0]['quote'] == source['cells']['조치내용(1)']
        # 구조용 공백 때문에 원문이 분량 제한에 걸리지 않아야 합니다. 셀 내부 공백은 보존합니다.
        compact_source = source | {'kind': '작업1', 'cells': source['cells'] | {
            'MTTR': '', '수리비': '0', '메모': '  "쉼표, 콜론: 공백"\\\t\n'}}
        compact_question = '조치내용(1), MTTR, 수리비와 메모는?'
        expected_context = json.dumps([compact_source], ensure_ascii=False, separators=(',', ':'))
        context_limit = len(llm.CLAIM_PROMPT) + len(f'질문: {compact_question}\n원문 자료:\n{expected_context}')
        compact_create = Mock(return_value=response(addressed))
        compact_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=compact_create)))
        compact_details = {}
        _, compact_context = llm.answer(compact_client, config | {'max_context_chars': context_limit},
            compact_question, [{'evidence': [compact_source]}], details=compact_details)
        assert json.loads(compact_context) == [compact_source], '빈칸·0·공백·개행·기호가 포함된 원문이 변경됐습니다.'
        assert len(compact_context) < len(json.dumps([compact_source], ensure_ascii=False))
        assert compact_create.call_args.kwargs['messages'][1]['content'].endswith(compact_context)
        assert compact_details['claims'][0]['evidence'][0]['quote'] == source['cells']['조치내용(1)']
        compact_create.reset_mock()
        try:
            llm.answer(compact_client, config | {'max_context_chars': context_limit - 1},
                       compact_question, [{'evidence': [compact_source]}])
        except ValueError as error:
            assert '분량' in str(error)
        else:
            raise AssertionError('압축 후에도 제한을 넘는 원문을 자르거나 그대로 전송했습니다.')
        assert compact_create.call_count == 0
        for reference in ({'id': 'unknown', 'column': '조치내용(1)'},
                          {'id': source['id'], 'column': '조치내용(2)'},
                          {'id': source['id'], 'column': '조치내용(1)', 'quote': source['cells']['조치내용(1)'].strip()}):
            try:
                llm.validate_claims({'claims': [addressed['claims'][0] | {'evidence': [reference]}]}, [{'evidence': [source]}])
            except ValueError:
                pass
            else:
                raise AssertionError('잘못된 셀 주소나 공백이 사라진 직접 인용을 허용했습니다.')
        # 보정 요청에는 실패한 응답 자체가 있어야 모델이 잘못된 내용을 고칠 수 있습니다.
        correction = client.chat.completions.create.call_args.kwargs['messages']
        assert correction[-2] == {'role': 'assistant', 'content': responses[0]}, '보정 대상 응답이 전달되지 않았습니다.'
        # 실제 실패한 claims 목록의 닫는 괄호 오류를 축소 재현합니다.
        malformed = '{"claims":[{"text":"첫 답변","evidence":[]}]}},{"text":"두 번째 답변","evidence":[]}]} '
        client.chat.completions.create.reset_mock()
        client.chat.completions.create.side_effect = [SimpleNamespace(choices=[SimpleNamespace(
            finish_reason='stop', message=SimpleNamespace(content=text))]) for text in (malformed, responses[1])]
        reply, _ = llm.answer(client, config | {'max_context_chars': 24000},
                              '관찰 시간은?', hits)
        assert '60분' in reply and client.chat.completions.create.call_count == 2
        request = client.chat.completions.create.call_args.kwargs
        assert request['messages'][-2]['content'] == malformed
        # 두 번 모두 잘못된 JSON이면 임의로 복구해 성공 처리하지 않습니다.
        client.chat.completions.create.reset_mock()
        client.chat.completions.create.side_effect = [SimpleNamespace(choices=[SimpleNamespace(
            finish_reason='stop', message=SimpleNamespace(content=malformed))])] * 2
        try:
            llm.answer(client, config | {'max_context_chars': 24000}, '관찰 시간은?', hits)
        except ValueError:
            pass
        else:
            raise AssertionError('깨진 JSON을 정상 답변으로 내보냈습니다.')
        assert client.chat.completions.create.call_count == 2
        # Qwen이 JSON 전체를 Markdown 코드 블록에 넣은 실제 응답 형식입니다.
        # 셀 값은 건드리지 않고 바깥 울타리만 제거해야 합니다.
        fenced = '```json\n' + json.dumps(good, ensure_ascii=False) + '\n```'
        client.chat.completions.create.reset_mock()
        client.chat.completions.create.side_effect = [SimpleNamespace(choices=[SimpleNamespace(
            finish_reason='stop', message=SimpleNamespace(content=fenced))])] * 2
        reply, _ = llm.answer(client, config | {'max_context_chars': 24000}, '관찰 시간은?', hits)
        assert '60분' in reply and client.chat.completions.create.call_count == 1
        # Mac에 설정한 생성 한도를 실제 요청에 전달하고, 잘린 JSON은 답변으로 내보내지 않습니다.
        client.chat.completions.create.side_effect = None
        client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason='length', message=SimpleNamespace(content='{"claims":['))])
        try:
            llm.answer(client, config | {'max_context_chars': 24000, 'answer_max_tokens': 1500,
                                        'answer_temperature': 0}, '관찰 시간은?', hits)
        except RuntimeError:
            pass
        else:
            raise AssertionError('출력 한도에서 잘린 응답을 정상 답변으로 사용했습니다.')
        request = client.chat.completions.create.call_args.kwargs
        assert request.get('max_tokens') == 1500 and request.get('temperature') == 0
        assert client.chat.completions.create.call_count == 3  # 직전 정상 응답 1회 + 잘림 2회
        # 잘린 부분은 재사용하지 않고 같은 설정·원문으로 한 번 재요청합니다.
        without_quotes = json.loads(json.dumps(good))
        for evidence in without_quotes['claims'][0]['evidence']:
            del evidence['quote']
        client.chat.completions.create.reset_mock()
        client.chat.completions.create.side_effect = [
            SimpleNamespace(choices=[SimpleNamespace(finish_reason='length', message=SimpleNamespace(content='잘린 응답'))]),
            SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=json.dumps(without_quotes)))])]
        details = {}
        reply, _ = llm.answer(client, config | {'max_context_chars': 24000}, '관찰 시간은?', hits, details=details)
        assert '60분' in reply and details['retried_length'] and details['attempts'] == 2
        assert details['claims'][0]['evidence'][0]['quote'] == good['claims'][0]['evidence'][0]['quote']
        first, retry = [call.kwargs for call in client.chat.completions.create.call_args_list]
        assert first['extra_body'] == retry['extra_body'] == {'reasoning_effort': 'high'}
        assert all(key not in request for request in (first, retry) for key in ('max_tokens', 'temperature', 'top_p'))
        assert retry['messages'][:2] == first['messages'] and all(m['role'] != 'assistant' for m in retry['messages'])
        client.chat.completions.create.reset_mock()
        for invalid in ({'answer_max_tokens': 0}, {'answer_max_tokens': True}, {'answer_temperature': float('nan')}):
            try:
                llm.answer(client, config | {'max_context_chars': 24000} | invalid, '관찰 시간은?', hits)
            except ValueError:
                pass
            else:
                raise AssertionError('잘못된 생성 한도 또는 샘플링 값을 서버로 보냈습니다.')
        assert client.chat.completions.create.call_count == 0
        # 이 검사는 정해진 오류의 재발만 찾습니다. 자연어의 모든 의미를 판정하지 않습니다.
        regression_cases = [
            ({'required': ['90\\s*분']}, '관찰 시간은 확인할 수 없습니다.', '90분 관찰했습니다.'),
            ({'required': ['미실시|아직|않|안\\s*함'], 'forbidden': ['교체를 완료했습니다']}, '씰 교체를 완료했습니다.', '발주만 완료했고 교체는 아직 미실시입니다.'),
            ({'required': ['예정|미실시|아직']}, '밸브 점검을 완료했습니다.', '밸브 점검은 예정입니다.'),
            ({'required': ['MTTR.{0,15}(미기재|없|확인)'], 'forbidden': ['MTTR.{0,5}0']}, 'MTTR은 0입니다.', 'MTTR은 미기재입니다.'),
        ]
        for criteria, wrong, correct in regression_cases:
            assert not answer_checks(wrong, criteria)['passed']
            assert answer_checks(correct, criteria)['passed']
    print('PASS: 접수 문서·원문/채택문 비교·전체/조건 건수·셀 근거·수치·빈칸·입력 분량')


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1] / 'windows-demo')
