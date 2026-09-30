"""영어 검색·원문 보존·재실행·검토 후 무효화를 실제 SQLite로 검사합니다."""
from contextlib import closing
import copy
import json
import re
import sqlite3
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from openai import APITimeoutError

from ragkit import cslist, english, records, storage, search, llm


def translation_repairs(root):
    config = {'model': 'test', 'reasoning_effort': 'high'}
    original = '1.SCW132 11.8 STEP Thinner Attack 불량 점검'
    cells = {'현상': original, '조치내용(1)': '필터 교체'}
    evidence = json.dumps([{'cells': cells, 'accepted': {'현상': original + ' 필요'}}])
    assert english.translation_cells(evidence, {'rewrite_columns': list(cells)})['현상'].endswith('필요')
    english.validate_translation({'현상': 'A'}, {'현상': 'A'})
    for result, phrase in [('', '빈 문자열'), ('미확인', '영어 단어'), ('A' * 1001, '응답 1001자')]:
        try:
            english.validate_translation({'현상': result}, {'현상': '미확인'})
        except ValueError as error:
            assert phrase in str(error), error
        else:
            raise AssertionError(result)

    def repair(_client, _config, prompt, payload):
        if prompt.startswith(english.TRANSLATE_PROMPT) and 'Correction attempt' not in prompt:
            return {'현상': original.replace('11.8', '11.9'), '조치내용(1)': 'Filter replaced'}
        assert set(payload) == {'현상'} and 'Previous validation error' in prompt
        assert '__CS_NUMBER_' in payload['현상'] and 'SCW132' not in payload['현상']
        return {'현상': payload['현상'].replace('불량 점검', 'defect inspection')}
    with patch.object(english, 'request_json', side_effect=repair) as request:
        translated = english.translate_cells(object(), config, cells)
        assert request.call_count == 2
        assert translated['조치내용(1)'] == 'Filter replaced'
        assert 'SCW132 11.8 STEP' in translated['현상']
        assert llm.measured(translated['현상']) == llm.measured(original)

    # 최초 빈 값 → 영어 없음 → 과도한 길이를 거쳐 세 번째 보완에서 성공합니다.
    # 각 보완에는 직전 오류만 전달하고 처음 통과한 조치 셀은 다시 보내지 않습니다.
    with patch.object(english, 'request_json', side_effect=[
            {'현상': '', '조치내용(1)': 'Filter replaced'}, {'현상': '점검 필요'},
            {'현상': 'X' * 1001}, {'현상': 'Inspection needed'}]) as request:
        value = english.translate_cells(object(), config, {'현상': '점검 필요', '조치내용(1)': '필터 교체'})
        assert value == {'현상': 'Inspection needed', '조치내용(1)': 'Filter replaced'}
        assert request.call_count == 4
        for attempt, call in enumerate(request.call_args_list[1:], 1):
            assert f'Correction attempt {attempt} of 3' in call.args[2]
            assert set(call.args[3]) == {'현상'}
        assert '빈 문자열' in request.call_args_list[1].args[2]
        assert '영어 단어가 없습니다' in request.call_args_list[2].args[2]
        assert '응답 1001자' in request.call_args_list[3].args[2]
    prompt = english.correction_prompt('x'*995+'LOCAL-SECRET', config | {'api_key':'LOCAL-SECRET'}, 3)
    assert 'LOCAL' not in prompt and '***' in prompt

    # 빈 응답·누락 열은 해당 셀만 보정합니다. 정상인 다른 셀은 다시 요청하지 않습니다.
    for initial in ({'현상': '', '조치내용(1)': 'Filter replaced'}, {'조치내용(1)': 'Filter replaced'}):
        with patch.object(english, 'request_json', side_effect=[initial, {'현상': 'Pump leak'}]) as request:
            value = english.translate_cells(object(), config, {'현상': '펌프 누수', '조치내용(1)': '필터 교체'})
            assert value == {'현상': 'Pump leak', '조치내용(1)': 'Filter replaced'}
            assert set(request.call_args.args[3]) == {'현상'}
    # 보정에서도 숫자를 바꾸거나 보호 표시자를 지우면 성공으로 저장하지 않습니다.
    with patch.object(english, 'request_json', return_value={'현상': 'Pressure 999 kPa'}) as request:
        try:
            english.translate_cells(object(), config, {'현상': '압력 -0.2 MPa'})
        except ValueError as error:
            assert '자동 보완 3회 후 실패' in str(error) and request.call_count == 4
        else:
            raise AssertionError('숫자 훼손을 성공으로 처리했습니다.')

    source = '센서 점검 10분 후 압력 -0.2 MPa 확인. ' * 40
    parts = list(english.split_cell('현상', source, 150))
    assert ''.join(parts) == source and len(parts) > 1
    assert all(len(json.dumps({'현상': p}, ensure_ascii=False)) <= 150 for p in parts)
    assert [item for p in parts for item in llm.measured(p)] == llm.measured(source)
    with patch.object(english, 'request_json', side_effect=lambda c, cfg, prompt, payload:
                      {key: value + ' inspection' for key, value in payload.items()}) as request:
        translated = english.translate_cells(object(), config | {'max_context_chars': 150}, {'현상': source})
        assert request.call_count == len(parts) and llm.measured(translated['현상']) == llm.measured(source)
    # 짧은 숫자들이 표시자로 길어져도 실제 전송 셀의 문자 한도를 지킵니다.
    def bounded(_client, _config, _prompt, payload):
        assert len(json.dumps(payload, ensure_ascii=False)) <= 80
        return {key: value + ' inspection' for key, value in payload.items()}
    with patch.object(english, 'request_json', side_effect=bounded) as request:
        dense = english.translate_cells(object(), config | {'max_context_chars': 80}, {'현상': '1 2 3 4 5 점검'})
        assert re.findall(r'\d+', dense['현상']) == ['1', '2', '3', '4', '5'] and request.call_count > 1
    try:
        list(english.split_cell('현상', 'X' * 200, 150))
    except ValueError as error:
        assert '안전하게 나눌 공백' in str(error)
    else:
        raise AssertionError('긴 식별자를 잘랐습니다.')

    schema, rules = cslist.load_schema(root / 'cslist_schema.json')
    rows = [{key: '' for key in schema['columns']} | {schema['id_column']: f'TIMEOUT-{index}',
            '_id_column': schema['id_column'], '현상': value}
            for index, value in enumerate(('펌프 누수', '센서 압력', '팬 진동'))]
    calls, stages = {}, []
    def timeout(_client, _config, prompt, payload):
        source = payload['현상']
        calls[source] = calls.get(source, 0) + 1
        if source == '센서 압력' or (source == '펌프 누수' and calls[source] == 1):
            raise APITimeoutError(request=Mock())
        return {'현상': source + ' maintenance'}
    with closing(sqlite3.connect(':memory:')) as db:
        storage.ingest(db, rows, rules, schema=schema)
        with patch.object(english, 'request_json', side_effect=timeout), patch.object(english.time, 'sleep'):
            result = english.build_english(db, object(), config | {'normalization_workers': 2}, schema,
                                           limit=3, progress=stages.append)
        assert result['generated'] == 2 and len(result['failures']) == 1, result
        assert result['processed'] == 3 and result['timeout_retries'] == 2 and result['active_workers'] == 1
        assert calls == {'펌프 누수': 2, '센서 압력': 2, '팬 진동': 1}, calls
        assert '자동 재시도 1회도 시간 초과' in result['failures'][0]['error']
        assert any('동시 처리를 1건' in stage for stage in stages)
        assert english.english_count(db) == 2
    print('PASS: 영어 셀별 보정·숫자 보호·채택문 우선·길이 분할·시간 초과 감속/재시도 상한')


def parallel_translation(root):
    schema, rules = cslist.load_schema(root / 'cslist_schema.json')
    rows = [{key: '' for key in schema['columns']} | {schema['id_column']: f'EN-PAR-{index}',
            '_id_column': schema['id_column'], '현상': value}
            for index, value in enumerate(('센서 압력', '펌프 누수', '센서 압력', '팬 진동'))]
    barrier = threading.Barrier(2)
    release = threading.Event()
    owner = threading.get_ident()
    calls = []
    def translate(_client, _config, _prompt, cells):
        assert threading.get_ident() != owner
        calls.append(cells['현상'])
        if cells['현상'] in ('센서 압력', '펌프 누수'):
            barrier.wait(timeout=5)
        if cells['현상'] == '센서 압력':
            assert release.wait(5), '먼저 완료한 번역이 저장되지 않았습니다.'
        if cells['현상'] == '팬 진동':
            raise ValueError('모의 번역 실패')
        return {'현상': cells['현상'] + ' maintenance'}
    config = {'model': 'test', 'normalization_workers': 2}
    with closing(sqlite3.connect(':memory:')) as db:
        storage.ingest(db, rows, rules, schema=schema)
        original = db.execute('SELECT * FROM normalized_records').fetchall()
        def progress(message):
            assert threading.get_ident() == owner
            if '완료' in message and 'EN-PAR-1' in message:
                assert db.execute("SELECT 1 FROM english_translations WHERE record_id='EN-PAR-1'").fetchone()
                release.set()
        with patch.object(english, 'request_json', side_effect=translate):
            result = english.build_english(db, object(), config, schema, limit=0, progress=progress)
        assert len(calls) == 6 and calls.count('팬 진동') == 4
        assert result['generated'] == 2 and result['reused'] == 1
        assert result['workers'] == 2 and [f['id'] for f in result['failures']] == ['EN-PAR-3']
        with patch.object(english, 'request_json', return_value={'현상': '팬 진동 maintenance'}) as request:
            result = english.build_english(db, object(), config, schema, limit=0, failed_only=True)
        assert request.call_count == 1 and result['generated'] == 1 and result['indexed'] == 4
        assert 'An earlier run failed' in request.call_args.args[2]
        assert '자동 보완 3회 후 실패' in request.call_args.args[2]
        assert db.execute('SELECT COUNT(*) FROM english_failures').fetchone()[0] == 0
        assert db.execute('SELECT * FROM normalized_records').fetchall() == original
    print('PASS: 영어 병렬 실행·완료순 저장·동일 내용 재사용·실패 재시도·SQLite 소유 스레드')


def bilingual_questions(root):
    """번역이 답변·분량 계산에 빠지거나 실패 번역이 사용되는 회귀를 잡습니다."""
    schema, rules = cslist.load_schema(root / 'cslist_schema.json')
    source = '센서 교체 후에도 동일 증상 발생. 배선 접촉 불량 확인하여 수정 후 정상화.'
    row = dict.fromkeys(schema['columns'], '')
    row.update({schema['id_column']: 'BQ-DEMO-001', '현상': '센서 신호 불량', '설비코드': 'SCW132',
                '조치내용(1)': source, '_id_column': schema['id_column'],
                '_source_key': 'fictional-bilingual-question'})
    question = '센서 교체 후에도 같은 증상이 반복됩니다. 과거 조치와 결과는?'
    translated = 'The same symptom persists after sensor replacement. What were the past actions and their outcomes?'
    config = {'model': 'offline-test', 'reasoning_effort': 'high', 'max_context_chars': 40000}
    def response(data, finish='stop'):
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish,
            message=SimpleNamespace(content=json.dumps(data, ensure_ascii=False)))])
    create = Mock(return_value=response({'query': translated}))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with closing(sqlite3.connect(':memory:')) as db:
        storage.ingest(db, [row], rules, schema=schema)
        original = db.execute('SELECT original_json FROM normalized_records').fetchone()[0]
        detail = {}
        hits = english.bilingual_retrieve(db, question, client, config, details=detail)
        assert hits and detail['search']['english_query'] == translated, '영어 문서가 없어도 질문 번역은 답변에 사용해야 합니다.'
        assert detail['search']['translation_status'] == 'translated'
        assert detail['search']['mode'] == 'korean' and detail['search']['warning']
        ref = {'id': 'BQ-DEMO-001/작업1', 'column': '조치내용(1)'}
        create.side_effect = [response({'claims': [{'text': '센서 교체 후 정상화되었습니다.', 'evidence': [ref]}]}),
                              response({'claims': [{'text': source, 'evidence': [ref]}]})]
        reply, context = llm.answer(client, config, question, hits, details=detail)
        assert '동일 증상' in reply and create.call_count == 3
        for call in create.call_args_list[1:]:
            sent = call.kwargs['messages'][1]['content']
            assert question in sent and translated in sent, '첫 답변과 보정 모두 같은 두 언어 질문을 사용해야 합니다.'
        assert translated not in context and source in context, '질문 번역을 원문 근거로 취급했습니다.'
        assert db.execute('SELECT original_json FROM normalized_records').fetchone()[0] == original
        assert english.english_count(db) == 0, '질문 번역 때문에 문서 번역이나 DB 재생성이 발생했습니다.'
        # 자동 참고 사례 선택도 실제 전달할 영어 질문의 길이를 포함합니다.
        _, plain = llm.case_answer_messages(question, hits)
        _, dual = llm.case_answer_messages(question, hits, english_question=translated)
        boundary = sum(len(m['content']) for m in plain)
        assert sum(len(m['content']) for m in dual) > boundary
        try:
            llm.select_answer_cases(config | {'max_context_chars': boundary}, question, hits, details=detail)
        except ValueError as error:
            assert '한 건도' in str(error)
        else:
            raise AssertionError('영어 질문을 뺀 크기로 참고 사례를 선택했습니다.')
        create.side_effect = None
        create.return_value = response({'claims': [{'text': source, 'evidence': [ref]}]})
        llm.answer(client, config, question, hits, details={})
        assert translated not in create.call_args.kwargs['messages'][1]['content'], '한국어 단독 질문에 이전 번역이 섞였습니다.'
        # 숫자·부호·단위·코드 변경, 빈 값, 잘림은 원문 검색으로 돌아갑니다.
        numeric = '센서 SCW132 유량이 11.8 L/min 입니다. 과거 조치는?'
        for value in (numeric, '', 'Sensor SCW123 flow is 11.8 L/min.', 'Sensor SCW132 flow is -11.8 L/min.',
                      'Sensor SCW132 flow is 11.8 L/s.', 'Sensor SCW132 flow is 11.8.', None):
            create.return_value = response({'query': value})
            detail = {}
            assert english.bilingual_retrieve(db, numeric, client, config, details=detail) == search.retrieve(db, numeric)
            assert detail['search']['translation_status'] == 'failed' and not detail['search']['english_query']
        for data, finish in (({'query': ''}, 'stop'), ({'query': translated}, 'length'), ({}, 'stop')):
            create.return_value = response(data, finish)
            detail = {}
            assert english.bilingual_retrieve(db, question, client, config, details=detail) == hits
            assert detail['search']['translation_status'] == 'failed' and not detail['search']['english_query']
        create.side_effect = APITimeoutError(request=Mock())
        detail = {}
        assert english.bilingual_retrieve(db, question, client, config, details=detail) == hits
        assert detail['search']['translation_status'] == 'failed'
        long_question = ('센서 교체 후에도 증상이 지속됩니다. 재기동 직후에는 사라지지만 야간 고속 운전 중 다시 발생합니다. '
                         '누수는 없고 원인은 확인되지 않았습니다. 유사한 과거 사례에서 어떤 후속 조치를 했고, '
                         '같은 운전 조건에서 관찰한 결과와 남은 제약은 무엇인가요?')
        long_translation = ('The symptom persists after sensor replacement. It disappears immediately after restarting '
                            'but recurs during high speed operation at night. There is no water leak and the cause '
                            'has not been confirmed. In similar past cases, what subsequent actions were performed, '
                            'what outcomes were observed under the same operating conditions, and what limitations remained?')
        assert len(long_translation.split()) > 48
        create.side_effect = None
        create.return_value = response({'query': long_translation})
        detail = {}
        english.bilingual_retrieve(db, long_question, client, config, details=detail, search_english=False)
        assert detail['search']['english_query'] == long_translation, '긴 질문의 조건을 검색어 개수 제한으로 버렸습니다.'
        create.side_effect = AssertionError('영어 질문 또는 없는 ID를 번역했습니다.')
        detail = {}
        english.bilingual_retrieve(db, translated, client, config, details=detail)
        assert detail['search']['english_query'] == translated and detail['search']['translation_status'] == 'original'
        assert english.bilingual_retrieve(db, 'BQ-DEMO-999 조치는?', client, config) == []
        # 대조 구절은 두 언어 검색에만 적용하고, 답변에는 전체 질문과 번역을 보냅니다.
        create.side_effect = None
        create.return_value = response({'현상': 'Sensor signal failure.', '조치내용(1)':
            'The symptom persisted after sensor replacement. Wiring contact corrected, then normal.'})
        assert english.build_english(db, client, config, schema, limit=0)['indexed'] == 1
        contrast = '팬 소음이 아니라 센서 신호 불량입니다.'
        translation = 'The issue is not fan noise but a sensor signal failure.'
        create.return_value = response({'query': translation})
        detail = {}
        hits = english.bilingual_retrieve(db, contrast, client, config, details=detail)
        assert hits and detail['search']['korean_ids'] and detail['search']['english_ids']
        assert detail['search']['english_query'] == translation
        assert json.loads(create.call_args.kwargs['messages'][1]['content']) == {'question': contrast}
        _, messages = llm.case_answer_messages(contrast, hits, english_question=translation)
        assert contrast in messages[1]['content'] and translation in messages[1]['content']
        assert source in hits[0]['content'] and db.execute('SELECT original_json FROM normalized_records').fetchone()[0] == original
    print('PASS: 질문 번역 재사용·한국어 원문 근거·보정·문맥 예약·번역 실패 대체·영어 입력')


def main(root):
    check_bilingual_normalization()
    bilingual_questions(root)
    translation_repairs(root)
    parallel_translation(root)
    schema, rules = cslist.load_schema(root / 'cslist_schema.json')
    template = records.read_excel(root / 'cslist_sample.xlsx', 'CSList',
                                 required_columns=schema['columns'], id_column=schema['id_column'])[0]
    cells = {'현상': '관로 기밀 저하', '조치내용(1)': '씰 발주완. 교체 안함. 누설 위치 모름. 10분 관찰.'}
    translated = {'현상': 'Vacuum integrity deteriorating.',
                  '조치내용(1)': 'Seal ordered. Not replaced. Leak location unknown. Observed for 10분 (minutes).'}
    rows = []
    for index in range(2000):
        row = copy.deepcopy(template)
        row.update(dict.fromkeys(schema['columns'], ''))
        row.update(cells)
        row.update({schema['id_column']: f'EN-DEMO-{index+1:04d}', '_typed_cells': {}})
        rows.append(row)
    config = {'model': 'test-model', 'base_url': 'http://127.0.0.1:1/v1', 'reasoning_effort': 'high'}
    response = lambda data, finish='stop': SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish, message=SimpleNamespace(content=json.dumps(data, ensure_ascii=False)))])
    create = Mock(return_value=response(translated))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with closing(sqlite3.connect(':memory:')) as db:
        storage.ingest(db, rows, rules, schema=schema)
        before = db.execute('SELECT * FROM normalized_records ORDER BY id').fetchall()
        details = {}
        assert english.bilingual_retrieve(db, '진공 유지 실패', client, config, details=details) == []
        assert details['search']['warning'] and details['search']['translation_status'] == 'failed'
        create.reset_mock()  # 문서 번역 응답은 질문 형식이 아니므로 거절합니다. 아래는 문서 생성 호출 수입니다.
        # 1,000건 시험 뒤 전체 처리. 병렬 묶음 안의 동일 내용도 한 번만 요청합니다.
        with patch.object(english, 'ThreadPoolExecutor', wraps=english.ThreadPoolExecutor) as pool:
            result = english.build_english(db, client, config | {'normalization_workers': 1000}, schema, limit=1000)
            pool.assert_called_once_with(max_workers=1000)
        assert result['indexed'] == 1000 and result['generated'] == 1 and result['reused'] == 999, result
        result = english.build_english(db, client, config, schema, limit=0)
        assert result['indexed'] == 2000 and result['reused'] == 1000 and create.call_count == 1, result
        assert english.build_english(db, client, config, schema, limit=0)['processed'] == 0
        assert create.call_count == 1
        assert db.execute('SELECT * FROM normalized_records ORDER BY id').fetchall() == before
        assert db.execute('SELECT COUNT(*) FROM cell_reviews').fetchone()[0] == 0
        assert 'max_tokens' not in create.call_args.kwargs and 'max_completion_tokens' not in create.call_args.kwargs
        assert create.call_args.kwargs['extra_body']['reasoning_effort'] == 'high'
        create.return_value = response({'query': 'vacuum deteriorate'})
        hits = english.bilingual_retrieve(db, '진공 유지 실패', client, config, details=details)
        assert hits and details['search']['mode'] == 'bilingual' and not details['search']['korean_ids']
        assert hits[0]['id'] == 'EN-DEMO-0001/문서'
        assert 'Vacuum' not in hits[0]['content'] and hits[0]['evidence'][0]['cells']['현상'] == cells['현상']
        assert 'deteriorate' not in translated['현상'].casefold(), 'Porter 어형 검사가 필요합니다.'
        claim = {'claims': [{'text': cells['조치내용(1)'], 'evidence': [
            {'id': 'EN-DEMO-0001/작업1', 'column': '조치내용(1)', 'quote': cells['조치내용(1)']}]}]}
        assert '교체 안함' in llm.validate_claims(claim, hits)
        prior_calls = create.call_count
        assert english.bilingual_retrieve(db, 'EN-DEMO-9999 조치는?', client, config) == []
        assert create.call_count == prior_calls  # 없는 ID를 번역으로 우회하지 않습니다.
        create.return_value = response({'query': 'vacuum E999'})
        assert english.bilingual_retrieve(db, '진공 유지 실패', client, config, details=details) == []
        assert details['search']['warning']
        create.return_value = response({}, finish='length')
        question = '관로 기밀 저하'
        assert english.bilingual_retrieve(db, question, client, config) == search.retrieve(db, question)
        # 채택문이 바뀌면 해당 접수의 영어 색인만 빠지고 원문 검색·기존 검토는 유지됩니다.
        entry = {'source_key': rows[0]['_source_key'], 'id': rows[0][schema['id_column']],
                 'revision': records.row_revision(rows[0]), 'column': '현상', 'action': 'edit',
                 'value': '관로 기밀이 저하되었습니다.'}
        storage.apply_reviews(db, {'format': 'cell-reviews-v1', 'entries': [entry]}, rules, schema=schema)
        assert english.english_count(db) == 1999
        assert db.execute("SELECT 1 FROM rag_english_documents WHERE record_id='EN-DEMO-0001'").fetchone() is None
        create.return_value = response(translated)
        assert english.build_english(db, client, config, schema, limit=0)['generated'] == 1
        assert english.english_count(db) == 2000
        # 번역 모델 설정을 바꾸면 새 요청을 하고, 실패해도 다음 행까지 처리합니다.
        # 첫 접수는 원 요청과 셀별 보정까지 실패하고, 다음 접수는 정상 처리합니다.
        create.side_effect = [response({}, finish='length')] * 7 + [response(translated)]
        result = english.build_english(db, client, config | {'model': 'next-model'}, schema, limit=2)
        assert len(result['failures']) == 1 and result['generated'] == 1, result
        assert db.execute('SELECT value FROM cell_reviews').fetchone()[0] == entry['value']
    # 누락 열·숫자/단위 변경·식별자 변경은 저장 전에 거절합니다.
    for original, wrong in [({'현상': '10분 관찰'}, {'현상': 'Observed 10 minutes'}),
                            ({'현상': 'E202 발생'}, {'현상': 'F202 occurred'}),
                            ({'현상': '펌프 누설'}, {}),
                            ({'현상': '누수 없음'}, {'현상': '누수 없음'})]:
        try:
            english.validate_translation(wrong, original)
        except ValueError:
            pass
        else:
            raise AssertionError((original, wrong))
    # 콘솔도 UI와 같은 저장·검색 함수를 거치고, --saved로 재정리 없이 질문합니다.
    from ragkit import cli
    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder)
        source = records.read_excel(root / 'cslist_sample.xlsx', 'CSList',
                                    required_columns=schema['columns'], id_column=schema['id_column'])
        with closing(sqlite3.connect(output / 'demo.sqlite3')) as db:
            storage.ingest(db, source, rules, schema=schema)
        def reply(**kwargs):
            prompt, user = [message['content'] for message in kwargs['messages']]
            if prompt.startswith(english.TRANSLATE_PROMPT):
                return response({key: value + ' maintenance' for key, value in json.loads(user).items()})
            if prompt == english.QUERY_PROMPT:
                return response({'query': 'maintenance'})
            sources = json.loads(user.split('원문 자료:\n')[1])
            section = next(item for item in sources if item['cells'].get('조치내용(1)'))
            return response({'claims': [{'text': section['cells']['조치내용(1)'], 'evidence': [
                {'id': section['id'], 'column': '조치내용(1)'}]}]})
        client.chat.completions.create = Mock(side_effect=reply)
        client.close = lambda: None
        args = ['--schema', str(root / 'cslist_schema.json'), '--output', str(output)]
        with patch.object(cli, 'OpenAI', return_value=client):
            cli.main(root, args + ['--translate-english', '--translation-limit', '2'])
            cli.main(root, args + ['--saved', '--bilingual', '--question', '소모품 정비'])
        report = json.loads((output / 'demo_results.json').read_text())
        assert report['cases'][0]['quality']['search']['mode'] == 'bilingual'
        assert report['cases'][0]['error'] is None
    print('PASS: 영어 검색 2000건·내용 재사용·한국어 인용·잘림/숫자 검사·실패 후 계속·검토 후 무효화')


def check_bilingual_normalization():
    """한 요청의 두 결과를 따로 검사하고 잘못된 번역이 한국어 성공을 취소하지 않습니다."""
    from ragkit import llm
    from types import SimpleNamespace as NS
    row = {'id': 'PAIR-1', '제목': '웨이퍼 확인',
           '현상': '1. iur low arm 으로 ifep2에서 pick한 wafer', '조치': '교체 안함. 30분 관찰', '원인': ''}
    rules = {key: 'rewrite' if key in ('현상', '조치', '원인') else 'preserve' for key in row}
    original = dict(row)
    calls = []
    bad = set()
    def create(**request):
        payload = json.loads(request['messages'][1]['content']); calls.append(payload)
        if '영어로 번역할 열' in payload:
            ko = {key: payload['원본 셀'][key] for key in payload['수정할 열']}
            en = {key: payload['원본 셀'][key] + ' maintenance' for key in payload['영어로 번역할 열']}
            if 'english' in bad:
                en['조치'] = ''
            if 'korean' in bad:
                ko['현상'] = 'ifem2 변경'
            values = {'korean': ko, 'english': en}
            if 'missing_english' in bad:
                values.pop('english')
        else:
            assert payload['수정할 열'] == ['현상'], '통과한 한국어 셀도 다시 요청했습니다.'
            values = {'현상': payload['원본 셀']['현상'] + ' 확인했습니다.'}
        return NS(choices=[NS(finish_reason='stop', message=NS(content=json.dumps(values, ensure_ascii=False)))])
    client = NS(chat=NS(completions=NS(create=create)))
    config = {'model': 'test', 'reasoning_effort': 'high'}
    result = english.normalize_bilingual(client, config, row, rules)
    assert len(calls) == 1 and result['normalized'] == row and row == original
    assert set(calls[0]['수정할 열']) == {'현상', '조치'} and '원인' not in calls[0]['영어로 번역할 열']
    assert result['english']['현상'] == row['현상'] + ' maintenance'
    assert result['english']['제목'] == row['제목'] + ' maintenance'
    assert not result['english_failures'] and '__CS_NUMBER_' not in json.dumps(result)
    bad.add('english'); calls.clear()
    result = english.normalize_bilingual(client, config, row, rules)
    assert len(calls) == 1 and result['normalized'] == row
    assert '조치' in result['english_failures'] and '조치' not in result['english']
    assert '현상' in result['english'], '정상 번역을 버렸습니다.'
    bad.clear(); bad.add('missing_english'); calls.clear()
    result = english.normalize_bilingual(client, config, row, rules)
    assert len(calls) == 1 and result['normalized'] == row and not result['english']
    assert set(result['english_failures']) == {'제목', '현상', '조치'}
    bad.clear(); bad.add('korean'); calls.clear()
    result = english.normalize_bilingual(client, config, row, rules)
    assert len(calls) == 2 and result['normalized']['현상'].endswith('확인했습니다.')
    assert '현상' not in result['english'] and '현상' in result['english_failures']
    assert '조치' in result['english'] and row == original
    llm.validate_normalized(result['normalized'], row, rules)
    # 실제 누적 양식에서도 실패 셀 보정 규칙을 유지합니다.
    from ragkit.input_formats import HISTORY_RULES, HISTORY_SCHEMA, HISTORY_SHEET, HISTORY_COLUMNS
    history = {key: '' for key in HISTORY_COLUMNS} | {'CS NO': 'PAIR-2', '현상': row['현상'], '조치': row['조치'],
               '_id_column': 'CS NO', '_source_sheet': HISTORY_SHEET, '_typed_cells': {}}
    result = english.normalize_bilingual(client, config, history, HISTORY_RULES, HISTORY_SCHEMA)
    assert result['normalized']['CS NO'] == 'PAIR-2' and '현상' not in result['english']
    # 작업1 문맥에는 작업2의 셀을 전달하지 않습니다.
    group_row = {'CS 접수번호': 'PAIR-3', '제목': '작업 기록', '조치내용(1)': '씰 교체', '조치내용(2)': '미실시'}
    schema = {'id_column': 'CS 접수번호', 'context_columns': list(group_row), 'rewrite_columns': ['조치내용(1)', '조치내용(2)']}
    group_row['_id_column'] = 'CS 접수번호'
    group_rules = {key: 'rewrite' if key.startswith('조치내용') else 'preserve' for key in group_row}
    bad.clear(); calls.clear()
    result = english.normalize_bilingual(client, config, group_row, group_rules, schema)
    assert len(calls) == 2 and set(result['english']) == {'제목', '조치내용(1)', '조치내용(2)'}
    assert '조치내용(2)' not in calls[0]['원본 셀'] and '조치내용(1)' not in calls[1]['원본 셀']
    print('PASS: 통합 요청·숫자 복원·영어 실패 격리·실패한 한국어 셀만 보정·번역 기준 변경 시 재사용 금지')


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1] / 'windows-demo')
