"""합성 원문과 모의 응답만으로 상세 근거·년월·후속 질문을 검사합니다."""
from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ragkit import llm, storage, web
from ragkit.input_formats import HISTORY_COLUMNS, HISTORY_SHEET


def response(payload):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
        message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False)))])


class DetailClaimsTests(unittest.TestCase):
    def setUp(self):
        self.a = {'id': '누적28상세:aaa/접수', 'kind': '접수', 'cells': {
            'CS NO': 'CS-TEST-01', '현상': '펌프 누수', '조치': '씰 교체',
            '기인 Unit(부품)': '펌프 씰', '불량발생년월': '2026-09'}}
        self.b = {'id': '누적28상세:bbb/접수', 'kind': '접수', 'cells': {
            'CS NO': 'CS-TEST-01', '현상': '펌프 누수',
            '조치': '바이패스 배관 밸브를 전면 교체 완료함',
            '기인 Unit(부품)': '배관 밸브', '불량발생년월': '2026-08'}}
        self.hits = [{'evidence': [self.a]}, {'evidence': [self.b]}]

    def claim(self, text, source, *columns):
        return {'text': text, 'evidence': [{'id': source['id'], 'column': key} for key in columns]}

    def test_detail_action_requires_its_original_cell(self):
        copied = self.b['cells']['조치']
        with self.assertRaises(llm.ClaimValidationError) as error:
            llm.validate_claims({'claims': [self.claim(copied, self.a, '현상')]}, self.hits)
        self.assertEqual((error.exception.category, error.exception.column), ('evidence', '조치'))
        self.assertIn(self.b['id'], str(error.exception))
        self.assertIn(copied, llm.validate_claims(
            {'claims': [self.claim(copied, self.b, '조치')]}, self.hits))
        # 다른 CS의 문장을 복사하거나 채택 번역만 인용해도 근거가 되지 않습니다.
        other = deepcopy(self.b)
        other['cells']['CS NO'] = 'CS-TEST-02'
        with self.assertRaises(llm.ClaimValidationError):
            llm.validate_claims({'claims': [self.claim(copied, self.a, '현상')]},
                                [{'evidence': [self.a, other]}])
        translated = self.a | {'accepted': {'조치': copied}}
        with self.assertRaises(llm.ClaimValidationError):
            llm.validate_claims({'claims': [self.claim(copied, translated, '조치')]},
                                [{'evidence': [translated, self.b]}])
        # 기존 작업 경계도 동일하게 검사합니다.
        work1 = {'id': 'CS-TEST-01/작업1', 'cells': {'조치내용(1)': '씰 교체'}}
        work2 = {'id': 'CS-TEST-01/작업2', 'cells': {'조치내용(2)': copied}}
        with self.assertRaises(llm.ClaimValidationError):
            llm.validate_claims({'claims': [self.claim(copied, work1, '조치내용(1)')]},
                                [{'evidence': [work1, work2]}])

    def test_history_action_supplements_keep_detail_scope(self):
        claims = [self.claim('펌프 누수', self.a, '현상')]
        items = llm.case_source_supplements(claims, self.hits, '과거 조치는?')
        self.assertEqual([(e['id'], e['column'], e['quote']) for e in items[0]['evidence']],
            [(self.a['id'], '현상', '펌프 누수'), (self.a['id'], '조치', '씰 교체')])
        self.assertIn('CS-TEST-01', items[0]['text'])
        self.assertNotIn('누적28상세:', items[0]['text'])
        self.assertNotIn('조치내용 기록이 없습니다', items[0]['text'])
        self.assertNotIn(self.b['cells']['조치'], items[0]['text'])
        for question in ('현상만 알려주세요.', '과거 원인만 알려주세요.', 'MTTR은?', '마지막 조치는?'):
            self.assertEqual(llm.case_source_supplements(claims, self.hits, question), [])
        self.assertEqual(len(llm.case_source_supplements(
            [self.claim('씰 교체', self.a, '조치')], self.hits, '마지막 조치는?')), 1)
        blank = self.a | {'cells': self.a['cells'] | {'조치': ''}}
        item = llm.case_source_supplements(claims, [{'evidence': [blank]}], '과거 조치는?')[0]
        self.assertEqual(item['evidence'][-1]['quote'], '')
        self.assertIn('미기재', item['text'])
        from ragkit.web import pack_answer
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: response({'claims': deepcopy(claims)}))))
        details = {}
        reply, _ = llm.answer(client, {'model': 'mock', 'reasoning_effort': 'high'},
                              '과거 조치는?', self.hits, details=details)
        self.assertIn('씰 교체', reply)  # CLI도 사용하는 실제 answer 경로입니다.
        packed = pack_answer('과거 조치는?', reply, self.hits, None, details, 0)
        self.assertEqual([(e['receipt'], e['column'], e['quote'])
            for item in packed['source_supplements'] for e in item['evidence']],
            [('CS-TEST-01', '조치', '씰 교체')])

    def test_claim_does_not_join_different_details_or_receipts(self):
        for second in (self.b, self.b | {'cells': self.b['cells'] | {'CS NO': 'CS-TEST-02'}}):
            claim = self.claim('펌프 누수 현상에 대해 ' + second['cells']['조치'], self.a, '현상')
            claim['evidence'].append({'id': second['id'], 'column': '조치'})
            with self.subTest(receipt=second['cells']['CS NO']), self.assertRaises(llm.ClaimValidationError):
                llm.validate_claims({'claims': [claim]}, [{'evidence': [self.a, second]}])
        legacy = self.a | {'id': '누적28:CS-TEST-01/접수'}
        claim = self.claim('펌프 누수 현상에 대해 ' + self.b['cells']['조치'], legacy, '현상')
        claim['evidence'].append({'id': self.b['id'], 'column': '조치'})
        with self.assertRaises(llm.ClaimValidationError):
            llm.validate_claims({'claims': [claim]}, [{'evidence': [legacy, self.b]}])

    def test_year_month_preserves_precision_and_measurements(self):
        source = self.a | {'cells': self.a['cells'] | {'압력': '-0.2 MPa', '측정값': '2026.09 V', '단위 없는 값': '2026.09'}}
        for text in ('2026년 9월', '2026-9', '2026/09', '2026.09', '2026-09.', '2026/09.', '2026.09.'):
            with self.subTest(text=text):
                self.assertIn(text, llm.validate_claims({'claims': [self.claim(
                    '불량발생년월은 ' + text + '입니다.', source, '불량발생년월')]}, [{'evidence': [source]}]))
        for text in ('2026년 8월', '2026년 9월 1일', '2026-09-01', '2026-09/01', '2026년 13월'):
            with self.subTest(unsupported=text), self.assertRaises(ValueError):
                llm.validate_claims({'claims': [self.claim(text, source, '불량발생년월')]}, [{'evidence': [source]}])
        day = source | {'cells': source['cells'] | {'불량발생일': '2026-09-01'}}
        with self.assertRaises(ValueError):
            llm.validate_claims({'claims': [self.claim('2026년 9월', day, '불량발생일')]}, [{'evidence': [day]}])
        for changed in ('압력 0.2 MPa', '압력 -0.2 kPa'):
            with self.subTest(measurement=changed), self.assertRaises(ValueError):
                llm.validate_claims({'claims': [self.claim('2026년 9월, ' + changed, source,
                    '불량발생년월', '압력')]}, [{'evidence': [source]}])
        self.assertIn('2026.09 V', llm.validate_claims({'claims': [self.claim(
            '측정값은 2026.09 V입니다.', source, '측정값')]}, [{'evidence': [source]}]))
        self.assertIn('2026.09', llm.validate_claims({'claims': [self.claim(
            '단위 없는 값은 2026.09입니다.', source, '단위 없는 값')]}, [{'evidence': [source]}]))
        with self.assertRaises(ValueError):
            llm.validate_claims({'claims': [self.claim('2026년 9월, 단위 없는 값은 2026.9입니다.',
                source, '불량발생년월', '단위 없는 값')]}, [{'evidence': [source]}])
        with self.assertRaises(ValueError):
            llm.validate_claims({'claims': [self.claim('2026년 9월', source, '측정값')]}, [{'evidence': [source]}])

    def test_follow_up_ids_are_validated_against_server_history(self):
        history = [{'question': '과거 조치는?', 'answer_claims': [
            {'text': source['cells']['조치'], 'receipts': ['CS-TEST-01'], 'sources': [
                {'id': source['id'], 'receipt': 'CS-TEST-01', 'cells': source['cells']}]}
            for source in (self.a, self.b)]}]
        def resolve(payload, turns=history, language='ko'):
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response(payload))))
            return llm.resolve_follow_up(client, {'model': 'mock', 'reasoning_effort': 'high',
                'answer_instruction_language': language}, '두번째 사례의 조치는?', turns)
        scoped = {'question': 'CS NO=CS-TEST-01의 조치는?', 'source_ids': [self.b['id']]}
        for language in ('ko', 'en'):
            self.assertEqual(resolve(scoped, language=language), scoped)
        self.assertIn('clarification', resolve({'question': scoped['question']}))
        for identities in ([], ['다른 세션/접수'], [True], [self.b['id']] * 5):
            with self.subTest(ids=identities), self.assertRaises(ValueError):
                resolve(scoped | {'source_ids': identities})
        old = [{'question': '조치는?', 'answer_claims': [{'text': '씰 교체', 'receipts': ['CS-OLD']}]}]
        self.assertEqual(resolve({'question': 'CS-OLD의 조치는?'}, old), {'question': 'CS-OLD의 조치는?'})
        with self.assertRaises(ValueError):
            resolve({'question': 'q' * 2001}, old)

    def test_year_month_datetime_does_not_create_a_day(self):
        for column in ('불량발생년월', '반입년월', '불량발생월'):
            for original in ('2026-09-01T00:00:00', '2026-09-01', '2026.09.01'):
                source = self.a | {'cells': self.a['cells'] | {column: original}}
                with self.subTest(column=column, original=original):
                    payload = {'claims': [self.claim('해당 년월은 2026년 9월입니다.', source, column)]}
                    self.assertIn('2026년 9월', llm.validate_claims(payload, [{'evidence': [source]}]))
                    self.assertEqual(payload['claims'][0]['evidence'][0]['quote'], original)
                    for unsupported in ('2026년 9월 1일', '2026년 9월 2일', '2026년 8월', '측정값은 00입니다.'):
                        with self.subTest(unsupported=unsupported), self.assertRaises(ValueError):
                            llm.validate_claims({'claims': [self.claim(unsupported, source, column)]}, [{'evidence': [source]}])

    def test_invalid_year_month_is_retried_once(self):
        bad = {'claims': [self.claim('불량발생년월은 2026년 13월입니다.', self.a, '불량발생년월')]}
        good = {'claims': [self.claim('불량발생년월은 2026년 9월입니다.', self.a, '불량발생년월')]}
        config = {'model': 'mock', 'reasoning_effort': 'high'}
        create = Mock(side_effect=[response(bad), response(good)])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        details = {}
        reply, _ = llm.answer(client, config, '불량발생년월은?', self.hits, details=details)
        self.assertIn('2026년 9월', reply)
        self.assertEqual(details['attempts'], 2)
        self.assertEqual(details['validation_events'][0]['outcome'], 'succeeded')
        self.assertEqual(details['validation_events'][1]['category'], 'passed')
        create = Mock(side_effect=[response(bad), response(bad)])
        client.chat.completions.create = create
        with self.assertRaises(ValueError):
            llm.answer(client, config, '불량발생년월은?', self.hits)
        self.assertEqual(create.call_count, 2)

    def test_follow_up_metadata_is_bounded_and_original_only(self):
        source = deepcopy(self.a)
        source['cells']['기인 모듈'] = '긴' * 161
        source['cells']['Part명(100)'] = '수집 범위 밖'
        source['accepted'] = {'기인 Unit(부품)': '보조 번역'}
        refs = [self.claim('조치', source, '조치', '현상')]
        packed = web.pack_answer('조치는?', '', [{'evidence': [source]}], None, {'claims': refs}, 0)
        metadata = packed['claims'][0]['sources']
        self.assertEqual(len(metadata), 1)
        self.assertEqual(metadata[0]['cells']['기인 Unit(부품)'], '펌프 씰')
        self.assertNotIn('기인 모듈', metadata[0]['cells'])
        self.assertNotIn('Part명(100)', metadata[0]['cells'])
        self.assertTrue(packed['claims'][0]['sources_truncated'])
        sections = [source | {'id': f'CS-TEST-01/작업{n}'} for n in range(1, 6)]
        claims = [{'text': '조치', 'evidence': [{'id': s['id'], 'column': '조치'} for s in sections]}]
        packed = web.pack_answer('조치는?', '', [{'evidence': sections}], None, {'claims': claims}, 0)
        self.assertEqual(len(packed['claims'][0]['sources']), 4)
        self.assertTrue(packed['claims'][0]['sources_truncated'])

    def test_follow_up_retrieves_second_detail_originals(self):
        root = Path(__file__).resolve().parents[1]
        demo_root = root / 'windows-demo' if (root / 'windows-demo').is_dir() else root
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            config = folder / 'config.json'
            config.write_text(json.dumps({'base_url': 'test://offline', 'api_key': 'EMPTY',
                'model': 'mock', 'reasoning_effort': 'high', 'timeout_seconds': 1}))
            app = web.App(demo_root, config, folder / 'out', demo_root / 'cslist_schema.json')
            rows = [{key: source['cells'].get(key, '') for key in HISTORY_COLUMNS} | {
                '_id_column': 'CS NO', '_source_sheet': HISTORY_SHEET, '_history_detail': True}
                for source in (self.a, self.b)]
            with closing(app.connect()) as db:
                storage.ingest(db, rows, app.rules, schema=app.schema)
            other_owner = app.session(SimpleNamespace(headers={}), create=True)
            resolved = 'CS NO=CS-TEST-01의 조치는?'
            resolutions = []
            def create(**request):
                user = request['messages'][1]['content']
                if request['messages'][0]['content'] == llm.FOLLOW_UP_PROMPTS['ko']:
                    data = json.loads(user)
                    self.assertNotIn('CLIENT-HISTORY', user)
                    self.assertLessEqual(len(data['history']), 3)
                    second = data['history'][0]['answer_claims'][1]['sources'][0]
                    self.assertEqual(second['receipt'], 'CS-TEST-01')
                    self.assertTrue(second['id'].startswith('누적28상세:'))
                    self.assertEqual(second['cells']['기인 Unit(부품)'], '배관 밸브')
                    self.assertEqual(second['cells']['불량발생년월'], '2026-08')
                    self.assertNotIn('조치', second['cells'])
                    resolutions.append(data)
                    return response({'clarification': '어느 부품의 사례인가요?'} if data['question'] == '모호한 사례'
                                    else {'question': resolved, 'source_ids': [second['id']]})
                self.assertNotIn('"history"', user)
                sources = json.loads(user.split('원문 자료:\n')[1])
                sources.sort(key=lambda s: s['cells'].get('기인 Unit(부품)') == '배관 밸브')
                if user.startswith('질문: ' + resolved + '\n'):
                    self.assertEqual(len(sources), 1, 'FTS 순위에 맡기지 않고 DB에서 상세를 제한해야 합니다.')
                    self.assertEqual(sources[0]['cells']['기인 Unit(부품)'], '배관 밸브')
                return response({'claims': [self.claim(s['cells']['조치'], s, '조치') for s in sources]})
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=lambda: None)
            with patch.object(web, 'OpenAI', return_value=client), patch.object(threading.Thread, 'start'), \
                    patch.object(web, 'retrieve', wraps=web.retrieve) as retrieval, \
                    patch.object(web.semantic, 'search_cases', wraps=web.semantic.search_cases) as semantic, \
                    patch.object(web, 'bilingual_retrieve', wraps=web.bilingual_retrieve) as bilingual_search:
                app.ask({'question': 'CS NO=CS-TEST-01 과거 조치는?', 'reference_limit': 0}, owner=app.token)
                app.run_questions()
                self.assertEqual(app.job_snapshot(app.token)['status'], 'done')
                first_history = deepcopy(app.sessions[app.token]['conversation'])
                self.assertEqual(len(first_history[0]['answer_claims']), 2)
                self.assertIn('sources', first_history[0]['answer_claims'][1],
                              '후속 질문용 문맥에서 상세 ID와 원문 부품·날짜가 사라졌습니다.')
                target = first_history[0]['answer_claims'][1]['sources'][0]['id'].rsplit('/', 1)[0]
                for mode in ('keyword', 'vector', 'hybrid'):
                    for bilingual in (False, True):
                        with self.subTest(mode=mode, bilingual=bilingual):
                            # 본문과 클라이언트 문맥이 오염돼도 검증된 ID로 현재 원문을 다시 읽습니다.
                            app.sessions[app.token]['conversation'] = deepcopy(first_history)
                            app.sessions[app.token]['conversation'][0]['answer_claims'][1]['text'] = '가짜 영구 해결'
                            app.ask({'question': '두번째 사례의 조치는?', 'follow_up': True,
                                'history': 'CLIENT-HISTORY', 'record_ids': ['다른 상세'],
                                'search_mode': mode, 'bilingual': bilingual}, owner=app.token)
                            app.run_questions()
                            job = app.job_snapshot(app.token)
                            self.assertEqual(job['status'], 'done', job['error'])
                            self.assertEqual(job['result']['resolved_question'], resolved)
                            self.assertIn('바이패스 배관 밸브를 전면 교체 완료함', job['result']['answer'])
                            self.assertNotIn('가짜 영구 해결', job['result']['answer'])
                            if mode != 'keyword':
                                self.assertEqual(semantic.call_args.kwargs['record_ids'], [target])
                                self.assertEqual(job['result']['vector_search']['mode'], 'exact_detail')
                            if bilingual:
                                self.assertEqual(bilingual_search.call_args.kwargs['record_ids'], [target])
                                self.assertEqual(job['result']['search']['mode'], 'exact_detail')
                self.assertEqual(retrieval.call_count, 2)
                self.assertEqual(retrieval.call_args.args[1], resolved)
                self.assertEqual(retrieval.call_args.kwargs['record_ids'], [target])
                self.assertEqual(app.job_snapshot(other_owner)['conversation'], [])
                before = deepcopy(app.sessions[app.token]['conversation'])
                app.ask({'question': '모호한 사례', 'follow_up': True}, owner=app.token)
                app.run_questions()
                self.assertEqual(app.job_snapshot(app.token)['result']['status'], 'clarification')
                self.assertEqual(app.sessions[app.token]['conversation'], before)
                self.assertEqual(retrieval.call_count, 2)
                self.assertEqual(len(resolutions), 7)
                # 원문 문서가 없어졌다면 같은 CS의 첫 번째 상세로 대신 답하지 않습니다.
                with closing(app.connect()) as db:
                    db.execute('DELETE FROM rag_documents WHERE record_id=?', (target,))
                    db.commit()
                app.ask({'question': '두번째 사례의 조치는?', 'follow_up': True}, owner=app.token)
                app.run_questions()
                job = app.job_snapshot(app.token)
                self.assertEqual(job['status'], 'done', job['error'])
                self.assertEqual(job['result']['status'], 'no_match')
                self.assertEqual(job['result']['claims'], [])


if __name__ == '__main__':
    unittest.main()
