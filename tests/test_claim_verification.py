"""운영 답변의 원문 우선·추가 검사·실패 차단을 네트워크 없이 확인합니다."""
from copy import deepcopy
import json
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from openai import APITimeoutError
import httpx2

from ragkit import llm, nli, claim_verification as guard


def response(payload):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
        message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False)))])


class ClaimVerificationTests(unittest.TestCase):
    def setUp(self):
        self.config = {'model': 'mock', 'reasoning_effort': 'high', 'claim_guard_mode': 'off'}
        self.source = {'id': 'CS-TEST-A/작업1', 'kind': '작업1',
                       'cells': {'조치내용(1)': 'Seal replacement is planned.'}}
        self.hits = [{'evidence': [self.source]}]
        self.payload = {'claims': [{'text': '씰 교체를 완료했습니다.',
            'evidence': [{'id': self.source['id'], 'column': '조치내용(1)'}]}]}
        self.complete = Mock(return_value=response(self.payload))
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.complete)))

    def test_invalid_mode_is_rejected_before_model_request(self):
        with self.assertRaisesRegex(ValueError, 'claim_guard_mode'):
            llm.answer(self.client, self.config | {'claim_guard_mode': 'enforcce'}, '조치는?', self.hits)
        self.complete.assert_not_called()

    def test_invalid_timeout_is_rejected_before_model_request(self):
        for value in (True, float('nan'), 0, 601, '120'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'claim_guard_timeout_seconds'):
                llm.answer(self.client, self.config | {'claim_guard_mode': 'enforce', 'claim_guard_timeout_seconds': value}, '조치는?', self.hits)
        self.complete.assert_not_called()

    def test_sdk_timeout_is_reported_separately_without_retry(self):
        self.complete.side_effect = APITimeoutError(request=httpx2.Request('POST', 'http://localhost/v1'))
        with self.assertRaises(guard.VerificationError) as error:
            guard.translate_text(self.client, self.config, '씰 교체 예정', kind='원문 셀', deadline=time.monotonic() + 30)
        self.assertEqual(error.exception.code, 'verification_timeout')
        self.assertEqual(self.complete.call_count, 1)

    def test_enforce_without_original_cells_cannot_bypass_guard(self):
        with self.assertRaisesRegex(RuntimeError, '원문'):
            llm.answer(self.client, self.config | {'claim_guard_mode': 'enforce'}, '조치는?',
                       [{'content': 'Seal replacement is planned.'}])
        self.complete.assert_not_called()

    def test_off_preserves_existing_behavior_and_original(self):
        before = deepcopy(self.hits)
        text, _ = llm.answer(self.client, self.config, '조치는?', self.hits)
        self.assertIn('씰 교체를 완료했습니다.', text)
        self.assertEqual(self.hits, before)
        self.assertEqual(self.complete.call_count, 1)

    def test_translation_preserves_number_unit_and_independent_input(self):
        self.complete.return_value = response({'text': 'Pressure __CS_NUMBER_0__.'})
        source = '압력 -0.2 kPa'
        value = guard.translate_text(self.client, self.config, source, kind='원문 셀', deadline=time.monotonic() + 30)
        self.assertEqual(value, 'Pressure -0.2 kPa.')
        request = self.complete.call_args.kwargs
        data = json.loads(request['messages'][1]['content'])
        self.assertEqual(set(data), {'text'})
        self.assertIn('-0.2 kPa', request['messages'][0]['content'])
        self.assertNotIn('씰 교체를 완료했습니다.', json.dumps(request, ensure_ascii=False))
        self.assertGreater(request['timeout'], 0)
        self.assertLessEqual(request['timeout'], 30)

    def test_translation_cannot_turn_plan_into_completion(self):
        self.complete.side_effect = [response({'text': 'The seal was replaced.'}),
                                    response({'text': 'Seal replacement is planned.'})]
        text = guard.translate_text(self.client, self.config, '씰 교체 예정', kind='원문 셀', deadline=time.monotonic() + 30)
        self.assertEqual(text, 'Seal replacement is planned.')
        self.assertEqual(self.complete.call_count, 2)
        self.assertIn('예정', self.complete.call_args.kwargs['messages'][0]['content'])

    def test_bad_translation_is_an_error_not_unsupported(self):
        self.complete.return_value = response({'text': 'The seal was replaced.'})
        with self.assertRaises(guard.VerificationError) as error:
            guard.translate_text(self.client, self.config, '씰 교체 예정', kind='원문 셀', deadline=time.monotonic() + 30)
        self.assertEqual(error.exception.code, 'verification_translation')
        self.assertEqual(self.complete.call_count, 2)

    def test_timeout_stops_before_a_request(self):
        with self.assertRaises(guard.VerificationError) as error:
            guard.translate_text(self.client, self.config, '씰 교체 예정', kind='원문 셀', deadline=time.monotonic() - 1)
        self.assertEqual(error.exception.code, 'verification_timeout')
        self.complete.assert_not_called()

    def test_blank_record_only_allows_a_scoped_absence_statement(self):
        claim = {'text': '조치 기록이 없습니다.', 'evidence': [{'column': '조치', 'quote': ''}]}
        self.assertTrue(guard.blank_record_claim(claim))
        for text in ('조치 기록이 없고 씰을 교체했습니다.', '원인 기록이 없습니다.',
                     '조치가 미기재되어 있지만 해결되었습니다.'):
            with self.subTest(text=text):
                self.assertFalse(guard.blank_record_claim(claim | {'text': text}))
        self.assertFalse(guard.blank_record_claim(claim | {'evidence': [{'column': '조치', 'quote': '씰 교체'}]}))

    def test_known_state_translation_guards(self):
        for source, wrong, correct in (
            ('씰 교체 예정', 'The seal was replaced.', 'Seal replacement is planned.'),
            ('씰 미교체', 'The seal was replaced.', 'The seal has not been replaced.'),
            ('暂时恢复', 'The equipment recovered.', 'The equipment temporarily recovered.'),
            ('Replacement is planned.', 'Replacement completed.', 'Replacement is scheduled.'),
            ('씰 교체를 완료했습니다.', 'Seal replacement is planned.', 'Seal replacement was completed.'),
            ('씰 교체 완료', 'The seal has not been replaced.', 'The seal was replaced.'),
            ('밸브 동작 정상', 'The valve temporarily operated normally.', 'The valve operated normally.'),
            ('씰 교체 예정', 'Seal replacement is planned and was completed.', 'Seal replacement is planned.'),
            ('씰 교체 필요', 'The seal was replaced.', 'Seal replacement is needed.'),
            ('씰 교체는 아직 완료되지 않았으며 예정 상태입니다.', 'The seal replacement was completed.',
             'Seal replacement has not yet been completed and is in a scheduled state.'),
        ):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    guard.check_translation_state(source, wrong)
                guard.check_translation_state(source, correct)
        guard.check_translation_state('씰 교체', 'The seal was replaced.')
        for source, correct in (
            ('씰 발주는 완료했고 교체는 하지 않았습니다.', 'The seal was ordered and was not replaced.'),
            ('씰 발주 완료. 교체 안함.', 'The seal was ordered. It was not replaced.'),
            ('원인 확인 대기 중. 필터 점검만 실시.', 'The cause is pending confirmation. Only a filter check was performed.'),
        ):
            with self.subTest(source=source):
                guard.check_translation_state(source, correct)

    def test_completed_claim_cannot_be_laundered_into_planned_translation(self):
        self.complete.return_value = response({'text': 'Seal replacement is planned.'})
        payload = deepcopy(self.payload)
        llm.validate_claims(payload, self.hits)
        with patch.object(nli, 'prepare_model'), patch.object(nli, 'score', return_value={'passed': True}) as scoring:
            result = guard.verify_claims(self.client, self.config | {'claim_guard_mode': 'enforce'}, payload['claims'])
        self.assertEqual((result['status'], result['code']), ('error', 'verification_translation'))
        self.assertEqual(self.complete.call_count, 2)
        scoring.assert_not_called()

    def test_original_rule_failure_never_reaches_nli(self):
        bad = deepcopy(self.payload)
        bad['claims'][0]['evidence'][0]['column'] = '다른 원문'
        self.complete.return_value = response(bad)
        with patch.object(guard, 'verify_claims') as verifying, self.assertRaisesRegex(ValueError, '근거 검사'):
            llm.answer(self.client, self.config | {'claim_guard_mode': 'enforce'}, '조치는?', self.hits)
        verifying.assert_not_called()
        self.assertEqual(self.complete.call_count, 2)

    def test_enforce_unsupported_uses_only_existing_single_repair(self):
        result = {'mode': 'enforce', 'status': 'unsupported', 'claims': [{'index': 1, 'status': 'unsupported'}]}
        details = {}
        with patch.object(guard, 'verify_claims', return_value=result), self.assertRaisesRegex(ValueError, '영어 근거 검사'):
            llm.answer(self.client, self.config | {'claim_guard_mode': 'enforce'}, '조치는?', self.hits, details=details)
        self.assertEqual(self.complete.call_count, 2)
        self.assertEqual(len(details['verification_attempts']), 2)
        self.assertEqual(details['validation_events'][-1]['category'], 'semantic')

    def test_enforce_infrastructure_error_is_not_answer_repair(self):
        result = {'mode': 'enforce', 'status': 'error', 'claims': [], 'code': 'verification_unavailable',
                  'message': '검사 모델 준비 필요'}
        details = {}
        with patch.object(guard, 'verify_claims', return_value=result), self.assertRaises(guard.VerificationError) as error:
            llm.answer(self.client, self.config | {'claim_guard_mode': 'enforce'}, '조치는?', self.hits, details=details)
        self.assertEqual(error.exception.code, 'verification_unavailable')
        self.assertEqual(self.complete.call_count, 1)
        self.assertNotIn('answer_text', details)

    def test_observe_exposes_warning_and_preserves_answer(self):
        for status in ('unsupported', 'error'):
            result = {'mode': 'observe', 'status': status, 'claims': [], 'message': '추가 검사 결과'}
            details = {}
            with patch.object(guard, 'verify_claims', return_value=result):
                text, _ = llm.answer(self.client, self.config | {'claim_guard_mode': 'observe'}, '조치는?', self.hits, details=details)
            self.assertIn('씰 교체를 완료했습니다.', text)
            self.assertEqual(details['claim_verification']['status'], status)
            self.assertIn('기록 모드', details['note'])

    def test_verified_claims_use_only_their_cited_original_cells(self):
        claims = [{'text': 'First claim.', 'evidence': [{'id': 'A/작업1', 'column': '조치', 'quote': 'First source.'}]},
                  {'text': 'Second claim.', 'evidence': [{'id': 'B/작업1', 'column': '조치', 'quote': 'Second source.'}]}]
        before = deepcopy(claims)
        score = {'label': 'entailment', 'support_probability': .9, 'passed': True, 'threshold': .8}
        with patch.object(nli, 'prepare_model'), patch.object(nli, 'score', return_value=score) as scoring:
            result = guard.verify_claims(self.client, self.config | {'claim_guard_mode': 'enforce'}, claims)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual([call.args for call in scoring.call_args_list],
                         [('First source.', 'First claim.'), ('Second source.', 'Second claim.')])
        self.assertEqual(claims, before)
        self.complete.assert_not_called()

    def test_model_missing_does_not_spend_translation_requests(self):
        payload = deepcopy(self.payload)
        llm.validate_claims(payload, self.hits)
        with patch.object(nli, 'prepare_model', side_effect=nli.NLIError('선택 설치 필요', 'dependencies')):
            result = guard.verify_claims(self.client, self.config | {'claim_guard_mode': 'enforce'}, payload['claims'])
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['code'], 'verification_unavailable')
        self.assertEqual(result['checked'], 0)
        self.complete.assert_not_called()

    def test_overlength_is_error_and_does_not_become_a_factual_rejection(self):
        claims = [{'text': 'A claim.', 'evidence': [{'id': 'A/작업1', 'column': '조치', 'quote': 'A source.'}]}]
        with patch.object(nli, 'prepare_model'), patch.object(nli, 'score', side_effect=nli.NLIError('513 tokens', 'overlength')):
            result = guard.verify_claims(self.client, self.config | {'claim_guard_mode': 'enforce'}, claims)
        self.assertEqual((result['status'], result['code'], result['checked']), ('error', 'verification_overlength', 0))

    def test_repaired_answer_reuses_only_the_independent_source_translation(self):
        first = {'claims': [{'text': 'Replacement completed.', 'evidence': [{'id': self.source['id'], 'column': '조치내용(1)'}]}]}
        second = {'claims': [{'text': 'Replacement is planned.', 'evidence': [{'id': self.source['id'], 'column': '조치내용(1)'}]}]}
        self.complete.side_effect = [response(first), response(second)]
        with patch.object(nli, 'prepare_model'), patch.object(nli, 'score', side_effect=[
                {'passed': False}, {'passed': True}]), patch.object(guard, 'translate_text', side_effect=lambda client, config, text, **kw: text) as translating:
            details = {}
            text, _ = llm.answer(self.client, self.config | {'claim_guard_mode': 'enforce'}, '조치는?', self.hits, details=details)
        self.assertIn('Replacement is planned.', text)
        self.assertEqual(self.complete.call_count, 2)
        self.assertEqual(details['claim_verification']['status'], 'passed')
        sources = [call for call in translating.call_args_list if call.kwargs['kind'] == '원문 셀']
        self.assertEqual(len(sources), 1)

    def test_blank_record_claim_is_checked_without_loading_model(self):
        claims = [{'text': '조치 기록이 없습니다.', 'evidence': [{'id': 'A/작업1', 'column': '조치', 'quote': ''}]}]
        with patch.object(nli, 'prepare_model', side_effect=AssertionError('blank record must not load NLI')):
            result = guard.verify_claims(self.client, self.config | {'claim_guard_mode': 'enforce'}, claims)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['claims'][0]['status'], 'source_blank')
        self.complete.assert_not_called()


if __name__ == '__main__':
    unittest.main()
