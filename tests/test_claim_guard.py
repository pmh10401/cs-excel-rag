"""claim_guard 합성 탐침의 자료·범위·오류 분리·재개를 모델 다운로드 없이 확인합니다."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import claim_guard as probe
from ragkit.retrieval_evaluation import atomic_report

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'experiments' / 'claim_guard_cases.json'


def fake_args(output, **overrides):
    values = dict(backend='baseline', dataset=DATASET, output=Path(output), device='cpu',
                  threshold=0.8, local_files_only=False)
    values.update(overrides)
    return Namespace(**values)


def unsupported_scores(premise, hypothesis):
    return {'nli_label': 'unsupported', 'probabilities': {'unsupported': 0.9, 'supported': 0.1}}


class ClaimGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = probe.load_cases(DATASET)

    def test_dataset_invariants(self):
        cases = self.data['cases']
        self.assertEqual(len(cases), 36)
        self.assertTrue(self.data['segmentation_note'])
        ids = [case['id'] for case in cases]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(probe.REQUIRED_CATEGORIES <= {case['category'] for case in cases})
        self.assertEqual({case['expected_semantic'] for case in cases}, set(probe.SEMANTIC))
        chinese = 0
        for case in cases:
            self.assertRegex(case['id'], r'^[A-Za-z0-9_-]+$')
            parents = {}
            for source in case['sources']:
                self.assertTrue(source['id'].startswith('누적28상세:'))
                receipt = source['cells']['CS NO']
                parents.setdefault(receipt, set()).add(source['id'].rsplit('/', 1)[0])
                chinese += sum(any('\u4e00' <= ch <= '\u9fff' for ch in value)
                               for value in source['cells'].values())
            for receipt, values in parents.items():
                count = sum(1 for source in case['sources'] if source['cells']['CS NO'] == receipt)
                self.assertEqual(len(values), count)
            for ref in case['citations']:
                source = next(item for item in case['sources'] if item['id'] == ref['id'])
                self.assertIn(ref['column'], source['cells'])
                self.assertIn(ref['column'], source['cells_en'])
        self.assertGreaterEqual(chinese, 2)
        part04 = next(item for item in cases if item['id'] == 'CG-PART-04')
        self.assertEqual([ref['id'] for ref in part04['citations']], ['누적28상세:part04a/접수'])
        zh = next(item for item in cases if item['id'] == 'CG-ZH-02')
        self.assertEqual(zh['expected_semantic'], 'supported')
        self.assertNotIn('小时', zh['sources'][0]['cells_en']['현상'])
        self.assertIn('小时', zh['sources'][0]['cells']['현상'])

    def test_source_scope_excludes_question_and_other_detail(self):
        case = next(item for item in self.data['cases'] if item['id'] == 'CG-PART-02')
        premise, hypothesis = probe.cited_pair(case, language='ko')
        self.assertIn('펌프 씰을 교체함', premise)
        self.assertNotIn('바이패스 배관 밸브', premise)
        self.assertNotIn(case['question']['ko'], premise)
        self.assertNotIn('조치:', premise)
        self.assertEqual(hypothesis, case['claim']['ko'])
        english, _ = probe.cited_pair(case, language='en')
        self.assertIn('pump seal', english.lower())
        self.assertNotIn('bypass', english.lower())
        self.assertNotIn('조치', english)
        part04 = next(item for item in self.data['cases'] if item['id'] == 'CG-PART-04')
        part04_en, _ = probe.cited_pair(part04, language='en')
        self.assertNotIn('pipe valve', part04_en.lower())

    def test_empty_evidence_stops_before_model(self):
        case = next(item for item in self.data['cases'] if item['id'] == 'CG-MISS-03')
        calls = []

        def model_fn(premise, hypothesis):
            calls.append((premise, hypothesis))
            raise AssertionError('빈 근거를 모델에 넘겼습니다.')

        result = probe.evaluate_case(case, backend='mdeberta', threshold=0.8, model_fn=model_fn)
        self.assertEqual(calls, [])
        self.assertEqual(result['model']['status'], 'empty_evidence')
        self.assertFalse(result['model']['passed'])
        self.assertEqual(result['baseline']['status'], 'rejected')
        self.assertEqual(result['compared_pair']['premise'], '')

    def test_blank_cited_value_stops_before_model(self):
        case = {
            'id': 'BLANK-01', 'category': 'missing_support', 'expected_semantic': 'not_mentioned',
            'question': {'ko': '조치는?', 'en': 'Action?'},
            'sources': [{'id': '누적28상세:blank01/접수', 'kind': '접수',
                         'cells': {'CS NO': 'CS-BLANK', '현상': '펌프 누수', '조치': '   '},
                         'cells_en': {'현상': 'Pump leak', '조치': '  '}}],
            'citations': [{'id': '누적28상세:blank01/접수', 'column': '조치'}],
            'claim': {'ko': '씰을 교체했습니다.', 'en': 'The seal was replaced.'},
            'rationale': '공백만 있는 인용 값',
        }
        calls = []

        def model_fn(premise, hypothesis):
            calls.append((premise, hypothesis))
            raise AssertionError('공백 전제를 모델에 넘겼습니다.')

        result = probe.evaluate_case(case, backend='minicheck', threshold=0.8, model_fn=model_fn)
        self.assertEqual(calls, [])
        self.assertEqual(result['model']['status'], 'empty_evidence')
        self.assertTrue(probe.empty_model_evidence(case, 'en'))

    def test_rule_deepcopy_and_selected_supported_case(self):
        case = next(item for item in self.data['cases'] if item['id'] == 'CG-PLAN-01')
        before = deepcopy(case)
        result = probe.rule_check(case)
        self.assertEqual(case, before)
        self.assertEqual(result['status'], 'ok')
        self.assertTrue(result['passed'])
        self.assertFalse(any('quote' in ref for ref in case['citations']))

    def test_invalid_probabilities_and_overlength_are_errors(self):
        with self.assertRaises(probe.ClassifierError):
            probe.decide_from_probabilities(
                {'entailment': float('nan'), 'neutral': 0.0, 'contradiction': 0.0},
                support_key='entailment', threshold=0.8, nli_label='entailment')
        with self.assertRaises(probe.ClassifierError):
            probe.nli_label_map({0: 'LABEL_0', 1: 'LABEL_1', 2: 'LABEL_2'})
        with self.assertRaises(probe.ClassifierError):
            probe.nli_label_map({0: 'entailment', 1: 'neutral', 3: 'contradiction'})
        mapping = probe.nli_label_map({2: 'entailment', 0: 'neutral', 1: 'contradiction'})
        self.assertEqual(set(mapping.values()), {0, 1, 2})
        with self.assertRaises(probe.OverlengthError):
            probe.reject_if_overlong(2049, 2048)
        case = next(item for item in self.data['cases'] if item['id'] == 'CG-PLAN-02')

        def nan_fn(premise, hypothesis):
            return {'nli_label': 'contradiction',
                    'probabilities': {'unsupported': float('nan'), 'supported': 1.0}}

        def long_fn(premise, hypothesis):
            raise probe.OverlengthError('입력 길이 3000가 한도 512를 넘습니다.')

        nan_result = probe.evaluate_case(case, backend='minicheck', threshold=0.8, model_fn=nan_fn)
        long_result = probe.evaluate_case(case, backend='mdeberta', threshold=0.8, model_fn=long_fn)
        self.assertEqual(nan_result['model']['status'], 'classifier_error')
        self.assertEqual(long_result['model']['status'], 'overlength')
        self.assertEqual(nan_result['combined']['status'], 'error')
        self.assertFalse(nan_result['combined']['passed'])

    def test_minicheck_two_logit_convention_and_threshold(self):
        self.assertEqual((probe.MINICHECK_TOKEN_UNSUPPORTED, probe.MINICHECK_TOKEN_SUPPORTED), (3, 209))
        probs = probe.softmax([0.0, 0.0])
        self.assertAlmostEqual(probs[0] + probs[1], 1.0)
        decided = probe.decide_from_probabilities(
            {'unsupported': 0.3, 'supported': 0.7}, support_key='supported',
            threshold=0.8, nli_label='supported')
        self.assertTrue(decided['class_pass'])
        self.assertFalse(decided['threshold_pass'])
        self.assertEqual(probe.sequence_limit('minicheck'), 2048)
        self.assertEqual((probe.CPU_THREADS, probe.TORCH_DTYPE), (4, 'float32'))

    def test_aggregation_keeps_errors_out_of_correct_rejects(self):
        rows = [
            {'id': 'ok-s', 'category': 'x', 'expected_semantic': 'supported',
             'baseline': {'status': 'ok', 'passed': True, 'seconds': 0.1},
             'model': {'status': 'ok', 'passed': True, 'seconds': 0.2, 'nli_label': 'entailment'},
             'combined': {'status': 'ok', 'passed': True}, 'seconds': 0.3},
            {'id': 'err-s', 'category': 'x', 'expected_semantic': 'supported',
             'baseline': {'status': 'ok', 'passed': True, 'seconds': 0.1},
             'model': {'status': 'overlength', 'passed': False, 'seconds': 0.2, 'review': True},
             'combined': {'status': 'error', 'passed': False}, 'seconds': 0.3},
            {'id': 'err-u', 'category': 'x', 'expected_semantic': 'contradicted',
             'baseline': {'status': 'rejected', 'passed': False, 'seconds': 0.1},
             'model': {'status': 'classifier_error', 'passed': False, 'seconds': 0.2, 'review': True},
             'combined': {'status': 'error', 'passed': False}, 'seconds': 0.3},
            {'id': 'fa', 'category': 'x', 'expected_semantic': 'not_mentioned',
             'baseline': {'status': 'ok', 'passed': True, 'seconds': 0.1},
             'model': {'status': 'ok', 'passed': True, 'seconds': 0.2, 'nli_label': 'entailment'},
             'combined': {'status': 'ok', 'passed': True}, 'seconds': 0.3},
            {'id': 'fr', 'category': 'x', 'expected_semantic': 'supported',
             'baseline': {'status': 'rejected', 'passed': False, 'seconds': 0.1},
             'model': {'status': 'ok', 'passed': False, 'seconds': 0.2, 'nli_label': 'neutral'},
             'combined': {'status': 'rejected', 'passed': False}, 'seconds': 0.3},
        ]
        stats = probe.aggregate(rows, backend='mdeberta', total=6)
        self.assertEqual(stats['completed'], 5)
        self.assertEqual(stats['missing'], 1)
        self.assertTrue(stats['complete_with_errors'])
        self.assertFalse(stats['complete'])
        model = stats['summary']['model']
        self.assertEqual(model['error_ids'], ['err-s', 'err-u'])
        self.assertEqual(model['false_accept_ids'], ['fa'])
        self.assertEqual(model['false_reject_ids'], ['fr'])
        self.assertNotIn('err-s', model['false_reject_ids'])
        self.assertNotIn('err-u', model['false_accept_ids'])

    def test_resume_and_reject_changed_conditions(self):
        case = next(item for item in self.data['cases'] if item['id'] == 'CG-PLAN-01')
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            first = probe.run(fake_args(output))
            self.assertTrue(first['complete'])
            self.assertEqual(first['total'], 36)
            calls = {'n': 0}
            original = probe.evaluate_case

            def counting(case_arg, **kwargs):
                calls['n'] += 1
                return original(case_arg, **kwargs)

            probe.evaluate_case = counting
            try:
                second = probe.run(fake_args(output))
            finally:
                probe.evaluate_case = original
            self.assertEqual(calls['n'], 0)
            self.assertEqual(second['results'][0]['id'], case['id'])
            with self.assertRaises(ValueError):
                probe.run(fake_args(output, threshold=0.81))
            changed = deepcopy(case)
            changed['claim'] = dict(case['claim'], ko='바뀐 주장입니다.')
            with self.assertRaises(ValueError):
                probe.read_checkpoint(output / 'checkpoints' / f'{case["id"]}.json',
                                      first['protocol_sha256'], changed)

    def test_complete_checkpoint_resume_skips_model_load(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            args = fake_args(output, backend='minicheck')
            first = probe.run(args, model_fn=unsupported_scores)
            self.assertTrue(first['complete'])
            loads = {'n': 0}

            def fake_load(*a, **k):
                loads['n'] += 1
                raise AssertionError('완료된 재개에서 모델을 로드하면 안 됩니다.')

            original = probe.load_backend
            probe.load_backend = fake_load
            try:
                second = probe.run(fake_args(output, backend='minicheck'))
            finally:
                probe.load_backend = original
            self.assertEqual(loads['n'], 0)
            self.assertEqual(len(second['results']), 36)
            self.assertTrue(second['complete'])

    def test_changed_conditions_before_completion(self):
        case = self.data['cases'][0]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            args = fake_args(output, backend='minicheck')
            probe.ensure_protocol(output, probe.freeze_conditions(args, self.data))
            first = probe.evaluate_case(case, backend='minicheck', threshold=0.8,
                                        model_fn=unsupported_scores)
            path = output / 'checkpoints' / f'{case["id"]}.json'
            atomic_report(path, {
                'protocol_sha256': probe.digest(output / 'protocol.json'),
                'case_sha256': probe.case_digest(case),
                'case_id': case['id'],
                'result': first,
            })
            with self.assertRaises(ValueError):
                probe.run(fake_args(output, backend='minicheck', threshold=0.81),
                          model_fn=unsupported_scores)
            self.assertTrue(path.exists())
            self.assertEqual(probe.read_checkpoint(path, probe.digest(output / 'protocol.json'), case)['id'],
                             case['id'])

    def test_final_verification_failure_never_certifies_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)

            def fail_verification(*args):
                saved = json.loads((output / 'results.json').read_text())
                self.assertEqual(saved['completed'], 36)
                self.assertFalse(saved['complete'])
                raise OSError('동결한 코드 파일을 읽을 수 없습니다.')

            with patch.object(probe, 'verify_frozen_inputs', side_effect=fail_verification):
                report = probe.run(fake_args(output))
            self.assertEqual(report['status'], 'verification_failed')
            self.assertFalse(report['complete'])
            self.assertEqual(len(list((output / 'checkpoints').glob('*.json'))), 36)

    def test_interrupted_resume_keeps_finished_error(self):
        case = self.data['cases'][0]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            args = fake_args(output, backend='minicheck')

            def once(premise, hypothesis):
                raise probe.ClassifierError('가짜 분류 실패')

            probe.ensure_protocol(output, probe.freeze_conditions(args, self.data))
            first = probe.evaluate_case(case, backend='minicheck', threshold=0.8, model_fn=once)
            atomic_report(output / 'checkpoints' / f'{case["id"]}.json', {
                'protocol_sha256': probe.digest(output / 'protocol.json'),
                'case_sha256': probe.case_digest(case),
                'case_id': case['id'],
                'result': first,
            })
            calls = []

            def later(premise, hypothesis):
                calls.append((premise, hypothesis))
                return unsupported_scores(premise, hypothesis)

            report = probe.run(args, model_fn=later)
            first_pair = probe.cited_pair(case, language='en')
            self.assertEqual(report['results'][0]['id'], case['id'])
            self.assertEqual(report['results'][0]['model']['status'], 'classifier_error')
            self.assertNotIn(first_pair, calls)
            self.assertGreaterEqual(len(calls), 1)
            self.assertTrue(report['complete'])
            self.assertTrue(report['complete_with_errors'])
            self.assertGreaterEqual(report['errors'], 1)


if __name__ == '__main__':
    unittest.main()
