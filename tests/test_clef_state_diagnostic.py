from copy import deepcopy
import unittest
from experiments.clef_state_diagnostic import baseline, checked, simple_state, wire
from experiments.clef_state_cases import authored
from experiments.clef_state_report import metrics


def case(source, claim):
    return {'claim': claim, 'cited_cells': [{'id': 'source/접수', 'column': '조치', 'text': source}],
            'label': 'DO_NOT_SEND', 'other_sources': ['UNRELATED_DO_NOT_SEND']}


class StateDiagnosticTests(unittest.TestCase):
    def test_only_actual_cited_cells_and_sentence_enter_model(self):
        request = wire(case('점검했다.', '점검했다.'))
        self.assertEqual(set(request['state']), {'answer_sentence', 'cited_cells'})
        self.assertNotIn('DO_NOT_SEND', str(request))

    def test_structural_accept_is_not_semantic_support(self):
        result = baseline(case('점검했다.', '점검했다.'))
        self.assertTrue(result['accept']); self.assertIsNone(result['semantic'])

    def test_limited_rules_distinguish_plan_nonexecution_and_missing_result(self):
        self.assertEqual(simple_state(case('정비 예정.', '정비를 완료했다.')), 'insufficient')
        self.assertEqual(simple_state(case('교체를 실시하지 않았다.', '교체를 실시했다.')), 'contradicted')
        self.assertEqual(simple_state(case('교체 완료. 결과는 미기재.', '복구됐다.')), 'insufficient')
        self.assertEqual(simple_state(case('교체 완료.', '교체는 완료되지 않았다.')), 'contradicted')

    def test_groups_are_disjoint_and_balanced(self):
        data = authored()
        dev = {c['group_id'] for c in data if c['split'] == 'DEV'}
        test = {c['group_id'] for c in data if c['split'] == 'TEST'}
        self.assertEqual((len(dev), len(test)), (4, 12)); self.assertTrue(dev.isdisjoint(test))
        for group in dev | test:
            self.assertEqual({c['label'] for c in data if c['group_id'] == group}, {'supported', 'contradicted', 'insufficient'})

    def test_incomplete_invalid_or_cpu_decision_is_failure(self):
        valid = {'model': 'clef:flash', 'backend': 'mlx', 'device': 'gpu', 'state_truncated': False,
                 'done_reason': 'decide', 'usage': {'output_tokens': 0}, 'answers': {'support': {
                     'type': 'choice', 'choice': 'supported', 'probabilities': {'supported': .6, 'contradicted': .2, 'insufficient': .2}}}}
        self.assertEqual(checked(valid), 'supported')
        for mutate in (lambda r: r.update(device='cpu'), lambda r: r.update(state_truncated=True),
                       lambda r: r['answers']['support']['probabilities'].update(supported=float('nan'))):
            invalid = deepcopy(valid); mutate(invalid)
            with self.assertRaises(ValueError): checked(invalid)

    def test_failed_check_is_not_additional_error_detection(self):
        row = {'case_id': 'a', 'label': 'contradicted', 'accept': False, 'semantic': None, 'failed': True}
        result = metrics([row], {'a': True})
        self.assertEqual((result['additional_contradiction_detection'], result['contradiction_held'], result['check_failure']), (0, 0, 1))

    def test_normal_failure_hold_and_insufficient_wrong_pass_are_separate(self):
        rows = [{'case_id': 'a', 'label': 'supported', 'accept': False, 'semantic': None, 'failed': True},
                {'case_id': 'b', 'label': 'insufficient', 'accept': True, 'semantic': None, 'failed': False},
                {'case_id': 'c', 'label': 'insufficient', 'accept': False, 'semantic': 'insufficient', 'failed': False}]
        result = metrics(rows, {'a': True, 'b': True, 'c': True})
        self.assertEqual((result['normal_false_hold'], result['normal_hold_including_check_failure'], result['insufficient_wrong_pass']), (0, 1, 1))
        self.assertEqual(result['additional_insufficient_hold'], 1)


if __name__ == '__main__':
    unittest.main()
