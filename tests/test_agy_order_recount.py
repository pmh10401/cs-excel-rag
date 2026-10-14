import unittest
from experiments.agy_order_recount import compare, grade, summarize


class OrderRecountTests(unittest.TestCase):
    def query(self, category='condition', expected=None):
        return {'id': 'q', 'group_id': 'g', 'category': category, 'question': 'fixed',
                'expected_ids': ['a'] if expected is None else expected}

    def test_reranking_improves_top1_without_changing_hit3(self):
        r = compare(self.query(), ['b', 'a'], {'ranked_ids': ['a', 'b'], 'status': 'ok'}, ['b', 'a'])
        self.assertEqual((r['top1_delta'], r['hit3_delta']), (1, 0))

    def test_worsening_and_duplicate_replacement_are_not_hidden(self):
        r = compare(self.query(), ['a', 'b'], {'ranked_ids': ['b', 'a'], 'status': 'ok'}, ['a', 'b'])
        self.assertEqual(r['top1_delta'], -1)
        with self.assertRaises(ValueError):
            compare(self.query(), ['a', 'b'], {'ranked_ids': ['a', 'a'], 'status': 'ok'}, ['a', 'b'])

    def test_missing_expected_stays_in_denominator(self):
        r = compare(self.query(), ['b'], {'ranked_ids': ['b'], 'status': 'ok'}, ['b', 'a'])
        total = summarize([r], 'ALL', 'all')
        self.assertEqual((total['n'], total['top1_same'], total['candidate_missing_k5']), (1, 1, 1))
        self.assertTrue(r['expected_in_full_pool'])

    def test_absent_and_bypass_are_separate(self):
        for category, expected in [('absent', []), ('exact', ['a'])]:
            r = compare(self.query(category, expected), ['a'], {'ranked_ids': ['a'], 'status': 'bypass'}, ['a'])
            self.assertFalse(r['primary'])
            self.assertNotIn('before_top1', r)

    def test_empty_and_multiple_expected_are_explicit(self):
        self.assertFalse(grade([], ['a'])['hit5'])
        self.assertIsNone(grade([], ['a'])['rank'])
        self.assertTrue(grade(['b', 'a'], ['a', 'b'])['top1'])


if __name__ == '__main__':
    unittest.main()
