import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.agy_verify import check_manifest,check_rank,interval


class VerifyTests(unittest.TestCase):
    def test_cluster_interval_known_endpoints_and_denominator(self):
        r=interval([('a',1),('a',1),('b',-1)])
        self.assertEqual((r['low'],r['high'],r['clusters'],r['questions']),(-1,1,2,3))
        self.assertAlmostEqual(r['paired_delta'],1/3)

    def test_rank_change_preserves_candidates_but_duplicate_and_replacement_fail(self):
        q={'expected_ids':['a']}
        check_rank({'K':3,'ranked_ids':['c','a','b']},q,['a','b','c'])
        for ids in (['a','a','b'],['a','b','other']):
            with self.assertRaises(AssertionError):check_rank({'K':3,'ranked_ids':ids},q,['a','b','c'])

    def test_manifest_rejects_modified_missing_and_extra_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'result.json';p.write_text('{}')
            (root/'manifest.json').write_text(json.dumps({'result.json':{'bytes':2,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}))
            check_manifest(root)
            p.write_text('[]')
            with self.assertRaises(AssertionError):check_manifest(root)
            p.unlink()
            with self.assertRaises(AssertionError):check_manifest(root)
            p.write_text('{}');(root/'unexpected.txt').write_text('extra')
            with self.assertRaises(AssertionError):check_manifest(root)


if __name__=='__main__':unittest.main()
