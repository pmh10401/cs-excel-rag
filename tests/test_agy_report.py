import unittest
import json
from pathlib import Path
import tempfile
from experiments.agy_report import answer_diagnostic,cohort_calls,delta_interval,paired_inputs,rank_grade

class ReportTests(unittest.TestCase):
    def test_input_pairing_includes_failed_attempts_but_not_missing_calls(self):
        rows=[{'query_id':'q','split':'TEST','K':k,'calls':[{'rag_messages_sha256':h}],'status':'error'} for k,h in [(5,'same'),(10,'same'),(20,'different')]]
        rows.append({'query_id':'unreached','split':'TEST','K':5,'calls':[]})
        result=paired_inputs(rows)
        self.assertEqual([(r['paired_cli_attempts'],r['identical_initial_rag_messages']) for r in result],[(1,1),(1,0)])

    def test_cluster_count_is_not_question_count(self):
        pairs=[('a',1),('a',1),('b',-1)]
        result=delta_interval(pairs)
        self.assertEqual(result['clusters'],2)
        self.assertEqual(result['questions'],3)
        self.assertAlmostEqual(result['paired_delta'],1/3)
        self.assertEqual(result,delta_interval(pairs))

    def test_paraphrase_and_missing_action_not_declared_correct(self):
        q={'category':'symptom','expected_ids':['a'],'contradicting_ids':[]}
        r={'status':'ok','details':{'claims':[{'text':'조정한 뒤 안정적으로 동작했다.','evidence':[{'id':'a/접수','column':'조치'}]}]}}
        self.assertIsNone(answer_diagnostic(q,r,{'a':{'조치':'연결부를 재조립했다. 재시험 후 정상화되었다.'}})['literal_complete'])
        self.assertIsNone(answer_diagnostic(q,r,{'a':{'조치':''}})['literal_complete'])

    def test_action_text_needs_actual_action_cell_address(self):
        q={'category':'symptom','expected_ids':['a'],'contradicting_ids':[]}
        r={'status':'ok','details':{'claims':[{'text':'부품을 교체했다.','evidence':[{'id':'a/접수','column':'현상'}]}]}}
        self.assertIsNone(answer_diagnostic(q,r,{'a':{'조치':'부품을 교체했다.'}})['literal_complete'])
        r['details']['claims'][0]['evidence'][0]['column']='조치'
        self.assertTrue(answer_diagnostic(q,r,{'a':{'조치':'부품을 교체했다.'}})['literal_complete'])

    def test_rank_membership_does_not_confuse_top1_and_hitK(self):
        q={'expected_ids':['a'],'contradicting_ids':['b']}
        r=rank_grade(q,['b','a'])
        self.assertFalse(r['top1']);self.assertTrue(r['hit3']);self.assertTrue(r['hitK']);self.assertTrue(r['contradiction_top1'])

    def test_scope_transport_failure_without_result_row_is_costed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'agy-traces').mkdir()
            for key in ('scope-DEV-followup-00-0','answer-TEST-symptom-00-5-0','dev-probe-0'):
                (root/'agy-traces'/f'{key}.metadata.json').write_text(json.dumps({'wall_seconds':3}))
            self.assertEqual(len(cohort_calls(root)),2)

    def test_linked_call_survives_missing_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            calls=cohort_calls(Path(folder),[{'query_id':'q','K':5,'calls':[{'wall_seconds':3,'error':'RuntimeError: unavailable'}]}])
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0]['wall_seconds'],3)
            self.assertTrue(calls[0]['metadata_missing'])

    def test_same_call_id_merges_missing_field_without_double_cost(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'agy-traces').mkdir()
            (root/'agy-traces/answer-q-5-0.metadata.json').write_text(json.dumps({'wall_seconds':3}))
            flow={'query_id':'q','K':5,'calls':[{'wall_seconds':3,'provider_duration':None}]}
            self.assertEqual(len(cohort_calls(root,[flow])),1)
            flow['calls'][0]['wall_seconds']=4
            with self.assertRaises(AssertionError):cohort_calls(root,[flow])

if __name__=='__main__':unittest.main()
