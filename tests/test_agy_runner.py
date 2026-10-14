"""실제 모델 없이 종료 출력과 실패 후 재개를 재현합니다."""
import io
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments import agy_breadth as runner


class RunnerTests(unittest.TestCase):
    def test_scope_transport_failure_keeps_calls_without_blocking_other_tasks(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);client=SimpleNamespace(calls=[{'wall_seconds':1,'error':'RuntimeError: unavailable'}])
            with patch.object(runner,'AgyClient',return_value=client),patch.object(runner,'history_for',return_value=[]),\
                 patch.object(runner.llm,'resolve_follow_up',side_effect=RuntimeError('unavailable')):
                result=runner.resolve_scope(root,{'model_stop_at':time.time()+60},{'id':'q','question':'synthetic'}, {})
            self.assertTrue(result['transport_failure']);self.assertEqual(len(result['calls']),1)
            self.assertIsNone(result['resolved'])

    def test_failed_followup_scopes_do_not_skip_normal_answer(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            qs=[{'id':name,'question':'synthetic','split':'DEV','category':kind} for name,kind in [('f1','followup'),('f2','followup'),('n','symptom')]]
            for q in qs:runner.append(root/'rankings.jsonl',{'query_id':q['id'],'K':5})
            def scope(root,p,q,frozen):return {'query_id':q['id'],'resolved':None,'error':'RuntimeError: unavailable','calls':[{'error':'RuntimeError'}]}
            def answer(root,p,q,k,frozen,ranking,scoped):
                return {'query_id':q['id'],'K':k,'status':'error' if scoped else 'ok','calls':[],'transport_failure':False}
            with patch.object(runner,'resolve_scope',side_effect=scope),patch.object(runner,'answer_one',side_effect=answer):
                runner.run_answers(root,{'model_stop_at':time.time()+60,'workers':2},{'queries':qs,'answer_ids':[]},[{'query_id':q['id']} for q in qs])
            results=runner.rows(root/'answers.jsonl')
            self.assertEqual(len(results),3)
            self.assertEqual(next(r['status'] for r in results if r['query_id']=='n'),'ok')
            self.assertEqual(len(runner.rows(root/'scopes.jsonl')),2)

    def test_exited_worker_stdout_is_drained_to_eof(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);runner.save(root/'protocol.json',{})
            read_fd,write_fd=os.pipe()
            for k in runner.KS:
                os.write(write_fd,('RESULT:'+json.dumps({'key':f'q::{k}'})+'\n').encode())
            os.close(write_fd)
            child=SimpleNamespace(stdin=io.StringIO(),stdout=os.fdopen(read_fd),pid=os.getpid(),poll=lambda:0,returncode=0)
            def record(root,q,hits,k,raw,request):
                runner.append(root/'rankings.jsonl',{'query_id':q['id'],'K':k})
            with patch.object(runner,'wait_gpu'),patch.object(runner,'wire',side_effect=lambda root,q,hits,k:({'K':k},{})),\
                 patch.object(runner,'record_rank',side_effect=record),patch.object(runner.subprocess,'Popen',return_value=child):
                runner.rank_all(root,{'model_stop_at':time.time()+60},{'queries':[{'id':'q','category':'symptom'}]},[{'query_id':'q','hits':[]}])
            self.assertEqual(len(runner.rows(root/'rankings.jsonl')),3)
            self.assertEqual(len(runner.rows(root/'gpu-results.jsonl')),3)
            self.assertEqual(runner.rows(root/'gpu-costs.jsonl')[0]['results'],3)

    def test_resume_does_not_label_two_transport_failures_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);runner.save(root/'dataset.json',{})
            for _ in range(792):runner.append(root/'rankings.jsonl',{})
            for i in range(280):runner.append(root/'answers.jsonl',{'transport_failure':i>=278})
            with patch.object(runner,'verify',return_value={}),patch.object(runner,'freeze_candidates',return_value=[]),\
                 patch.object(runner,'rank_all'),patch.object(runner,'run_answers'):
                runner.run(root)
            status=runner.read(root/'status.json')
            self.assertFalse(status['complete'])
            self.assertEqual(status['transport_failures'],2)
            self.assertEqual(status['stage'],'model_experiments_finished_with_errors')


if __name__=='__main__':unittest.main()
