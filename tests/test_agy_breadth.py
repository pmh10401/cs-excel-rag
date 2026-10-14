"""격리 어댑터·라벨 분리·고정 표본의 오프라인 검사. 실제 AGY/GPU 평가는 별도입니다."""
from collections import Counter
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.agy_cases import build, KINDS
from experiments.agy_client import AgyClient, decode_events, MODEL


def events(model=MODEL, step='agent_response'):
    return '\n'.join(json.dumps(x) for x in [
        {'event':'init','init':{'model':model}},
        {'event':'step_update','step_update':{'step_type':step}},
        {'event':'result','result':{'status':'SUCCESS','structured_output':{'claims':[]},'duration_seconds':1}}])


class AgyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=build(Path(__file__).resolve().parents[1])

    def test_split_groups_types_and_fixed_answer_subset(self):
        d=self.data
        self.assertEqual(len(d['records']),2998)
        for split,size,per_type in [('DEV',40,4),('TEST',200,20)]:
            qs=[q for q in d['queries'] if q['split']==split]
            self.assertEqual(len(qs),size)
            self.assertEqual(Counter(q['category'] for q in qs),Counter({k:per_type for k in KINDS}))
        groups={s:{q['group_id'] for q in d['queries'] if q['split']==s} for s in ('DEV','TEST','REGRESSION')}
        self.assertTrue(groups['DEV'].isdisjoint(groups['TEST']))
        answer=[q for q in d['queries'] if q['id'] in d['answer_ids']]
        self.assertEqual(len(answer),80)
        self.assertEqual(Counter(q['category'] for q in answer),Counter({k:8 for k in KINDS}))
        self.assertEqual(len({q['group_id'] for q in answer}),20)
        expected={r['_record_id'] for r in d['records']}
        self.assertTrue(all(set(q['expected_ids'])<=expected for q in d['queries']))

    def test_decode_model_and_no_tool_constraint(self):
        self.assertEqual(decode_events(events())['structured_output'],{'claims':[]})
        for data in (events(model='other'),events(step='run_command'),events(step='tool'),'not json'):
            with self.assertRaises(ValueError):decode_events(data)

    def test_internal_finalization_is_not_external_tool_access(self):
        value=events().splitlines()
        value.insert(2,json.dumps({'event':'step_update','step_update':{'step_type':'tool','tool_name':'finish'}}))
        self.assertEqual(decode_events('\n'.join(value))['status'],'SUCCESS')

    def test_unsupported_controls_not_silently_promised(self):
        with tempfile.TemporaryDirectory() as directory:
            c=AgyClient(directory,'q')
            with self.assertRaises(ValueError):c.create(model=MODEL,messages=[],temperature=0,extra_body={'reasoning_effort':'high'})

    def test_deadline_rechecked_before_each_new_cli_call(self):
        with tempfile.TemporaryDirectory() as directory,patch('experiments.agy_client.subprocess.Popen') as start:
            c=AgyClient(directory,'q',timeout=600,deadline=1)
            with self.assertRaises(TimeoutError):c.create(model=MODEL,messages=[],extra_body={'reasoning_effort':'high'})
            start.assert_not_called()
            self.assertEqual(c.calls,[])

    def test_adapter_uses_structured_contract_and_retains_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            p=SimpleNamespace(pid=100000,returncode=0,communicate=lambda **kw:(events(),''))
            with patch('experiments.agy_client.subprocess.Popen',return_value=p) as start:
                c=AgyClient(directory,'q');r=c.create(model=MODEL,messages=[{'role':'user','content':'synthetic'}],extra_body={'reasoning_effort':'high'},max_tokens=100)
            self.assertEqual(json.loads(r.choices[0].message.content),{'claims':[]})
            self.assertEqual(c.calls[0]['unmapped_options'],{'max_tokens':100})
            self.assertIn('/usr/bin/sandbox-exec',start.call_args.args[0])
            self.assertNotIn('100',start.call_args.args[0])
            self.assertEqual(c.calls[0]['exit'],0)

if __name__=='__main__':unittest.main()
