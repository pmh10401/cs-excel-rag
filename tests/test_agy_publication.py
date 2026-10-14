"""실제 모델 결과 없이 공개 도구의 부분 실행·비공개 로그 제외를 확인합니다."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.agy_cases import build
from experiments.agy_publish import publish
from experiments.agy_report import analyze
from experiments.agy_verify import verify


class PublicationTests(unittest.TestCase):
    def test_interrupted_before_inference_exports_zero_without_raw_messages(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'private';root.mkdir()
            data=build(Path(__file__).resolve().parents[1])
            def save(name,value):(root/name).write_text(json.dumps(value,ensure_ascii=False))
            save('dataset.json',data)
            save('protocol.json',{'dataset_sha256':hashlib.sha256((root/'dataset.json').read_bytes()).hexdigest(),'workers':2})
            save('status.json',{'stage':'interrupted','complete':False,'error':'RuntimeError: DO_NOT_EXPORT_STATUS'})
            (root/'gpu-costs.jsonl').write_text(json.dumps({'wall_seconds':1,'error':'DO_NOT_EXPORT_GPU_COST'})+'\n')
            with (root/'candidates.jsonl').open('w') as out:
                for q in data['queries']:
                    out.write(json.dumps({'query_id':q['id'],'hits':[]})+'\n')
            q=next(q for q in data['queries'] if q['split']=='TEST' and q['category']=='symptom')
            with (root/'rankings.jsonl').open('w') as out:
                for k in (5,10):out.write(json.dumps({'query_id':q['id'],'K':k,'ranked_ids':[],'seconds':0,'status':'bypass'})+'\n')
            (root/'agy-traces').mkdir()
            save('agy-traces/scope-failed-0.metadata.json',{'wall_seconds':1,'error':'RuntimeError: private-message','request':{'secret':'DO_NOT_EXPORT_RAW'}})
            q=next(q for q in data['queries'] if q['split']=='DEV' and q['category']=='symptom')
            call={'wall_seconds':3,'provider_duration':2,'confirmed_model':'gemini-3.8-flash-high',
                'request':{'messages':[{'role':'user','content':'DO_NOT_EXPORT_RAG_PROMPT'}]},'schema':{'type':'object'},
                'usage':{'input_tokens':5,'output_tokens':2,'unexpected':'DO_NOT_EXPORT_USAGE'},'actual_output':{'private':'DO_NOT_EXPORT_OUTPUT'}}
            (root/'answers.jsonl').write_text(json.dumps({'query_id':q['id'],'K':5,'status':'error','seconds':3,'source_ids':[],
                'details':{'answer_text':'DO_NOT_EXPORT_ANSWER','claims':[]},'calls':[call]})+'\n')
            analyze(root)
            destination=Path(folder)/'public';publish(root,destination)
            verify(destination)
            self.assertIn('python verify_results.py',(destination/'README.md').read_text())
            self.assertFalse((destination/'cases.md').read_text().endswith('\n\n'))
            text=''.join(p.read_text(encoding='utf-8-sig') for p in destination.iterdir())
            self.assertNotIn('DO_NOT_EXPORT_RAW',text)
            self.assertNotIn('private-message',text)
            self.assertNotIn('DO_NOT_EXPORT_STATUS',text)
            self.assertNotIn('DO_NOT_EXPORT_GPU_COST',text)
            for marker in ('DO_NOT_EXPORT_RAG_PROMPT','DO_NOT_EXPORT_USAGE','DO_NOT_EXPORT_OUTPUT','DO_NOT_EXPORT_ANSWER'):
                self.assertNotIn(marker,text)
            cost=json.loads((destination/'cost.json').read_text())
            self.assertEqual(cost['attempts_without_completed_flow_row'],1)
            self.assertEqual(cost['provider_seconds_sum'],2)
            self.assertEqual(cost['provider_unmeasured_calls'],1)
            self.assertEqual(cost['input_tokens'],5)
            self.assertIsNone(cost['thinking_tokens'])
            self.assertEqual(cost['linked_calls_without_metadata'],1)
            # 해시를 새로 만들어도 잘못된 통계는 통과하지 않아야 합니다.
            (destination/'paired-deltas.json').write_text('[{"K":10,"paired_delta":1}]')
            manifest=json.loads((destination/'manifest.json').read_text());p=destination/'paired-deltas.json'
            manifest[p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            (destination/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(AssertionError):verify(destination)
            (destination/'paired-deltas.json').write_text('[]')
            changes=destination/'changes.csv';changes.write_text(changes.read_text(encoding='utf-8-sig').splitlines()[0]+'\n')
            for name in ('paired-deltas.json','changes.csv'):
                p=destination/name;manifest[name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            (destination/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(AssertionError):verify(destination)

    def test_complete_flag_with_missing_results_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data=build(Path(__file__).resolve().parents[1])
            (root/'dataset.json').write_text(json.dumps(data))
            (root/'protocol.json').write_text(json.dumps({'dataset_sha256':hashlib.sha256((root/'dataset.json').read_bytes()).hexdigest()}))
            (root/'status.json').write_text(json.dumps({'stage':'model_experiments_finished','complete':True}))
            with self.assertRaises(AssertionError):analyze(root)


if __name__=='__main__':unittest.main()
