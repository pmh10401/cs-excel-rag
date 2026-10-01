"""선택 모델의 입력 한도·라벨·확률·캐시를 다운로드 없이 검사합니다."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import itertools
import math
import multiprocessing
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ragkit import nli


class Tensor:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values), len(values[0])) if values and isinstance(values[0], list) else (len(values),)
        self.ndim = len(self.shape)
    def __getitem__(self, index):
        value = self.values[index]
        return Tensor(value) if isinstance(value, list) else SimpleNamespace(item=lambda: value)
    def float(self):
        return self
    def to(self, device):
        return self


def fake_runtime(logits):
    torch = ModuleType('torch')
    torch.inference_mode = nullcontext
    torch.isfinite = lambda tensor: SimpleNamespace(all=lambda: all(math.isfinite(v) for row in tensor.values for v in row))
    nn = ModuleType('torch.nn')
    functional = ModuleType('torch.nn.functional')
    def softmax(tensor, dim):
        row = tensor.values[0]
        values = [math.exp(v - max(row)) for v in row]
        return Tensor([[value / sum(values) for value in values]])
    functional.softmax = softmax
    nn.functional = functional
    torch.nn = nn
    def tokenizer(*texts, **options):
        return {'input_ids': Tensor([[1, 2, 3]]) if options.get('return_tensors') else [1, 2, 3]}
    token = Mock(side_effect=tokenizer)
    token.model_max_length = 512
    model = Mock(return_value=SimpleNamespace(logits=Tensor(logits)))
    model.config = SimpleNamespace(id2label={0: 'entailment', 1: 'neutral', 2: 'contradiction'}, max_position_embeddings=512)
    return {'torch': torch, 'torch.nn': nn, 'torch.nn.functional': functional}, token, model


def ready_worker(connection):
    connection.send({'ok': True, 'value': {'loaded': True, 'device': 'cpu', 'limit': 512}})
    try:
        while True:
            connection.recv()
            connection.send({'ok': True, 'value': {'passed': True}})
    except (EOFError, OSError):
        pass


def stalled_load(connection):
    time.sleep(30)


def stalled_score(connection):
    connection.send({'ok': True, 'value': {'loaded': True, 'device': 'cpu', 'limit': 512}})
    connection.recv()
    time.sleep(30)


class NLITests(unittest.TestCase):
    def test_empty_input_is_rejected_without_loading_model(self):
        with patch.object(nli, 'prepare_model') as preparing, self.assertRaises(nli.NLIError) as error:
            nli.score(' ', 'claim')
        self.assertEqual(error.exception.code, 'empty')
        preparing.assert_not_called()

    def test_module_and_status_do_not_import_torch(self):
        result = subprocess.run([sys.executable, '-c',
            'import sys; from ragkit import web, nli; nli.status(); assert "torch" not in sys.modules; '
            'assert "transformers" not in sys.modules; assert "faiss" not in sys.modules'],
            capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_label_permutations_and_bad_labels(self):
        names = sorted(nli.LABELS)
        for order in itertools.permutations(names):
            mapping = nli.label_map(dict(enumerate(order)))
            self.assertEqual({mapping[name] for name in names}, {0, 1, 2})
        for labels in (None, {}, {0: 'LABEL_0'}, {0: 'entailment', 1: 'entailment', 2: 'contradiction'},
                       {0: 'entailment', 1: 'neutral', 3: 'contradiction'}):
            with self.subTest(labels=labels), self.assertRaises(nli.NLIError) as error:
                nli.label_map(labels)
            self.assertEqual(error.exception.code, 'labels')

    def test_sequence_limit_uses_smaller_real_limit(self):
        token = SimpleNamespace(model_max_length=1024)
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512))
        self.assertEqual(nli.sequence_limit(token, model), 512)
        token.model_max_length = 256
        self.assertEqual(nli.sequence_limit(token, model), 256)
        with self.assertRaises(nli.NLIError):
            nli.sequence_limit(SimpleNamespace(), SimpleNamespace(config=SimpleNamespace()))

    def test_overlength_is_not_truncated_or_sent_to_model(self):
        modules, token, model = fake_runtime([[4, 0, -1]])
        token.side_effect = lambda *args, **kwargs: {'input_ids': list(range(513))}
        with patch.dict(sys.modules, modules), self.assertRaises(nli.NLIError) as error:
            nli.score_pair('original premise', 'original claim', tokenizer=token, model=model)
        self.assertEqual(error.exception.code, 'overlength')
        model.assert_not_called()
        self.assertEqual(token.call_args.args, ('original premise', 'original claim'))
        self.assertFalse(token.call_args.kwargs['truncation'])

    def test_probability_threshold_not_argmax(self):
        modules, token, model = fake_runtime([[1, 0.9, 0.8]])
        with patch.dict(sys.modules, modules):
            result = nli.score_pair('premise', 'claim', tokenizer=token, model=model)
        self.assertEqual(result['label'], 'entailment')
        self.assertFalse(result['passed'])
        self.assertEqual(result['threshold'], 0.8)
        modules, token, model = fake_runtime([[4, 0, -1]])
        with patch.dict(sys.modules, modules):
            self.assertTrue(nli.score_pair('premise', 'claim', tokenizer=token, model=model)['passed'])

    def test_nonfinite_and_bad_shape_are_errors(self):
        for logits in ([[float('nan'), 0, 1]], [[float('inf'), 0, 1]], [[1, 2]], [[1, 2, 3], [1, 2, 3]]):
            modules, token, model = fake_runtime(logits)
            with self.subTest(logits=logits), patch.dict(sys.modules, modules), self.assertRaises(nli.NLIError) as error:
                nli.score_pair('premise', 'claim', tokenizer=token, model=model)
            self.assertEqual(error.exception.code, 'classifier')

    def test_concurrent_preparation_loads_once_and_never_downloads(self):
        context = multiprocessing.get_context('spawn')
        process = Mock(side_effect=context.Process)
        fake_context = SimpleNamespace(Pipe=context.Pipe, Process=process)
        with patch.object(nli, '_worker', None), patch.object(nli, '_worker_main', ready_worker), \
                patch.object(nli.multiprocessing, 'get_context', return_value=fake_context), patch.object(nli, 'missing_dependencies', return_value=[]), \
                patch.object(nli, 'cache_present', side_effect=AssertionError('no cache or network lookup during inference')):
            try:
                with ThreadPoolExecutor(max_workers=8) as pool:
                    result = list(pool.map(lambda _: nli.prepare_model(download=False), range(16)))
                self.assertEqual(process.call_count, 1)
                self.assertTrue(all(value['loaded'] and value['device'] == 'cpu' for value in result))
            finally:
                nli._stop_worker()

    def test_stalled_load_is_killed_and_later_request_can_run(self):
        with patch.object(nli, '_worker', None), patch.object(nli, 'missing_dependencies', return_value=[]):
            try:
                with patch.object(nli, '_worker_main', stalled_load), self.assertRaises(nli.NLIError) as error:
                    started = time.monotonic()
                    nli.prepare_model(deadline=started + .1)
                self.assertEqual(error.exception.code, 'verification_timeout')
                self.assertLess(time.monotonic() - started, 3)
                self.assertIsNone(nli._worker)
                with patch.object(nli, '_worker_main', ready_worker):
                    self.assertTrue(nli.prepare_model()['loaded'])
                    self.assertTrue(nli.score('source', 'claim')['passed'])
            finally:
                nli._stop_worker()

    def test_stalled_inference_is_killed_and_worker_does_not_hold_lock(self):
        with patch.object(nli, '_worker', None), patch.object(nli, '_worker_main', stalled_score), \
                patch.object(nli, 'missing_dependencies', return_value=[]):
            try:
                nli.prepare_model()
                started = time.monotonic()
                with self.assertRaises(nli.NLIError) as error:
                    nli.score('source', 'claim', deadline=started + .1)
                self.assertEqual(error.exception.code, 'verification_timeout')
                self.assertLess(time.monotonic() - started, 3)
                self.assertIsNone(nli._worker)
            finally:
                nli._stop_worker()

    def test_download_requires_explicit_preparation(self):
        hub = ModuleType('huggingface_hub')
        hub.snapshot_download = Mock()
        with patch.dict(sys.modules, {'huggingface_hub': hub}), patch.object(nli, '_request', return_value={'loaded': True}), \
                patch.object(nli, 'cache_present', return_value=False), patch.object(nli, 'missing_dependencies', return_value=[]):
            nli.prepare_model(download=True)
        self.assertEqual(hub.snapshot_download.call_count, 1)
        self.assertEqual(hub.snapshot_download.call_args.args, (nli.MODEL_ID,))
        self.assertEqual(hub.snapshot_download.call_args.kwargs['revision'], nli.MODEL_REV)
        self.assertNotIn('pytorch_model.bin', hub.snapshot_download.call_args.kwargs['allow_patterns'])


if __name__ == '__main__':
    unittest.main()
