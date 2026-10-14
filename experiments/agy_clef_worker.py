"""독립 평가용 상주 Clef MLX GPU. 운영 로더의 엄격한 길이 계약을 그대로 사용합니다."""
import json
import hashlib
from pathlib import Path
import resource
import sys
import time


def main():
    import mlx.core as mx
    from experiments.clef_mlx_server import validate_checkpoint, MODEL, REPO, REVISION
    spec=json.load(sys.stdin);path=Path(spec['path']);validate_checkpoint(path)
    if not mx.metal.is_available():raise RuntimeError('Metal unavailable; CPU fallback prohibited')
    if json.loads((path/'config.json').read_text()).get('quantization',{}).get('bits')!=8:raise ValueError('Expected8bit checkpoint')
    mx.set_default_device(mx.gpu)
    if mx.default_device()!=mx.gpu:raise RuntimeError('GPU was not selected')
    sys.path.insert(0,str(path));import clef_mlx
    started=time.perf_counter();model=clef_mlx.load(path,head_dtype=mx.float32,backend='lm');mx.synchronize();loaded=time.perf_counter()-started
    for index,item in enumerate(spec['requests']):
        mx.reset_peak_memory();started=time.perf_counter()
        try:
            result=model.systemone(item['request'],max_length=16384,truncate=False);mx.synchronize()
            if result.get('state_truncated',False) is not False:raise ValueError('Unexpected truncation indicator')
            result.update(model=MODEL,state_truncated=False,done_reason='decide',
                metadata_origin='wrapper after strict truncate=False loader returned; same contract as production infer',
                model_repo=REPO,model_revision=REVISION,quantization_bits=8)
        except Exception as error:
            result={'error':f'{type(error).__name__}: {error}'}
            if isinstance(error,MemoryError) or any(t in str(error).lower() for t in ('out of memory','metal device lost')):
                print('RESULT:'+json.dumps(result|{'key':item['key']}),flush=True);raise
        result.update(key=item['key'],request_sha256=hashlib.sha256(json.dumps(item['request'],ensure_ascii=False,sort_keys=True).encode()).hexdigest(),load_seconds=loaded if index==0 else 0,compute_seconds=time.perf_counter()-started,
            device_peak_bytes=mx.get_peak_memory(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            backend='mlx',device='gpu',prototype=True)
        print('RESULT:'+json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
