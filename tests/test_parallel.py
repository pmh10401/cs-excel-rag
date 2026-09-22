"""병렬 요청의 상한·완료순 저장·실패 후 재개를 실제 SQLite와 모의 모델로 검사합니다."""
import json
import sqlite3
import threading
from pathlib import Path
from unittest.mock import patch

from ragkit import storage, records


def main(root):
    rules = {'id': 'preserve', '메모': 'rewrite'}
    rows = [{'id': f'PAR-{n}', '메모': '점검 기록', '_source_key': 'parallel-demo'} for n in range(1, 5)]
    config = {'base_url': 'http://127.0.0.1:1/v1', 'model': 'mock', 'reasoning_effort': 'high', 'normalization_workers': 2}
    for bad in (0, 1001, -1, True, False, 2.0, '2', None):
        try:
            storage.validate_workers(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'잘못된 동시 처리 수를 허용했습니다: {bad!r}')
    assert [storage.validate_workers(n) for n in (1, 2, 100, 1000)] == [1, 2, 100, 1000]
    # 100개 요청이 실제로 동시에 진입하고, 101번째는 자리가 난 뒤 실행되는지 확인합니다.
    hundred = threading.Barrier(100)
    many = [rows[0] | {'id': f'CAP-{n}'} for n in range(101)]
    def cap_normalize(client, config, row, rules, schema=None):
        if row['id'] != 'CAP-100':
            hundred.wait(timeout=10)
        return records.public_cells(row)
    cap_progress = []
    with sqlite3.connect(':memory:') as db, patch.object(storage, 'normalize', side_effect=cap_normalize):
        prepared = storage.ingest(db, many, rules, object(), config | {'normalization_workers': 100}, progress=cap_progress.append)
        assert len(prepared) == 101 and cap_progress[-1]['completed'] == 101
        assert max(p['active'] for p in cap_progress) == 100
        assert db.execute('SELECT COUNT(*) FROM normalized_records').fetchone()[0] == 101
        assert db.execute('SELECT COUNT(*) FROM normalization_checkpoints').fetchone()[0] == 0
    owner = threading.get_ident()
    barrier = threading.Barrier(2)
    release = threading.Event()
    lock = threading.Lock()
    active, peak = 0, 0
    progress = []
    calls = []

    def normalize(client, config, row, rules, schema=None):
        nonlocal active, peak
        with lock:
            calls.append(row['id'])
            active += 1
            peak = max(peak, active)
        try:
            if row['id'] in ('PAR-1', 'PAR-2'):
                barrier.wait(timeout=3)
            if row['id'] == 'PAR-1':
                assert release.wait(3), '먼저 끝난 두 번째 행이 DB에 저장되지 않았습니다.'
            return records.public_cells(row) | {'메모': '점검한 기록입니다.'}
        finally:
            with lock:
                active -= 1

    with sqlite3.connect(':memory:') as db:
        def on_progress(info):
            assert threading.get_ident() == owner, '진행 콜백이 DB 소유 스레드 밖에서 호출되었습니다.'
            progress.append(dict(info))
            if info['completed'] and info['current_id'] == 'PAR-2':
                cached = db.execute("SELECT normalized_json FROM normalization_checkpoints WHERE record_id='PAR-2'").fetchone()
                assert cached and json.loads(cached[0])['메모'] == '점검한 기록입니다.'
                release.set()
        with patch.object(storage, 'normalize', side_effect=normalize):
            prepared = storage.ingest(db, rows, rules, object(), config, progress=on_progress)
        assert peak == 2 and [x['id'] for x in prepared] == [x['id'] for x in rows]
        assert sorted(calls) == sorted(row['id'] for row in rows)
        assert progress[-1]['completed'] == 4 and progress[-1]['active'] == 0
        assert all(0 <= p['active'] <= 2 for p in progress)
        # 처리 수는 문장 내용에 영향을 주는 설정이 아니므로 바꿔도 캐시를 재사용합니다.
        with patch.object(storage, 'normalize', side_effect=AssertionError('캐시를 다시 요청했습니다.')):
            storage.ingest(db, rows, rules, object(), config | {'normalization_workers': 1}, progress=progress.append)
        assert progress[-1]['reused'] == 4

    # 첫 행과 중간 행이 실패해도 순차·병렬 모두 마지막 행까지 시도합니다.
    for workers, failed_ids in ((1, {'PAR-1', 'PAR-3'}), (2, {'PAR-1', 'PAR-3'}),
                               (2, {row['id'] for row in rows})):
        calls.clear()
        progress.clear()
        def failing(client, config, row, rules, schema=None):
            calls.append(row['id'])
            if row['id'] in failed_ids:
                if row['id'] == 'PAR-3':
                    return records.public_cells(row) | {'알 수 없는 열': '잘못된 응답'}
                raise ValueError('의도한 모델 요청 실패')
            return records.public_cells(row)
        with sqlite3.connect(':memory:') as db:
            storage.ingest(db, rows, rules)
            before = db.execute('SELECT * FROM normalized_records').fetchall()
            settings = config | {'normalization_workers': workers}
            with patch.object(storage, 'normalize', side_effect=failing):
                try:
                    storage.ingest(db, rows, rules, object(), settings, progress=progress.append)
                except storage.NormalizationBatchError as error:
                    assert {item['id'] for item in error.failures} == failed_ids
                    assert all(item['error'] for item in error.failures)
                else:
                    raise AssertionError('실패를 정상 완료로 처리했습니다.')
            assert set(calls) == {row['id'] for row in rows}, '나머지 행을 건너뛰었습니다.'
            assert progress[-1]['attempted'] == 4 and progress[-1]['active'] == 0
            assert progress[-1]['failed'] == len(failed_ids)
            assert progress[-1]['completed'] == 4 - len(failed_ids)
            assert db.execute('SELECT * FROM normalized_records').fetchall() == before
            assert {row[0] for row in db.execute('SELECT record_id FROM normalization_checkpoints')} == set(calls) - failed_ids
            calls.clear()
            def resume(client, config, row, rules, schema=None):
                calls.append(row['id'])
                return records.public_cells(row)
            with patch.object(storage, 'normalize', side_effect=resume):
                storage.ingest(db, rows, rules, object(), settings, progress=progress.append)
            assert set(calls) == failed_ids
            assert progress[-1]['completed'] == 4 and progress[-1]['reused'] == 4 - len(failed_ids)
            assert progress[-1]['failed'] == 0 and progress[-1]['failures'] == []

    # 저장 계층 자체의 오류는 행 단위 모델 실패로 숨기지 않습니다.
    with sqlite3.connect(':memory:') as db:
        calls.clear()
        with patch.object(storage, 'normalize', side_effect=resume), \
                patch.object(storage, 'prepared_record', side_effect=sqlite3.OperationalError('디스크 저장 실패')):
            try:
                storage.ingest(db, rows, rules, object(), config | {'normalization_workers': 1})
            except sqlite3.OperationalError:
                pass
            else:
                raise AssertionError('저장 오류를 숨겼습니다.')
        assert calls == ['PAR-1']
    print('PASS: 동시 처리 상한·완료순 JSON 저장·입력순 반환·순차·병렬 실패 후 전체 처리·실패 행만 재시도·저장 오류 전파·설정 검사')


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1] / 'windows-demo')
