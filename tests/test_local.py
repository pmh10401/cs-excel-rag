"""외부 API 없이 로컬 벡터 경로를 검사합니다."""
import json
import sqlite3
import numpy as np
from ragkit.records import text_for
from ragkit.vectors import FIELDS, unit_vectors, ingest
from ragkit.local_cli import read_excel

def main(root):
    # 실제 모델 없이 실행하는 작은 검사입니다. assert 조건이 거짓이면 즉시 실패합니다.
    # 같은 방향의 유사도는 1, 직각인 방향의 유사도는 0인지 확인합니다.
    vectors = unit_vectors([[3, 0], [0, 4]])
    assert np.allclose(vectors @ vectors[0], [1, 0])
    # 잘못된 벡터를 넣었을 때 오류를 내는지도 확인합니다.
    for invalid in ([[0, 0]], [[float('nan'), 1]], []):
        try:
            unit_vectors(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid vector accepted')
    # 샘플 ID의 고유성과 '열 이름: 값'이 원문 그대로 본문에 들어가는지 확인합니다.
    sample = read_excel(root / 'outputs' / '01a083b4' / 'sample_data.xlsx')
    assert len({r['id'] for r in sample}) == 6
    assert all(f"{k}: {sample[0][k]}" in text_for(sample[0]) for k in FIELDS)
    # 실제 서버 대신 고정된 벡터를 주는 작은 대역입니다. 네트워크 요청은 없습니다.
    # client.embeddings.create(...)라는 호출 모양만 실제 클라이언트와 맞춥니다.
    class FakeClient:
        def __init__(self):
            self.calls = 0
            self.embeddings = self
        def create(self, **kwargs):
            from types import SimpleNamespace
            self.calls += 1
            return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1, 2])])
    fake = FakeClient()
    # 메모리 안에서만 검사하므로 실제 demo.sqlite3의 자료에는 영향을 주지 않습니다.
    # 첫 등록은 6건 생성, 재등록은 0건 생성, 한 행 수정 후에는 1건만 생성해야 합니다.
    # 목록에서 제거된 나머지 행이 삭제되고 수정한 원본이 보존되는지도 확인합니다.
    with sqlite3.connect(':memory:') as db:
        assert ingest(fake, db, sample) == 6
        assert ingest(fake, db, sample) == 0
        changed = [dict(sample[0], 비고='변경')]
        assert ingest(fake, db, changed) == 1
        assert db.execute('SELECT count(*) FROM records').fetchone()[0] == 1
        assert json.loads(db.execute('SELECT original_json FROM records').fetchone()[0])['비고'] == '변경'
    print('PASS: 벡터 검증·열 보존·중복 임베딩 방지·변경 반영·삭제 동기화', flush=True)
