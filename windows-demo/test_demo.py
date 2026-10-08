"""Windows 데모의 네트워크 없는 전체 검사."""
from pathlib import Path
import sys
import sqlite3
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
if not (ROOT / 'ragkit').is_dir():
    sys.path.insert(0, str(ROOT.parent))
from tests.test_windows import main as test_windows
from tests.test_cslist import main as test_cslist
from tests.test_excel_app import main as test_excel_app
from tests.test_rag import main as test_rag
from tests.test_action_outcomes import main as test_action_outcomes
from tests.test_headers import main as test_headers
from tests.test_ui import main as test_ui
from tests.test_multiuser import main as test_multiuser
from tests.test_resume import main as test_resume
from tests.test_quality import main as test_quality
from tests.test_parallel import main as test_parallel
from tests.test_search_focus import main as test_search_focus
from tests.test_korean_search import main as test_korean_search
from tests.test_bulk_reviews import main as test_bulk_reviews
from tests.test_english import main as test_english
from tests.test_source_cache import main as test_source_cache
from tests.test_semantic import main as test_semantic
from tests.test_relations import main as test_relations
from tests.test_clef import main as test_clef
from tests.test_embedding_startup import main as test_embedding_startup

from tests.test_workflow_auto import main as test_workflow_auto
from tests.test_overnight import main as test_overnight
from tests.test_incremental import main as test_incremental
from tests.test_history_input import main as test_history_input
from tests.test_import_queue import main as test_import_queue
from tests.test_retrieval_evaluation import main as test_retrieval_evaluation

def test_detail_claims(_root):
    suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_detail_claims')
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.wasSuccessful(), '상세 근거·후속 질문 검사에 실패했습니다.'

def test_claim_verification(_root):
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in
        ('tests.test_nli', 'tests.test_claim_verification', 'tests.test_claim_verification_ui'))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.wasSuccessful(), '영어 근거 검사·선택 모델·HTTP 검사에 실패했습니다.'

def check_db_lifecycle(test):
    # Mac에서는 열린 파일도 삭제되므로, 연결을 추적해 Windows 전용 오류를 미리 잡습니다.
    opened = []
    connect = sqlite3.connect

    class TrackedConnection(sqlite3.Connection):
        closed = False

        def close(self):
            super().close()
            self.closed = True

    def tracked(path, *args, **kwargs):
        db = connect(path, *args, factory=TrackedConnection, **kwargs)
        if str(path) != ':memory:':
            opened.append((str(path), db))
        return db

    with patch.object(sqlite3, 'connect', side_effect=tracked):
        test(ROOT)
    unclosed = [path for path, db in opened if not db.closed]
    assert not unclosed, f'{test.__module__}: 닫히지 않은 DB 연결: {unclosed}'


if __name__ == "__main__":
    for test in (test_windows, test_cslist, test_excel_app, test_rag, test_action_outcomes, test_headers,
                 test_ui, test_multiuser, test_resume, test_quality, test_parallel, test_search_focus, test_korean_search, test_bulk_reviews, test_english, test_source_cache, test_semantic, test_relations, test_clef, test_embedding_startup, test_workflow_auto, test_retrieval_evaluation, test_overnight, test_incremental, test_history_input, test_import_queue, test_detail_claims, test_claim_verification):
        check_db_lifecycle(test)
    print('PASS: 모든 파일 DB 연결을 명시적으로 닫았습니다.')
