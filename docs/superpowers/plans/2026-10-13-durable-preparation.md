# 자료 준비 작업 복구 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 여러 파일 등록과 야간 자동 처리의 요청·진행·결과를 보관하고 서버 재시작 후 성공분을 재사용하여 이어서 실행한다.

**Architecture:** 기존 App 실행 잠금과 처리 함수를 재사용한다. 장시간 자료 쓰기와 복구 상태 쓰기가 서로 잠기지 않도록 출력 폴더의 별도 `jobs.sqlite3`에 작업을 저장한다. OS 파일 잠금으로 살아 있는 다른 실행기의 작업을 가져오지 않는다. 단계 1의 첫 전달 범위는 파일 대기열과 야간 처리이며 질문·대화 영속화 및 동시 갱신은 후속 단계다.

**Tech Stack:** Python 표준 sqlite3, fcntl/macOS·msvcrt/Windows, 기존 HTML/JS와 unittest. 추가 의존성 없음.

**Spec:** [승인된 제품화 설계](../../제품화-설계-20261013.md), 2026-10-13 사용자가 진행 승인. 운영은 Windows 사내 서버 + 여러 사용자 브라우저 접속으로 확정.

## Global Constraints

- 원문·기존 ID·검토·설정 보존. 원격 모델 호출의 정확히 한 번 실행은 보장하지 않는다.
- 중단 여부는 시각만으로 추정하지 않고 실행 잠금을 확인한다. 질문과 자료 관리의 기존 상호 배제는 유지한다.
- 재시작은 성공 데이터 재사용이며 완료 상태를 추측하지 않는다. 등록 완료 파일은 삭제돼도 건너뛴다.
- 요청·진행 상태에 인증 정보를 저장하지 않는다. 상태 저장 실패는 복구 가능인 척하지 않고 오류로 표시한다.
- 전체 제품 완료로 표현하지 않는다. 별도 작업 DB는 운영 원문 DB를 이관하지 않기 위한 이번 단계의 선택이다.

## Review Focus

- 원문 DB 커밋 직후 상태 기록 전 종료: 같은 파일 재실행 시 기존 ID·검토를 보존해야 한다.
- 완료 파일 삭제·미완료 파일 오류: 완료분을 다시 읽지 않고 남은 파일과 오류를 표시해야 한다.
- 같은 출력 폴더를 여는 두 실행기: 살아 있는 작업을 복구하거나 덮어쓰면 안 된다.
- 재시도 대기 중 종료·중지 요청 직후 종료: 복구 가능과 사용자 중지를 구분해야 한다.
- 상태 DB 오류·읽기 전용 Clef: 오류를 숨기거나 원문 DB에 새 테이블을 만들어서는 안 된다.

## Task 1: 작업 저장과 실행 소유권

**Files:** Create `ragkit/jobs.py`, `tests/test_jobs.py`.

**Interfaces:** `JobStore(output)`, `acquire()` → 잠금 파일 핸들, `release(handle)`, `latest()` → 저장 객체 또는 None, `save(kind, request, snapshot, identity=None)` → ID, `interrupted()` → 중단 작업 또는 None. 모든 SQLite 연결은 호출 내 닫는다.

- [x] 임시 폴더에서 저장·재생성·실제 자식 프로세스 강제 종료·동시 잠금·손상 DB를 검증하는 테스트를 먼저 작성하고 실패를 확인한다.
- [x] 표준 라이브러리로 별도 DB 스키마 버전·단일 작업 저널·실행 잠금을 구현한다. 완료된 과거 작업은 최근 20개만 유지하되 미완료 항목은 보존한다.
- [x] `.venv/bin/python -m unittest tests.test_jobs` 상당 명령으로 통과를 확인한다. 작업 폴더에서는 본 저장소의 기존 Python 환경을 사용한다.

## Task 2: 기존 등록·야간 처리와 UI 연결

**Files:** Modify `ragkit/web.py`, `ragkit/overnight.py`, `ragkit/assets/app.html`, `ragkit/assets/app.js`; extend `tests/test_jobs.py` and package/test entrypoints.

**Interfaces:** `/api/job/resume`, `/api/job/dismiss`는 `{id}`만 받으며 기존 세션·Origin 검사를 적용한다. `App.resume_job(body)`는 저장된 종류에 대응하는 기존 검증 진입점을 호출한다. `App.recover_jobs()`는 서버 시작 시 미완료 작업을 확인해 재개하거나 복구 안내를 남긴다.

- [x] 실제 App으로 파일 대기열 종료/재시작·완료 파일 삭제·원문 커밋 후 종료·야간 중지와 성공분 재사용·잘못된 ID·동시 복구 테스트를 추가하고 실패를 확인한다.
- [x] 요청은 실행 전 저장하고, 파일 경계·야간 단계·종료 시 상태를 저장한다. 야간 복구는 DB 캐시 검사를 다시 수행하고 저장된 제안을 재사용한다. 자료 준비가 끝났다면 원본 Excel 없이 계속한다.
- [x] 작업 카드에 복구 사유, 이어서 실행, 대기 해제를 표시한다. 시작 시 임베딩 자동 작업이 중단 작업을 가리지 않게 한다.
- [x] 관련 검사, 전체 오프라인 검사, JS 문법·실제 브라우저, Windows 패키지 검사로 확인한다.
- [x] README·전체 흐름·임원 보고서·승인 설계를 현재 구현 범위에 맞춘다.
- [x] 최종 독립 검토의 3건을 회귀 검사로 재현·수정하고 Ponytail 검토를 마쳤다. 커밋·main 반영·푸시의 최종 SHA는 작업 완료 보고에 기록한다.

## 범위 이후

개별 변환·다듬기·영어·벡터 버튼의 자동 복구, 질문/대화 저장, 자료 갱신 중 질문, 권한 분리와 FastAPI 전환은 이번 작업 저널의 검증 뒤 이어갈 제품화 단계다. 현재 기능과 혼동하지 않도록 문서에 표시한다.

## 최종 검증 근거 (2026-10-13)

- 신규 작업 저장·복구 검사 14개 통과. 실제 프로세스 종료, 저장 직후 재실행과 검토 보존, 완료 Excel 삭제, 야간 캐시 재사용, 중지 기록, HTTP Origin·ID 검사, 손상 파일 보존을 확인했다.
- 독립 검토의 SQLite 미완료 저널·외부 실행기 완료 상태·재시작 후 실패 파일 결과 누락을 모두 재현하고 수정했다.
- `windows-demo/test_demo.py` 전체 오프라인 검사와 `package_windows.py` 독립 압축 해제·CLI·배치 형식 검사가 종료 코드 0으로 통과했다.
- 실제 Chromium에서 1440×1024 재개, 390×844 대기 해제, 재시작·자동 임베딩 이후 실패 파일 목록과 재시도 버튼을 확인했다. 콘솔 오류는 없었다. Browser 플러그인이 없어 설치된 Playwright를 사용했다.
- 별도 과거 `test_review_workflow_ui.py` 모의 검사는 기준 커밋과 변경 코드 모두 `ResizeObserver` 대역 누락으로 실패했다. 이번 범위에서는 해당 기존 대역을 수정하지 않고 실제 브라우저로 확인했다.
- Windows DRM·OS 잠금, 실제 모델 장애의 복구, 현장 데이터 부하·정확도는 미검증이다. 운영 DB·모델·개인 설정을 변경하지 않았다.
