# 발표 주장과 증거

|주장|출처|해석 범위|금지 확대|
|---|---|---|---|
|조건 선택 단계|A_selection.jsonl, summary.csv|동일 H후보·새 합성24조건|Clef 최고/필수, 점수=진실확률|
|근거·사실 진단|A_answers.jsonl, factcheck.py|자동 확인 가능한 항목과 unknown|자연어 전체/현장 정확도|
|원문 보호|B_controls.jsonl|정상/단일 주입오류12쌍|실제 모델 오류12건, 환각0|
|공통 양식 재사용|C_reuse.jsonl, C_reuse_gpu.jsonl|지원2종·합성6입력·실제검색|타팀 도입/범용Excel/Windows 검증|
|반복 운영|C_repeat.jsonl|소유 프로세스 복구·근거사본|정확히1회 실행, 무중단, 사람시간절감|

표의 원시 JSONL은 로컬 비공개 근거다. 공개 재집계는 ranking-results.json·answer-outcomes.json·validator-outcomes.json·schema-outcomes.json을 사용한다.
