"""외부 프로세스·서버 경계만 대체하여 임베딩 자동 준비를 검사합니다."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import httpx2 as httpx
from openai import APIConnectionError, AuthenticationError
from ragkit import embedding_runtime


def main(root):
    cfg = {'base_url': 'http://127.0.0.1:1234/v1', 'api_key': 'PRIVATE-EMBED-KEY',
           'model': 'cs-embedding', 'auto_start': True}
    catalog = [{'type': 'embedding', 'modelKey': 'text-embedding-qwen3-embedding-0.6b'}]
    missing = NS(data=[])
    available = NS(data=[NS(id='cs-embedding')])
    client = NS(models=NS(list=Mock(return_value=available)))
    run = Mock(return_value=NS(returncode=0, stdout=json.dumps(catalog), stderr=''))
    with patch.object(embedding_runtime.shutil, 'which', return_value='/tools/lms'), \
            patch.object(embedding_runtime.subprocess, 'run', run):
        embedding_runtime.prepare(client, cfg)
        run.assert_not_called()  # 실행 중인 모델은 다시 로드하지 않습니다.
        connection_error = APIConnectionError(request=httpx.Request('GET', cfg['base_url']+'/models'))
        client.models.list.side_effect = [connection_error, missing, available]
        embedding_runtime.prepare(client, cfg)
        commands = [c.args[0][1:] for c in run.call_args_list]
        assert commands[0] == ['daemon', 'up']
        assert commands[1] == ['server', 'start', '--port', '1234', '--bind', '127.0.0.1']
        assert commands[2] == ['ls', '--embedding', '--json']
        assert commands[3] == ['load', catalog[0]['modelKey'], '--identifier', 'cs-embedding', '--context-length', '8192']
        assert all('PRIVATE' not in str(command) for command in commands)
        assert all(c.kwargs.get('shell', False) is False for c in run.call_args_list)
        run.reset_mock(); client.models.list.reset_mock(side_effect=True)
        for changed in ({'base_url':'http://example.invalid/v1'}, {'auto_start':False}):
            embedding_runtime.prepare(client, cfg | changed)
        client.models.list.assert_not_called(); run.assert_not_called()
        # 인증 오류를 서버 종료로 오해하여 별도 서버를 시작하지 않습니다.
        response = httpx.Response(401, request=httpx.Request('GET', cfg['base_url']+'/models'))
        client.models.list.side_effect = AuthenticationError('invalid key', response=response, body={})
        try:
            embedding_runtime.prepare(client, cfg)
        except AuthenticationError:
            pass
        else:
            raise AssertionError('인증 오류를 숨겼습니다.')
        run.assert_not_called()
        # 잘못된 별칭은 임의의 다른 임베딩 모델로 바꾸지 않습니다.
        client.models.list.side_effect = None
        client.models.list.return_value = missing
        try:
            embedding_runtime.prepare(client, cfg | {'model':'unrelated-alias'})
        except RuntimeError as e:
            assert '찾지 못' in str(e)
        else:
            raise AssertionError('다른 모델을 임의로 선택했습니다.')
        assert not any('load' in c.args[0] for c in run.call_args_list)
        run.reset_mock()
        run.return_value = NS(returncode=1, stdout='', stderr='runtime unavailable')
        try:
            embedding_runtime.prepare(client, cfg)
        except RuntimeError as e:
            assert 'runtime unavailable' in str(e)
        else:
            raise AssertionError('CLI 실행 실패를 숨겼습니다.')
        run.reset_mock()
        run.return_value = NS(returncode=0, stdout=json.dumps(catalog), stderr='')
        client.models.list.side_effect = [missing, available]
        with patch.object(embedding_runtime.shutil, 'which', return_value=None), \
                patch.object(embedding_runtime.Path, 'is_file', return_value=True), \
                patch.object(embedding_runtime.sys, 'platform', 'win32'):
            embedding_runtime.prepare(client, cfg)
        assert all(c.args[0][0].endswith('lms.exe') for c in run.call_args_list)
        assert all(c.kwargs['creationflags'] == 0x08000000 for c in run.call_args_list)
    print('PASS: 임베딩 실행 재사용·로컬 서버/모델 자동 시작·원격/인증 보호·잘못된 모델/CLI 실패')


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1] / 'windows-demo')
