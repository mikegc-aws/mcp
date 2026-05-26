# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the durable execution module."""

import json
import pytest
from awslabs.mcp_lambda_handler import MCPLambdaHandler
from awslabs.mcp_lambda_handler.durable import (
    _get_type_schema,
    _is_durable_context_type,
    _read_task_record,
    _task_tools,
    _update_task_record,
    _write_task_record,
    create_durable_handler,
    handle_tasks_get,
    task_tool,
)
from unittest.mock import MagicMock, patch


# --- Helpers ---


class FakeDurableContext:
    """Fake DurableContext for testing."""

    class _StepResult:
        def __init__(self, value):
            self._value = value

    def step(self, func):
        return func

    def wait(self, duration):
        pass

    def create_callback(self, name=None):
        return MagicMock(callback_id='cb-123', result=lambda: 'approved')


# Give it the right __name__ so _is_durable_context_type recognizes it
FakeDurableContext.__name__ = 'DurableContext'


def _reset_module_state():
    """Reset module-level state between tests."""
    import awslabs.mcp_lambda_handler.durable as durable_mod

    _task_tools.clear()
    durable_mod._status_tool_registered = False


@pytest.fixture(autouse=True)
def reset_state():
    """Reset module state before each test."""
    _reset_module_state()
    yield
    _reset_module_state()


# --- Test _is_durable_context_type ---


class TestIsDurableContextType:
    def test_recognizes_class_with_name(self):
        assert _is_durable_context_type(FakeDurableContext) is True

    def test_recognizes_string_annotation(self):
        assert _is_durable_context_type('DurableContext') is True

    def test_rejects_other_types(self):
        assert _is_durable_context_type(str) is False
        assert _is_durable_context_type(int) is False
        assert _is_durable_context_type(None) is False

    def test_rejects_other_strings(self):
        assert _is_durable_context_type('SomeOtherContext') is False


# --- Test schema generation ---


class TestGetTypeSchema:
    def test_basic_types(self):
        assert _get_type_schema(str) == {'type': 'string'}
        assert _get_type_schema(int) == {'type': 'integer'}
        assert _get_type_schema(float) == {'type': 'number'}
        assert _get_type_schema(bool) == {'type': 'boolean'}


# --- Test task_tool decorator ---


class TestTaskToolDecorator:
    def test_registers_tool_schema(self):
        mcp = MCPLambdaHandler('test')

        @task_tool(mcp)
        def my_task(query: str, limit: int, context: FakeDurableContext) -> str:
            """Search for data.

            Args:
                query: The search query
                limit: Maximum results
            """
            return 'result'

        assert 'my_task' in mcp.tools
        schema = mcp.tools['my_task']
        assert schema['name'] == 'my_task'
        assert schema['description'] == 'Search for data.'
        assert 'query' in schema['inputSchema']['properties']
        assert 'limit' in schema['inputSchema']['properties']
        assert 'context' not in schema['inputSchema']['properties']
        assert 'query' in schema['inputSchema']['required']
        assert 'limit' in schema['inputSchema']['required']
        assert 'context' not in schema['inputSchema']['required']

    def test_registers_dispatch_implementation(self):
        mcp = MCPLambdaHandler('test')

        @task_tool(mcp)
        def my_task(query: str, context: FakeDurableContext) -> str:
            """Do work."""
            return 'done'

        assert 'my_task' in mcp.tool_implementations
        assert callable(mcp.tool_implementations['my_task'])

    def test_stores_in_task_tools_registry(self):
        mcp = MCPLambdaHandler('test')

        @task_tool(mcp)
        def my_task(x: str, context: FakeDurableContext) -> str:
            """Do work."""
            return x

        assert 'my_task' in _task_tools
        assert _task_tools['my_task'] is not None

    def test_async_registers_status_tool(self):
        mcp = MCPLambdaHandler('test')

        @task_tool(mcp, invoke_mode='async')
        def slow_task(data: str, context: FakeDurableContext) -> str:
            """Long running."""
            return data

        assert 'get_task_status' in mcp.tools
        assert 'get_task_status' in mcp.tool_implementations

    def test_sync_does_not_register_status_tool(self):
        mcp = MCPLambdaHandler('test')

        @task_tool(mcp, invoke_mode='sync')
        def fast_task(data: str, context: FakeDurableContext) -> str:
            """Quick."""
            return data

        assert 'get_task_status' not in mcp.tools

    def test_preserves_function_metadata(self):
        mcp = MCPLambdaHandler('test')

        @task_tool(mcp)
        def my_task(x: str, context: FakeDurableContext) -> str:
            """Original docstring."""
            return x

        assert my_task.__name__ == 'my_task'
        assert my_task.__doc__ == 'Original docstring.'
        assert hasattr(my_task, '_task_tool_meta')
        assert my_task._task_tool_meta['tool_name'] == 'my_task'
        assert my_task._task_tool_meta['context_param'] == 'context'
        assert my_task._task_tool_meta['invoke_mode'] == 'sync'


# --- Test sync dispatch ---


class TestSyncDispatch:
    @patch('boto3.client')
    def test_sync_invokes_lambda_and_returns_result(self, mock_boto_client):
        mcp = MCPLambdaHandler('test')

        @task_tool(mcp, function_name='my-func:1')
        def analyze(dataset_url: str, context: FakeDurableContext) -> str:
            """Analyze data.

            Args:
                dataset_url: URL to analyze
            """
            return 'analysis complete'

        mock_lambda = MagicMock()
        mock_boto_client.return_value = mock_lambda

        payload_stream = MagicMock()
        payload_stream.read.return_value = json.dumps('analysis complete').encode()
        mock_lambda.invoke.return_value = {
            'Payload': payload_stream,
            'StatusCode': 200,
        }

        result = mcp.tool_implementations['analyze'](dataset_url='http://example.com/data.csv')

        mock_lambda.invoke.assert_called_once()
        call_kwargs = mock_lambda.invoke.call_args[1]
        assert call_kwargs['FunctionName'] == 'my-func:1'
        assert call_kwargs['InvocationType'] == 'RequestResponse'

        dispatch_payload = json.loads(call_kwargs['Payload'])
        assert dispatch_payload['_mcp_task_dispatch']['tool_name'] == 'analyze'
        assert dispatch_payload['_mcp_task_dispatch']['arguments'] == {
            'dataset_url': 'http://example.com/data.csv'
        }

        parsed = json.loads(result)
        assert parsed == 'analysis complete'

    @patch('boto3.client')
    def test_sync_raises_on_function_error(self, mock_boto_client):
        mcp = MCPLambdaHandler('test')

        @task_tool(mcp, function_name='my-func:1')
        def failing_task(x: str, context: FakeDurableContext) -> str:
            """Fail."""
            raise RuntimeError('boom')

        mock_lambda = MagicMock()
        mock_boto_client.return_value = mock_lambda

        payload_stream = MagicMock()
        payload_stream.read.return_value = json.dumps(
            {'errorMessage': 'boom', 'errorType': 'RuntimeError'}
        ).encode()
        mock_lambda.invoke.return_value = {
            'Payload': payload_stream,
            'StatusCode': 200,
            'FunctionError': 'Unhandled',
        }

        with pytest.raises(RuntimeError, match='boom'):
            mcp.tool_implementations['failing_task'](x='hello')


# --- Test async dispatch ---


class TestAsyncDispatch:
    @patch('awslabs.mcp_lambda_handler.durable._write_task_record')
    @patch('boto3.client')
    def test_async_invokes_and_returns_task_handle(self, mock_boto_client, mock_write):
        mcp = MCPLambdaHandler('test')

        @task_tool(mcp, function_name='my-func:1', invoke_mode='async')
        def train(config: str, context: FakeDurableContext) -> str:
            """Train model.

            Args:
                config: Training config
            """
            return 'model-123'

        mock_lambda = MagicMock()
        mock_boto_client.return_value = mock_lambda
        mock_lambda.invoke.return_value = {'StatusCode': 202}

        result = mcp.tool_implementations['train'](config='{"lr": 0.01}')
        parsed = json.loads(result)

        assert parsed['status'] == 'RUNNING'
        assert 'taskId' in parsed

        mock_lambda.invoke.assert_called_once()
        call_kwargs = mock_lambda.invoke.call_args[1]
        assert call_kwargs['InvocationType'] == 'Event'

        mock_write.assert_called_once()


# --- Test create_durable_handler ---


class TestCreateDurableHandler:
    def test_routes_mcp_requests_to_handler(self):
        mcp = MCPLambdaHandler('test')

        @mcp.tool()
        def ping_tool() -> str:
            """Quick ping."""
            return 'pong'

        handler = create_durable_handler(mcp)

        event = {
            'httpMethod': 'POST',
            'headers': {'content-type': 'application/json'},
            'body': json.dumps(
                {
                    'jsonrpc': '2.0',
                    'id': '1',
                    'method': 'tools/list',
                    'params': {},
                }
            ),
        }

        result = handler(event, None)
        assert result['statusCode'] == 200
        body = json.loads(result['body'])
        tool_names = [t['name'] for t in body['result']['tools']]
        assert 'ping_tool' in tool_names

    def test_routes_task_dispatch_to_executor(self):
        mcp = MCPLambdaHandler('test')

        @task_tool(mcp)
        def compute(x: int, context: FakeDurableContext) -> str:
            """Compute.

            Args:
                x: Input number
            """
            return f'result-{x}'

        handler = create_durable_handler(mcp)

        event = {
            '_mcp_task_dispatch': {
                'tool_name': 'compute',
                'arguments': {'x': 42},
                'task_id': 'task-abc',
            }
        }

        fake_context = FakeDurableContext()
        result = handler(event, fake_context)
        assert result == 'result-42'

    def test_task_dispatch_injects_context(self):
        mcp = MCPLambdaHandler('test')
        received_context = {}

        @task_tool(mcp)
        def check_ctx(x: str, context: FakeDurableContext) -> str:
            """Check.

            Args:
                x: Input
            """
            received_context['ctx'] = context
            return 'ok'

        handler = create_durable_handler(mcp)
        fake_context = FakeDurableContext()

        event = {
            '_mcp_task_dispatch': {
                'tool_name': 'check_ctx',
                'arguments': {'x': 'hello'},
                'task_id': 'task-xyz',
            }
        }

        handler(event, fake_context)
        assert received_context['ctx'] is fake_context

    def test_task_dispatch_unknown_tool_raises(self):
        mcp = MCPLambdaHandler('test')
        handler = create_durable_handler(mcp)

        event = {
            '_mcp_task_dispatch': {
                'tool_name': 'nonexistent',
                'arguments': {},
                'task_id': 'task-000',
            }
        }

        with pytest.raises(ValueError, match="Task tool 'nonexistent' not found"):
            handler(event, None)


# --- Test handle_tasks_get ---


class TestHandleTasksGet:
    @patch('awslabs.mcp_lambda_handler.durable._read_task_record')
    def test_returns_task_record(self, mock_read):
        mcp = MCPLambdaHandler('test')
        mock_read.return_value = {
            'taskId': 'task-123',
            'status': 'SUCCEEDED',
            'result': '"done"',
        }

        result = handle_tasks_get(mcp, {'taskId': 'task-123'})
        assert result['status'] == 'SUCCEEDED'
        assert result['taskId'] == 'task-123'

    @patch('awslabs.mcp_lambda_handler.durable._read_task_record')
    def test_returns_not_found(self, mock_read):
        mcp = MCPLambdaHandler('test')
        mock_read.return_value = None

        result = handle_tasks_get(mcp, {'taskId': 'no-such-task'})
        assert 'error' in result

    def test_returns_none_for_missing_params(self):
        mcp = MCPLambdaHandler('test')
        assert handle_tasks_get(mcp, None) is None
        assert handle_tasks_get(mcp, {}) is None


# --- Test tasks/get JSON-RPC method ---


class TestTasksGetMethod:
    @patch('awslabs.mcp_lambda_handler.durable._read_task_record')
    def test_tasks_get_via_handle_request(self, mock_read):
        mcp = MCPLambdaHandler('test')
        mock_read.return_value = {
            'taskId': 'task-abc',
            'status': 'RUNNING',
            'toolName': 'slow_task',
        }

        event = {
            'httpMethod': 'POST',
            'headers': {'content-type': 'application/json'},
            'body': json.dumps(
                {
                    'jsonrpc': '2.0',
                    'id': '1',
                    'method': 'tasks/get',
                    'params': {'taskId': 'task-abc'},
                }
            ),
        }

        result = mcp.handle_request(event, None)
        assert result['statusCode'] == 200
        body = json.loads(result['body'])
        assert body['result']['status'] == 'RUNNING'
        assert body['result']['taskId'] == 'task-abc'

    def test_tasks_get_missing_task_id(self):
        mcp = MCPLambdaHandler('test')

        event = {
            'httpMethod': 'POST',
            'headers': {'content-type': 'application/json'},
            'body': json.dumps(
                {
                    'jsonrpc': '2.0',
                    'id': '1',
                    'method': 'tasks/get',
                    'params': {},
                }
            ),
        }

        result = mcp.handle_request(event, None)
        assert result['statusCode'] == 400
        body = json.loads(result['body'])
        assert body['error']['code'] == -32602


# --- Test async task completion writes record ---


class TestAsyncTaskCompletion:
    @patch('awslabs.mcp_lambda_handler.durable._update_task_record')
    def test_successful_async_task_updates_record(self, mock_update):
        mcp = MCPLambdaHandler('test')

        @task_tool(mcp, invoke_mode='async')
        def async_task(data: str, context: FakeDurableContext) -> str:
            """Process data.

            Args:
                data: Input data
            """
            return f'processed-{data}'

        handler = create_durable_handler(mcp)
        fake_context = FakeDurableContext()

        event = {
            '_mcp_task_dispatch': {
                'tool_name': 'async_task',
                'arguments': {'data': 'hello'},
                'task_id': 'task-success',
            }
        }

        result = handler(event, fake_context)
        assert result == 'processed-hello'
        mock_update.assert_called_once_with('task-success', 'SUCCEEDED', result='processed-hello')

    @patch('awslabs.mcp_lambda_handler.durable._update_task_record')
    def test_failed_async_task_updates_record(self, mock_update):
        mcp = MCPLambdaHandler('test')

        @task_tool(mcp, invoke_mode='async')
        def failing_task(data: str, context: FakeDurableContext) -> str:
            """Fail.

            Args:
                data: Input
            """
            raise ValueError('intentional failure')

        handler = create_durable_handler(mcp)
        fake_context = FakeDurableContext()

        event = {
            '_mcp_task_dispatch': {
                'tool_name': 'failing_task',
                'arguments': {'data': 'x'},
                'task_id': 'task-fail',
            }
        }

        with pytest.raises(ValueError, match='intentional failure'):
            handler(event, fake_context)

        mock_update.assert_called_once_with('task-fail', 'FAILED', error='intentional failure')


# --- Test tools/list includes task tools ---


class TestToolsListIncludesTaskTools:
    def test_task_tools_appear_in_tools_list(self):
        mcp = MCPLambdaHandler('test')

        @mcp.tool()
        def regular_tool() -> str:
            """Fast."""
            return 'fast'

        @task_tool(mcp)
        def slow_tool(x: str, context: FakeDurableContext) -> str:
            """Slow.

            Args:
                x: Input
            """
            return x

        event = {
            'httpMethod': 'POST',
            'headers': {'content-type': 'application/json'},
            'body': json.dumps(
                {
                    'jsonrpc': '2.0',
                    'id': '1',
                    'method': 'tools/list',
                    'params': {},
                }
            ),
        }

        result = mcp.handle_request(event, None)
        body = json.loads(result['body'])
        tool_names = [t['name'] for t in body['result']['tools']]
        assert 'regular_tool' in tool_names
        assert 'slow_tool' in tool_names


# --- Test edge cases ---


class TestEdgeCases:
    def test_task_tool_without_context_param(self):
        mcp = MCPLambdaHandler('test')

        @task_tool(mcp)
        def no_context_tool(x: str) -> str:
            """No context param.

            Args:
                x: Input value
            """
            return f'got {x}'

        assert 'no_context_tool' in mcp.tools
        schema = mcp.tools['no_context_tool']
        assert 'x' in schema['inputSchema']['properties']
        assert schema['inputSchema']['required'] == ['x']

    def test_task_tool_without_context_executes(self):
        mcp = MCPLambdaHandler('test')

        @task_tool(mcp)
        def no_context_tool(x: str) -> str:
            """No context.

            Args:
                x: Input
            """
            return f'got {x}'

        handler = create_durable_handler(mcp)
        event = {
            '_mcp_task_dispatch': {
                'tool_name': 'no_context_tool',
                'arguments': {'x': 'hello'},
                'task_id': 'task-nc',
            }
        }

        result = handler(event, FakeDurableContext())
        assert result == 'got hello'

    def test_task_tool_no_args(self):
        mcp = MCPLambdaHandler('test')

        @task_tool(mcp)
        def no_args_tool(context: FakeDurableContext) -> str:
            """Tool with no user arguments."""
            return 'done'

        schema = mcp.tools['no_args_tool']
        assert schema['inputSchema']['properties'] == {}
        assert schema['inputSchema']['required'] == []

    def test_import_without_durable_sdk(self):
        from awslabs.mcp_lambda_handler import durable

        assert hasattr(durable, 'task_tool')
        assert hasattr(durable, 'create_durable_handler')

    def test_get_self_function_name_from_env(self, monkeypatch):
        from awslabs.mcp_lambda_handler.durable import _get_self_function_name

        monkeypatch.setenv('MCP_TASK_FUNCTION_NAME', 'my-custom-func:prod')
        assert _get_self_function_name() == 'my-custom-func:prod'

    def test_get_self_function_name_from_lambda_env(self, monkeypatch):
        from awslabs.mcp_lambda_handler.durable import _get_self_function_name

        monkeypatch.delenv('MCP_TASK_FUNCTION_NAME', raising=False)
        monkeypatch.setenv('AWS_LAMBDA_FUNCTION_NAME', 'my-func')
        monkeypatch.setenv('AWS_LAMBDA_FUNCTION_VERSION', '3')
        assert _get_self_function_name() == 'my-func:3'


# --- Test get_task_status bridging tool ---


class TestGetTaskStatusTool:
    @patch('awslabs.mcp_lambda_handler.durable._read_task_record')
    def test_get_task_status_via_tools_call(self, mock_read):
        mcp = MCPLambdaHandler('test')

        @task_tool(mcp, invoke_mode='async')
        def some_task(data: str, context: FakeDurableContext) -> str:
            """Task.

            Args:
                data: Input
            """
            return data

        mock_read.return_value = {
            'taskId': 'task-poll',
            'status': 'SUCCEEDED',
            'result': '"all done"',
            'toolName': 'some_task',
        }

        event = {
            'httpMethod': 'POST',
            'headers': {'content-type': 'application/json'},
            'body': json.dumps(
                {
                    'jsonrpc': '2.0',
                    'id': '2',
                    'method': 'tools/call',
                    'params': {
                        'name': 'get_task_status',
                        'arguments': {'task_id': 'task-poll'},
                    },
                }
            ),
        }

        result = mcp.handle_request(event, None)
        assert result['statusCode'] == 200
        body = json.loads(result['body'])
        content_text = body['result']['content'][0]['text']
        parsed = json.loads(content_text)
        assert parsed['status'] == 'SUCCEEDED'
        assert parsed['taskId'] == 'task-poll'

    @patch('awslabs.mcp_lambda_handler.durable._read_task_record')
    def test_get_task_status_not_found(self, mock_read):
        mcp = MCPLambdaHandler('test')

        @task_tool(mcp, invoke_mode='async')
        def some_task(data: str, context: FakeDurableContext) -> str:
            """Task.

            Args:
                data: Input
            """
            return data

        mock_read.return_value = None

        event = {
            'httpMethod': 'POST',
            'headers': {'content-type': 'application/json'},
            'body': json.dumps(
                {
                    'jsonrpc': '2.0',
                    'id': '3',
                    'method': 'tools/call',
                    'params': {
                        'name': 'get_task_status',
                        'arguments': {'task_id': 'nonexistent'},
                    },
                }
            ),
        }

        result = mcp.handle_request(event, None)
        assert result['statusCode'] == 200
        body = json.loads(result['body'])
        content_text = body['result']['content'][0]['text']
        parsed = json.loads(content_text)
        assert 'error' in parsed


# --- Test DynamoDB task record operations with moto ---


class TestDynamoDBTaskRecords:
    @pytest.fixture
    def task_table(self, monkeypatch):
        """Create a mocked DynamoDB table for task records."""
        import boto3
        from moto import mock_aws

        with mock_aws():
            monkeypatch.setenv('MCP_TASK_TABLE', 'test_tasks')
            dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
            dynamodb.create_table(
                TableName='test_tasks',
                KeySchema=[{'AttributeName': 'session_id', 'KeyType': 'HASH'}],
                AttributeDefinitions=[{'AttributeName': 'session_id', 'AttributeType': 'S'}],
                BillingMode='PAY_PER_REQUEST',
            )
            yield dynamodb

    def test_write_and_read_task_record(self, task_table, monkeypatch):
        monkeypatch.setenv('AWS_DEFAULT_REGION', 'us-east-1')
        mcp = MCPLambdaHandler('test')

        _write_task_record(mcp, 'task-w1', 'my_tool', 'RUNNING')
        record = _read_task_record(mcp, 'task-w1')

        assert record is not None
        assert record['taskId'] == 'task-w1'
        assert record['status'] == 'RUNNING'
        assert record['toolName'] == 'my_tool'

    def test_update_task_record(self, task_table, monkeypatch):
        monkeypatch.setenv('AWS_DEFAULT_REGION', 'us-east-1')
        mcp = MCPLambdaHandler('test')

        _write_task_record(mcp, 'task-u1', 'my_tool', 'RUNNING')
        _update_task_record('task-u1', 'SUCCEEDED', result='"done"')

        record = _read_task_record(mcp, 'task-u1')
        assert record['status'] == 'SUCCEEDED'
        assert record['result'] == '"done"'

    def test_update_task_record_with_error(self, task_table, monkeypatch):
        monkeypatch.setenv('AWS_DEFAULT_REGION', 'us-east-1')
        mcp = MCPLambdaHandler('test')

        _write_task_record(mcp, 'task-e1', 'my_tool', 'RUNNING')
        _update_task_record('task-e1', 'FAILED', error='something broke')

        record = _read_task_record(mcp, 'task-e1')
        assert record['status'] == 'FAILED'
        assert record['error'] == 'something broke'

    def test_read_nonexistent_task(self, task_table, monkeypatch):
        monkeypatch.setenv('AWS_DEFAULT_REGION', 'us-east-1')
        mcp = MCPLambdaHandler('test')

        record = _read_task_record(mcp, 'does-not-exist')
        assert record is None
