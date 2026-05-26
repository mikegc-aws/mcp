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

"""Durable execution support for MCP Lambda Handler.

Enables MCP tools that run as AWS Lambda Durable Functions, supporting
long-running workflows with checkpointing, waits, and callbacks.

Requires: aws-durable-execution-sdk-python (lazy-imported, not a hard dependency).
"""

import functools
import inspect
import json
import logging
import os
import time
import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, get_args, get_origin, get_type_hints


logger = logging.getLogger(__name__)

_DURABLE_CONTEXT_TYPE_NAMES = frozenset({'DurableContext'})

_task_tools: Dict[str, Callable] = {}

_status_tool_registered = False


def _is_durable_context_type(type_hint: Any) -> bool:
    """Check if a type hint refers to DurableContext."""
    if type_hint is None:
        return False
    type_name = getattr(type_hint, '__name__', None)
    if type_name and type_name in _DURABLE_CONTEXT_TYPE_NAMES:
        return True
    if isinstance(type_hint, str) and type_hint in _DURABLE_CONTEXT_TYPE_NAMES:
        return True
    return False


def _get_self_function_name() -> str:
    """Get the qualified function name for self-invocation."""
    override = os.environ.get('MCP_TASK_FUNCTION_NAME')
    if override:
        return override
    func_name = os.environ.get('AWS_LAMBDA_FUNCTION_NAME', '')
    func_version = os.environ.get('AWS_LAMBDA_FUNCTION_VERSION', '$LATEST')
    return f'{func_name}:{func_version}'


def _get_type_schema(type_hint: Any) -> Dict[str, Any]:
    """Convert a Python type hint to JSON Schema."""
    if type_hint is int:
        return {'type': 'integer'}
    elif type_hint is float:
        return {'type': 'number'}
    elif type_hint is bool:
        return {'type': 'boolean'}
    elif type_hint is str:
        return {'type': 'string'}

    if isinstance(type_hint, type) and issubclass(type_hint, Enum):
        return {'type': 'string', 'enum': [e.value for e in type_hint]}

    origin = get_origin(type_hint)

    if origin is Union:
        args = get_args(type_hint)
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return _get_type_schema(non_none_args[0])
        return _get_type_schema(non_none_args[0]) if non_none_args else {'type': 'string'}

    if origin is None:
        return {'type': 'string'}

    if origin is dict or origin is Dict:
        args = get_args(type_hint)
        if not args:
            return {'type': 'object', 'additionalProperties': True}
        value_schema = _get_type_schema(args[1])
        return {'type': 'object', 'additionalProperties': value_schema}

    if origin is list or origin is List:
        args = get_args(type_hint)
        if not args:
            return {'type': 'array', 'items': {}}
        item_schema = _get_type_schema(args[0])
        return {'type': 'array', 'items': item_schema}

    return {'type': 'string'}


def task_tool(mcp, *, function_name: Optional[str] = None, invoke_mode: str = 'sync'):
    """Decorator to register an MCP tool backed by a durable execution.

    The decorated function runs as a Lambda Durable Function, supporting
    checkpointing, waits, callbacks, and execution times up to 1 year.

    Args:
        mcp: The MCPLambdaHandler instance to register the tool on.
        function_name: Qualified Lambda function name/ARN for self-invocation.
            Defaults to MCP_TASK_FUNCTION_NAME env var, then self.
        invoke_mode: "sync" (wait up to 15 min) or "async" (return task handle).
    """

    def decorator(func: Callable):
        tool_name = func.__name__

        doc = inspect.getdoc(func) or ''
        description = doc.split('\n\n')[0]

        hints = get_type_hints(func)
        hints.pop('return', None)

        # Parse docstring for argument descriptions
        arg_descriptions = {}
        if doc:
            lines = doc.split('\n')
            in_args = False
            for line in lines:
                if line.strip().startswith('Args:'):
                    in_args = True
                    continue
                if in_args:
                    if not line.strip() or line.strip().startswith('Returns:'):
                        break
                    if ':' in line:
                        arg_name, arg_desc = line.split(':', 1)
                        arg_descriptions[arg_name.strip()] = arg_desc.strip()

        # Build schema, excluding DurableContext parameter
        properties = {}
        required = []
        context_param_name = None

        for param_name, param_type in hints.items():
            if _is_durable_context_type(param_type):
                context_param_name = param_name
                continue
            param_schema = _get_type_schema(param_type)
            if param_name in arg_descriptions:
                param_schema['description'] = arg_descriptions[param_name]
            properties[param_name] = param_schema
            required.append(param_name)

        tool_schema = {
            'name': tool_name,
            'description': description,
            'inputSchema': {'type': 'object', 'properties': properties, 'required': required},
        }

        # Register schema for tools/list
        mcp.tools[tool_name] = tool_schema

        # Store the actual implementation and metadata for the durable handler
        _task_tools[tool_name] = {
            'func': func,
            'context_param': context_param_name,
            'invoke_mode': invoke_mode,
        }

        # Create the dispatch function that runs in the MCP protocol path
        def dispatch(**kwargs):
            import boto3

            task_id = str(uuid.uuid4())
            target = function_name or _get_self_function_name()

            payload = {
                '_mcp_task_dispatch': {
                    'tool_name': tool_name,
                    'arguments': kwargs,
                    'task_id': task_id,
                }
            }

            lambda_client = boto3.client('lambda')

            if invoke_mode == 'sync':
                response = lambda_client.invoke(
                    FunctionName=target,
                    InvocationType='RequestResponse',
                    Payload=json.dumps(payload),
                )
                response_payload = json.loads(response['Payload'].read())

                if response.get('FunctionError'):
                    error_msg = response_payload.get('errorMessage', 'Task execution failed')
                    raise RuntimeError(error_msg)

                return json.dumps(response_payload)

            else:
                # Async: write task record, invoke async, return handle
                _write_task_record(mcp, task_id, tool_name, 'RUNNING')

                lambda_client.invoke(
                    FunctionName=target,
                    InvocationType='Event',
                    Payload=json.dumps(payload),
                )

                return json.dumps({
                    'taskId': task_id,
                    'status': 'RUNNING',
                    'message': f'Task {tool_name} started. Use tasks/get or get_task_status to poll.',
                })

        mcp.tool_implementations[tool_name] = dispatch

        # Register status tool if this is the first async task tool
        if invoke_mode == 'async':
            _ensure_status_tool_registered(mcp)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._task_tool_meta = {
            'tool_name': tool_name,
            'context_param': context_param_name,
            'invoke_mode': invoke_mode,
        }

        return wrapper

    return decorator


def _ensure_status_tool_registered(mcp):
    """Register the get_task_status bridging tool once."""
    global _status_tool_registered
    if _status_tool_registered:
        return
    _status_tool_registered = True

    tool_schema = {
        'name': 'get_task_status',
        'description': 'Get the status of a running task. Returns status and result when complete.',
        'inputSchema': {
            'type': 'object',
            'properties': {
                'task_id': {
                    'type': 'string',
                    'description': 'The task ID returned when the task was started.',
                },
            },
            'required': ['task_id'],
        },
    }

    mcp.tools['get_task_status'] = tool_schema

    def get_task_status_impl(task_id: str) -> str:
        record = _read_task_record(mcp, task_id)
        if record is None:
            return json.dumps({'error': f'Task {task_id} not found'})
        return json.dumps(record)

    mcp.tool_implementations['get_task_status'] = get_task_status_impl


def _write_task_record(mcp, task_id: str, tool_name: str, status: str, result: Any = None, error: Any = None):
    """Write a task record to the task store."""
    import boto3
    from botocore.config import Config

    table_name = os.environ.get('MCP_TASK_TABLE', os.environ.get('MCP_SESSION_TABLE', 'mcp_sessions'))
    config = Config(user_agent_extra=f'md/awslabs#mcp#mcp-lambda-handler#durable')
    dynamodb = boto3.resource('dynamodb', config=config)
    table = dynamodb.Table(table_name)

    item = {
        'session_id': f'task#{task_id}',
        'task_id': task_id,
        'tool_name': tool_name,
        'status': status,
        'created_at': int(time.time()),
        'updated_at': int(time.time()),
        'expires_at': int(time.time()) + (7 * 24 * 60 * 60),  # 7 day TTL
    }
    if result is not None:
        item['result'] = result
    if error is not None:
        item['error'] = error

    table.put_item(Item=item)


def _update_task_record(task_id: str, status: str, result: Any = None, error: Any = None):
    """Update a task record on completion."""
    import boto3
    from botocore.config import Config

    table_name = os.environ.get('MCP_TASK_TABLE', os.environ.get('MCP_SESSION_TABLE', 'mcp_sessions'))
    config = Config(user_agent_extra=f'md/awslabs#mcp#mcp-lambda-handler#durable')
    dynamodb = boto3.resource('dynamodb', config=config)
    table = dynamodb.Table(table_name)

    update_expr = 'SET #status = :status, updated_at = :updated_at'
    expr_values = {':status': status, ':updated_at': int(time.time())}
    expr_names = {'#status': 'status'}

    if result is not None:
        update_expr += ', #result = :result'
        expr_values[':result'] = result
        expr_names['#result'] = 'result'
    if error is not None:
        update_expr += ', #error = :error'
        expr_values[':error'] = error
        expr_names['#error'] = 'error'

    table.update_item(
        Key={'session_id': f'task#{task_id}'},
        UpdateExpression=update_expr,
        ExpressionAttributeValues=expr_values,
        ExpressionAttributeNames=expr_names,
    )


def _read_task_record(mcp, task_id: str) -> Optional[Dict[str, Any]]:
    """Read a task record from the task store."""
    import boto3
    from botocore.config import Config

    table_name = os.environ.get('MCP_TASK_TABLE', os.environ.get('MCP_SESSION_TABLE', 'mcp_sessions'))
    config = Config(user_agent_extra=f'md/awslabs#mcp#mcp-lambda-handler#durable')
    dynamodb = boto3.resource('dynamodb', config=config)
    table = dynamodb.Table(table_name)

    response = table.get_item(Key={'session_id': f'task#{task_id}'})
    item = response.get('Item')
    if not item:
        return None

    record = {
        'taskId': item['task_id'],
        'status': item['status'],
        'toolName': item.get('tool_name'),
        'createdAt': item.get('created_at'),
        'updatedAt': item.get('updated_at'),
    }
    if 'result' in item:
        record['result'] = item['result']
    if 'error' in item:
        record['error'] = item['error']
    return record


def handle_tasks_get(mcp, params: Optional[Dict]) -> Optional[Dict[str, Any]]:
    """Handle the tasks/get JSON-RPC method.

    Returns a result dict if handled, None if params are invalid.
    """
    if not params or 'taskId' not in params:
        return None

    task_id = params['taskId']
    record = _read_task_record(mcp, task_id)

    if record is None:
        return {'error': f'Task {task_id} not found'}

    return record


def create_durable_handler(mcp):
    """Create the Lambda handler for a durable MCP server.

    Returns a handler function that should be wrapped with @durable_execution
    by the user (or used directly if the Lambda runtime applies it via config).

    The handler routes between:
    - MCP protocol requests (HTTP events from API Gateway / Function URL)
    - Task dispatch events (self-invoked for durable tool execution)
    """

    def handler(event, context):
        if isinstance(event, dict) and '_mcp_task_dispatch' in event:
            return _execute_task(event['_mcp_task_dispatch'], context)
        return mcp.handle_request(event, context)

    return handler


def _execute_task(dispatch: Dict[str, Any], context: Any) -> Any:
    """Execute a task tool within a durable execution."""
    tool_name = dispatch['tool_name']
    arguments = dispatch.get('arguments', {}).copy()
    task_id = dispatch.get('task_id')

    entry = _task_tools.get(tool_name)
    if entry is None:
        raise ValueError(f"Task tool '{tool_name}' not found in registry")

    func = entry['func']
    context_param = entry['context_param']
    invoke_mode = entry['invoke_mode']

    try:
        if context_param:
            arguments[context_param] = context
        result = func(**arguments)

        if invoke_mode == 'async' and task_id:
            result_str = result if isinstance(result, str) else json.dumps(result)
            _update_task_record(task_id, 'SUCCEEDED', result=result_str)

        return result

    except Exception as e:
        if invoke_mode == 'async' and task_id:
            _update_task_record(task_id, 'FAILED', error=str(e))
        raise
