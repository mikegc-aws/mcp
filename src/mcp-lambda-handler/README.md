# MCP Lambda Handler Module

A Python library for creating serverless HTTP handlers for the Model Context Protocol (MCP) using AWS Lambda. This library provides a minimal, extensible framework for building MCP HTTP endpoints with pluggable session management support.

## Features

- 🚀 Easy serverless MCP HTTP handler creation using AWS Lambda
- 🔌 Pluggable session management system (NoOp or DynamoDB, or custom backends)

## Quick Start

1. Install the package with development dependencies:
```bash
pip install -e .[dev]
```

2. Use the handler in your AWS Lambda function:

## Basic Usage

```python
from awslabs.mcp_lambda_handler import MCPLambdaHandler

mcp = MCPLambdaHandler(name="mcp-lambda-server", version="1.0.0")

@mcp.tool()
def add_two_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

def lambda_handler(event, context):
    """AWS Lambda handler function."""
    return mcp.handle_request(event, context)
```

## Durable Task Tools (Long-Running MCP Tools)

For tools that need more than 15 minutes, checkpoint progress, or wait for external input, use `task_tool` with [AWS Lambda Durable Functions](https://docs.aws.amazon.com/lambda/latest/dg/durable-functions.html).

**Requirements:**
- Python 3.13 runtime (durable SDK pre-installed)
- Lambda configured with DurableConfig enabled

```python
from awslabs.mcp_lambda_handler import MCPLambdaHandler, create_durable_handler, task_tool

mcp = MCPLambdaHandler(name="my-server", version="1.0.0")

# Regular tool — runs inline
@mcp.tool()
def quick_lookup(query: str) -> str:
    """Fast lookup."""
    return cache.get(query)

# Task tool (sync) — blocks up to 15 min, checkpointed
@task_tool(mcp)
def analyze_dataset(dataset_url: str, context) -> str:
    """Analyze a large dataset with checkpointed steps.

    Args:
        dataset_url: URL of the dataset
    """
    from aws_durable_execution_sdk_python import durable_step
    from aws_durable_execution_sdk_python.types import StepContext

    @durable_step
    def download(ctx: StepContext):
        return fetch(dataset_url)

    @durable_step
    def analyze(ctx: StepContext, data):
        return summarize(data)

    data = context.step(download())
    return context.step(analyze(data))

# Task tool (async) — returns immediately, client polls for result
@task_tool(mcp, invoke_mode='async')
def train_model(config: str, context) -> str:
    """Train a model (may take hours).

    Args:
        config: Training configuration JSON
    """
    # ... long workflow with waits, callbacks, etc.
    return model_id

# Single Lambda handler entry point
handler = create_durable_handler(mcp)
```

**How it works:**
- `task_tool` registers the tool in `tools/list` like any other tool (the `context` parameter is excluded from the schema)
- On `tools/call`, the handler self-invokes the Lambda as a durable execution
- **Sync mode** (default): waits for the result and returns it inline
- **Async mode**: returns a task handle immediately; client polls via `tasks/get` JSON-RPC method or the auto-registered `get_task_status` tool

**Environment variables:**
- `MCP_TASK_FUNCTION_NAME` — Qualified ARN for self-invocation (defaults to current function)
- `MCP_TASK_TABLE` — DynamoDB table for async task state (defaults to `MCP_SESSION_TABLE`)

See [`examples/durable_mcp_server.py`](examples/durable_mcp_server.py) for a complete example with SAM template.

## Session Management

The library provides flexible session management with built-in support for DynamoDB and the ability to create custom session backends. You can use the default stateless (NoOp) session store, or configure a DynamoDB-backed store for persistent sessions.

## Example Architecture for Auth & Session Management

A typical serverless deployment using this library might look like:

- **API Gateway**: Exposes the `/mcp` endpoint.
- **Lambda Authorizer**: Validates authentication tokens (e.g., bearer tokens in the `Authorization` header).
- **MCP Server Lambda**: Implements MCP tools and session logic using this library.
- **DynamoDB**: Stores session data (if using the DynamoDB session backend).

## Development

1. Clone the repository:
```bash
git clone https://github.com/awslabs/mcp.git
cd mcp/src/mcp-lambda-handler
```

2. Install development dependencies:
```bash
pip install -e .[dev]
```

3. Run tests:
```bash
pytest
```

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](../../CONTRIBUTING.md) in the monorepo root for guidelines.

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

## Python Version Support

- Python 3.10+

## Dependencies

Core dependencies:
- python-dateutil >= 2.8.2

Optional dependencies:
- boto3 >= 1.38.1 (for AWS/DynamoDB support)
- botocore >= 1.38.1 (for AWS/DynamoDB support)

Development dependencies:
- pytest >= 8.0.0
- black >= 24.2.0
- isort >= 5.13.0
- flake8 >= 7.0.0
- moto >= 5.0.3 (for AWS mocking in tests)
