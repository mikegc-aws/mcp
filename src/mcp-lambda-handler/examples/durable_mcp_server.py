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

"""Example: MCP server with durable (long-running) tools.

This example shows a single Lambda function that serves both regular MCP tools
and durable task tools. The Lambda must be deployed with DurableConfig enabled.

Deployment requirements:
  - Python 3.13 runtime (durable SDK pre-installed)
  - DurableConfig enabled on the Lambda function
  - DynamoDB table for async task state (if using invoke_mode='async')
  - Environment variable MCP_TASK_FUNCTION_NAME set to the function's qualified ARN

SAM template snippet:
  DurableMcpFunction:
    Type: AWS::Serverless::Function
    Properties:
      Runtime: python3.13
      Handler: durable_mcp_server.handler
      Timeout: 900
      DurableConfig:
        ExecutionTimeout: 86400
        RetentionPeriodInDays: 14
      Environment:
        Variables:
          MCP_TASK_FUNCTION_NAME: !Sub "${DurableMcpFunction}:$LATEST"
          MCP_TASK_TABLE: !Ref TaskTable
      Policies:
        - DynamoDBCrudPolicy:
            TableName: !Ref TaskTable
        - LambdaInvokePolicy:
            FunctionName: !Ref DurableMcpFunction

  TaskTable:
    Type: AWS::Serverless::SimpleTable
    Properties:
      PrimaryKey:
        Name: session_id
        Type: String
"""

from awslabs.mcp_lambda_handler import MCPLambdaHandler, create_durable_handler, task_tool


mcp = MCPLambdaHandler(name='data-pipeline', version='1.0.0')


# --- Regular tool (runs inline, fast) ---


@mcp.tool()
def list_datasets(prefix: str) -> str:
    """List available datasets by prefix.

    Args:
        prefix: S3 key prefix to filter datasets
    """
    import boto3

    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket='my-datasets', Prefix=prefix, MaxKeys=20)
    keys = [obj['Key'] for obj in response.get('Contents', [])]
    return '\n'.join(keys) if keys else 'No datasets found'


# --- Sync task tool (blocks up to 15 min, returns result) ---


@task_tool(mcp)
def analyze_dataset(dataset_key: str, context) -> str:
    """Analyze a dataset with multiple checkpointed steps.

    Runs as a durable execution. Each step is checkpointed — if the function
    is interrupted, it resumes from the last completed step.

    Args:
        dataset_key: S3 key of the dataset to analyze
    """
    from aws_durable_execution_sdk_python import durable_step
    from aws_durable_execution_sdk_python.types import StepContext

    @durable_step
    def download(ctx: StepContext) -> str:
        import boto3

        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket='my-datasets', Key=dataset_key)
        return obj['Body'].read().decode('utf-8')

    @durable_step
    def compute_stats(ctx: StepContext, data: str) -> dict:
        lines = data.strip().split('\n')
        return {
            'rows': len(lines),
            'columns': len(lines[0].split(',')) if lines else 0,
            'size_bytes': len(data),
        }

    @durable_step
    def generate_summary(ctx: StepContext, stats: dict) -> str:
        return (
            f"Dataset: {dataset_key}\n"
            f"Rows: {stats['rows']}\n"
            f"Columns: {stats['columns']}\n"
            f"Size: {stats['size_bytes']} bytes"
        )

    data = context.step(download())
    stats = context.step(compute_stats(data))
    summary = context.step(generate_summary(stats))
    return summary


# --- Async task tool (returns immediately, client polls for result) ---


@task_tool(mcp, invoke_mode='async')
def train_model(model_name: str, dataset_key: str, epochs: int, context) -> str:
    """Train an ML model on a dataset.

    This can take hours. Returns immediately with a task ID.
    Use get_task_status to poll for completion.

    Args:
        model_name: Name for the trained model
        dataset_key: S3 key of the training dataset
        epochs: Number of training epochs
    """
    from aws_durable_execution_sdk_python import durable_step
    from aws_durable_execution_sdk_python.types import StepContext

    @durable_step
    def prepare_data(ctx: StepContext) -> str:
        # Download and preprocess training data
        return f's3://processed/{dataset_key}'

    @durable_step
    def run_training(ctx: StepContext, processed_path: str) -> str:
        # Actual training logic — each epoch could be a separate step
        # for finer checkpointing
        return f's3://models/{model_name}/final'

    @durable_step
    def register_model(ctx: StepContext, model_path: str) -> str:
        # Register in model registry
        return f'model:{model_name}:v1'

    processed = context.step(prepare_data())
    model_path = context.step(run_training(processed))

    # Wait 30 seconds between training and registration (zero compute cost)
    context.wait(seconds=30)

    model_id = context.step(register_model(model_path))
    return model_id


# --- Lambda handler entry point ---

handler = create_durable_handler(mcp)
