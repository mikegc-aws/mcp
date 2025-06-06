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
"""Tests for the sam_deploy module."""

import os
import pytest
import subprocess
import tempfile
from awslabs.aws_serverless_mcp_server.tools.sam.sam_deploy import SamDeployTool
from unittest.mock import AsyncMock, MagicMock, patch


class TestSamDeploy:
    """Tests for the sam_deploy function."""

    @pytest.mark.asyncio
    async def test_sam_deploy_success(self):
        """Test successful SAM deployment."""
        # Mock the subprocess.run function
        mock_result = MagicMock()
        mock_result.stdout = b'Successfully deployed SAM project'
        mock_result.stderr = b''

        with patch(
            'awslabs.aws_serverless_mcp_server.tools.sam.sam_deploy.run_command',
            return_value=(mock_result.stdout, mock_result.stderr),
        ) as mock_run:
            # Call the function
            result = await SamDeployTool(MagicMock(), True).handle_sam_deploy(
                AsyncMock(),
                application_name='test-app',
                project_directory=os.path.join(tempfile.gettempdir(), 'test-project'),
                template_file=None,
                s3_bucket=None,
                s3_prefix=None,
                region=None,
                profile=None,
                parameter_overrides=None,
                capabilities=None,
                config_file=None,
                config_env=None,
                metadata=None,
                tags=None,
                resolve_s3=False,
                debug=False,
            )

            # Verify the result
            assert result['success'] is True
            assert 'SAM project deployed successfully' in result['message']
            assert result['output'] == 'Successfully deployed SAM project'

            # Verify run_command was called with the correct arguments
            mock_run.assert_called_once()
            args, kwargs = mock_run.call_args
            cmd = args[0]

            # Check required parameters
            assert 'sam' in cmd
            assert 'deploy' in cmd
            assert '--stack-name' in cmd
            assert 'test-app' in cmd
            assert kwargs['cwd'] == os.path.join(tempfile.gettempdir(), 'test-project')

    @pytest.mark.asyncio
    async def test_sam_deploy_with_optional_params(self):
        """Test SAM deployment with optional parameters."""
        # Mock the subprocess.run function
        mock_result = MagicMock()
        mock_result.stdout = b'Successfully deployed SAM project'
        mock_result.stderr = b''

        with patch(
            'awslabs.aws_serverless_mcp_server.tools.sam.sam_deploy.run_command',
            return_value=(mock_result.stdout, mock_result.stderr),
        ) as mock_run:
            # Call the function
            result = await SamDeployTool(MagicMock(), True).handle_sam_deploy(
                AsyncMock(),
                application_name='test-app',
                project_directory=os.path.join(tempfile.gettempdir(), 'test-project'),
                template_file='template.yaml',
                s3_bucket='my-bucket',
                s3_prefix='my-prefix',
                region='us-west-2',
                profile='default',
                parameter_overrides='ParameterKey=Key1,ParameterValue=Value1',
                capabilities=['CAPABILITY_IAM', 'CAPABILITY_NAMED_IAM'],
                config_file='samconfig.toml',
                config_env='dev',
                metadata={'key1': 'value1', 'key2': 'value2'},
                tags={'tag1': 'value1', 'tag2': 'value2'},
                resolve_s3=True,
                debug=True,
            )

            # Verify the result
            assert result['success'] is True

            # Verify run_command was called with the correct arguments
            mock_run.assert_called_once()
            args, kwargs = mock_run.call_args
            cmd = args[0]

            # Check optional parameters
            assert '--template-file' in cmd
            assert 'template.yaml' in cmd
            assert '--s3-bucket' in cmd
            assert 'my-bucket' in cmd
            assert '--s3-prefix' in cmd
            assert 'my-prefix' in cmd
            assert '--region' in cmd
            assert 'us-west-2' in cmd
            assert '--profile' in cmd
            assert 'default' in cmd
            assert '--parameter-overrides' in cmd
            assert '--capabilities' in cmd
            assert 'CAPABILITY_IAM' in cmd
            assert 'CAPABILITY_NAMED_IAM' in cmd
            assert '--no-confirm-changeset' in cmd
            assert '--config-file' in cmd
            assert 'samconfig.toml' in cmd
            assert '--config-env' in cmd
            assert 'dev' in cmd
            assert '--metadata' in cmd
            assert '--tags' in cmd
            assert '--resolve-s3' in cmd
            assert '--debug' in cmd

    @pytest.mark.asyncio
    async def test_sam_deploy_failure(self):
        """Test SAM deployment failure."""
        # Mock the subprocess.run function to raise an exception
        error_message = b'Command failed with exit code 1'
        with patch(
            'awslabs.aws_serverless_mcp_server.tools.sam.sam_deploy.run_command',
            side_effect=subprocess.CalledProcessError(1, 'sam deploy', stderr=error_message),
        ):
            # Call the function
            result = await SamDeployTool(MagicMock(), True).handle_sam_deploy(
                AsyncMock(),
                application_name='test-app',
                project_directory=os.path.join(tempfile.gettempdir(), 'test-project'),
                template_file=None,
                s3_bucket=None,
                s3_prefix=None,
                region=None,
                profile=None,
                parameter_overrides=None,
                capabilities=None,
                config_file=None,
                config_env=None,
                metadata=None,
                tags=None,
                resolve_s3=False,
                debug=False,
            )

            # Verify the result
            assert result['success'] is False
            assert 'Failed to deploy SAM project' in result['message']
            assert 'Command failed with exit code 1' in result['message']

    @pytest.mark.asyncio
    async def test_sam_deploy_general_exception(self):
        """Test SAM deployment with a general exception."""
        # Mock the subprocess.run function to raise a general exception
        error_message = 'Some unexpected error'
        with patch(
            'awslabs.aws_serverless_mcp_server.tools.sam.sam_deploy.run_command',
            side_effect=Exception(error_message),
        ):
            # Call the function
            result = await SamDeployTool(MagicMock(), True).handle_sam_deploy(
                AsyncMock(),
                application_name='test-app',
                project_directory=os.path.join(tempfile.gettempdir(), 'test-project'),
                template_file=None,
                s3_bucket=None,
                s3_prefix=None,
                region=None,
                profile=None,
                parameter_overrides=None,
                capabilities=None,
                config_file=None,
                config_env=None,
                metadata=None,
                tags=None,
                resolve_s3=False,
                debug=False,
            )

            # Verify the result
            assert result['success'] is False
            assert 'Failed to deploy SAM project' in result['message']
            assert error_message in result['message']

    @pytest.mark.asyncio
    async def test_sam_deploy_allow_write_false(self):
        """Test SAM deployment when allow_write is False."""
        # Create the tool with allow_write set to False
        tool = SamDeployTool(MagicMock(), allow_write=False)

        # Call the function
        with pytest.raises(Exception) as exc_info:
            await tool.handle_sam_deploy(
                AsyncMock(),
                application_name='test-app',
                project_directory=os.path.join(tempfile.gettempdir(), 'test-project'),
                template_file=None,
                s3_bucket=None,
                s3_prefix=None,
                region=None,
                profile=None,
                parameter_overrides=None,
                capabilities=None,
                config_file=None,
                config_env=None,
                metadata=None,
                tags=None,
                resolve_s3=False,
                debug=False,
            )

        # Verify the exception message
        assert (
            'Write operations are not allowed. Set --allow-write flag to true to enable write operations.'
            in str(exc_info.value)
        )
