@echo off
setlocal

REM --- Configuration ---
set IMAGE_NAME=elastic-mcp


REM --- Step 2: Enable Docker BuildKit ---
set DOCKER_BUILDKIT=1

REM --- Step 3: Resolve the path to GCP credentials ---
REM The gcloud CLI stores credentials in the %APPDATA%\gcloud directory on Windows.
set "GCP_CREDENTIALS_HOST_PATH=%APPDATA%\gcloud\application_default_credentials.json"
set "GCP_CREDENTIALS_CONTAINER_PATH=/root/gcloud_credentials.json"

REM Check if the credentials file exists
if not exist "%GCP_CREDENTIALS_HOST_PATH%" (
    echo ERROR: GCP credentials file not found at "%GCP_CREDENTIALS_HOST_PATH%".
    echo Please run "gcloud auth application-default login" first.
    exit /b 1
)
echo Found GCP credentials at: %GCP_CREDENTIALS_HOST_PATH%

REM --- Step 4: Get the gcloud access token for the Docker build ---
echo Generating gcloud access token...
for /f "tokens=*" %%i in ('gcloud auth application-default print-access-token') do set ARTIFACT_REGISTRY_TOKEN=%%i

if not defined ARTIFACT_REGISTRY_TOKEN (
    echo ERROR: Failed to obtain gcloud access token.
    echo Please ensure you are logged in with gcloud.
    exit /b 1
)
echo Token generated successfully.

REM --- Step 5: Build the Docker image using build args ---
echo Building Docker image '%IMAGE_NAME%' using build args...
docker build ^
    --build-arg GCP_TOKEN=%ARTIFACT_REGISTRY_TOKEN% ^
    -t %IMAGE_NAME% .

if %errorlevel% neq 0 (
    echo ERROR: Docker build failed.
    exit /b 1
)
echo Docker image built successfully.

REM Usage:
REM   build-local.bat

