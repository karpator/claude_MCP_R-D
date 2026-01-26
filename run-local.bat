@echo off
setlocal

REM --- Configuration ---
set IMAGE_NAME=elastic-mcp
set CONTAINER_NAME=elastic-mcp-container
set PORT=8000

REM --- Step 1: Check if container is already running ---
docker ps -q -f name=%CONTAINER_NAME% >nul 2>&1
if %errorlevel% equ 0 (
    echo Stopping existing container...
    docker stop %CONTAINER_NAME% >nul 2>&1
    docker rm %CONTAINER_NAME% >nul 2>&1
)

REM --- Step 2: Set GCP credentials path ---
set "GCP_CREDENTIALS_HOST_PATH=%APPDATA%\gcloud\application_default_credentials.json"

if not exist "%GCP_CREDENTIALS_HOST_PATH%" (
    echo WARNING: GCP credentials file not found at "%GCP_CREDENTIALS_HOST_PATH%"
    echo The container will try to use other authentication methods.
    echo To set up credentials, run: gcloud auth application-default login
    echo.
    set USE_CREDENTIALS=false
) else (
    echo Found GCP credentials at: %GCP_CREDENTIALS_HOST_PATH%
    set USE_CREDENTIALS=true
)

REM --- Step 3: Run the container ---
echo Starting container '%CONTAINER_NAME%' from image '%IMAGE_NAME%'...

if "%USE_CREDENTIALS%"=="true" (
    REM Run with mounted credentials
    docker run -d ^
        --name %CONTAINER_NAME% ^
        -p %PORT%:%PORT% ^
        -v "%GCP_CREDENTIALS_HOST_PATH%:/app/gcp-key.json:ro" ^
        -e GOOGLE_APPLICATION_CREDENTIALS=/app/gcp-key.json ^
        -e GoogleAuthenticationMethod=application_default ^
        -e GoogleProjectId=hu0092-bus-t-ai ^
        %IMAGE_NAME%
) else (
    REM Run without credentials (will fail if GCP access is required)
    docker run -d ^
        --name %CONTAINER_NAME% ^
        -p %PORT%:%PORT% ^
        -e GoogleAuthenticationMethod=application_default ^
        -e GoogleProjectId=hu0092-bus-t-ai ^
        %IMAGE_NAME%
)

if %errorlevel% neq 0 (
    echo ERROR: Failed to start container.
    exit /b 1
)

echo Container started successfully!
echo.
echo Container name: %CONTAINER_NAME%
echo Port: %PORT%
echo.
echo To view logs: docker logs %CONTAINER_NAME%
echo To stop: docker stop %CONTAINER_NAME%
echo To remove: docker rm %CONTAINER_NAME%
echo.
echo Access the service at: http://localhost:%PORT%

REM Wait a moment and show initial logs
timeout /t 2 /nobreak >nul
echo.
echo === Initial Container Logs ===
docker logs %CONTAINER_NAME%

REM Usage:
REM   run-local.bat

