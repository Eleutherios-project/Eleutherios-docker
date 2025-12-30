@echo off
echo ==================================================
echo   Aegis Insight - Windows Setup
echo ==================================================
echo.

REM Check Docker
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running
    echo Please start Docker Desktop
    pause
    exit /b 1
)

echo Docker is ready

REM Start services
echo.
echo Starting services...
docker-compose up -d

echo.
echo Waiting for services...
timeout /t 10 /nobreak

echo.
echo ==================================================
echo   Setup Complete!
echo ==================================================
echo.
echo   Web Interface:  http://localhost:8001
echo   Neo4j Browser:  http://localhost:7474
echo.
echo   View logs:      docker-compose logs -f
echo   Stop:           docker-compose down
echo.
pause
