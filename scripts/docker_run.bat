@echo off
REM =============================================================================
REM SAGIN Intelligent Routing - Docker Helper for Windows
REM =============================================================================

setlocal enabledelayedexpansion

if "%1"=="build" (
    echo [BUILD] Building Docker image...
    docker-compose build
    goto :eof
)

if "%1"=="train" (
    echo [TRAIN] Starting training...
    docker-compose run --rm sagin python src/experiments/train.py --config configs/routing_config.yaml
    goto :eof
)

if "%1"=="shell" (
    echo [SHELL] Opening interactive shell...
    docker-compose run --rm --profile shell sagin bash
    goto :eof
)

if "%1"=="test" (
    echo [TEST] Testing GPU and imports...
    docker-compose run --rm sagin python -c ^
"import torch; ^
print(f'PyTorch version: {torch.__version__}'); ^
print(f'CUDA available: {torch.cuda.is_available()}'); ^
if torch.cuda.is_available(): print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB'); ^
print(); ^
print('Testing imports...'); ^
from src.env import SAGINRoutingEnv; ^
from src.models import GCNEncoder, TemporalTransformer; ^
from src.agents import DQNAgent; ^
print('All imports successful!'); ^
print(); ^
print('Testing environment...'); ^
config = {'network': {'num_satellites': 3, 'num_uavs': 6, 'num_ground': 10}}; ^
env = SAGINRoutingEnv(config); ^
obs, info = env.reset(); ^
print(f'State dim: {obs[\"state\"].shape}'); ^
print(f'Action space: {env.action_space}'); ^
print('Environment OK!');"
    goto :eof
)

if "%1"=="jupyter" (
    echo [JUPYTER] Starting Jupyter Lab on port 8888...
    echo Access at: http://localhost:8888
    docker-compose --profile jupyter up jupyter
    goto :eof
)

if "%1"=="tensorboard" (
    echo [TENSORBOARD] Starting TensorBoard on port 6006...
    echo Access at: http://localhost:6006
    docker-compose --profile tensorboard up tensorboard
    goto :eof
)

if "%1"=="all" (
    echo [ALL] Starting training with Jupyter and TensorBoard...
    docker-compose --profile all up
    goto :eof
)

if "%1"=="stop" (
    echo [STOP] Stopping all containers...
    docker-compose --profile all down
    goto :eof
)

if "%1"=="clean" (
    echo [CLEAN] Removing all containers and images...
    docker-compose --profile all down --rmi all --volumes
    goto :eof
)

if "%1"=="logs" (
    echo [LOGS] Showing training logs...
    docker-compose logs -f sagin
    goto :eof
)

echo =============================================================================
echo SAGIN Intelligent Routing - Docker Helper
echo =============================================================================
echo.
echo Usage: docker_run.bat {command}
echo.
echo Commands:
echo   build       - Build Docker image
echo   train       - Start training (default config)
echo   shell       - Open interactive bash shell
echo   test        - Test GPU and imports
echo   jupyter     - Start Jupyter Lab (port 8888)
echo   tensorboard - Start TensorBoard (port 6006)
echo   all         - Start training + Jupyter + TensorBoard
echo   stop        - Stop all containers
echo   clean       - Remove all containers and images
echo   logs        - Show training logs
echo.
echo Examples:
echo   docker_run.bat build
echo   docker_run.bat train
echo   docker_run.bat jupyter
echo =============================================================================
