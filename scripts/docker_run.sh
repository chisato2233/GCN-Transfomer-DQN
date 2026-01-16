#!/bin/bash
# =============================================================================
# SAGIN Intelligent Routing - Docker Helper for Linux/macOS
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    echo -e "${GREEN}[$1]${NC} $2"
}

print_warn() {
    echo -e "${YELLOW}[$1]${NC} $2"
}

print_error() {
    echo -e "${RED}[$1]${NC} $2"
}

case "$1" in
    build)
        print_msg "BUILD" "Building Docker image..."
        docker-compose build
        ;;

    train)
        print_msg "TRAIN" "Starting training..."
        docker-compose run --rm sagin python src/experiments/train.py --config configs/routing_config.yaml
        ;;

    shell)
        print_msg "SHELL" "Opening interactive shell..."
        docker-compose run --rm sagin bash
        ;;

    test)
        print_msg "TEST" "Testing GPU and imports..."
        docker-compose run --rm sagin python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

print()
print('Testing imports...')
from src.env import SAGINRoutingEnv
from src.models import GCNEncoder, TemporalTransformer
from src.agents import DQNAgent
print('All imports successful!')

print()
print('Testing environment...')
config = {'network': {'num_satellites': 3, 'num_uavs': 6, 'num_ground': 10}}
env = SAGINRoutingEnv(config)
obs, info = env.reset()
print(f'State dim: {obs[\"state\"].shape}')
print(f'Action space: {env.action_space}')
print('Environment OK!')
"
        ;;

    jupyter)
        print_msg "JUPYTER" "Starting Jupyter Lab on port 8888..."
        echo "Access at: http://localhost:8888"
        docker-compose --profile jupyter up jupyter
        ;;

    tensorboard)
        print_msg "TENSORBOARD" "Starting TensorBoard on port 6006..."
        echo "Access at: http://localhost:6006"
        docker-compose --profile tensorboard up tensorboard
        ;;

    all)
        print_msg "ALL" "Starting training with Jupyter and TensorBoard..."
        docker-compose --profile all up
        ;;

    stop)
        print_warn "STOP" "Stopping all containers..."
        docker-compose --profile all down
        ;;

    clean)
        print_error "CLEAN" "Removing all containers and images..."
        docker-compose --profile all down --rmi all --volumes
        ;;

    logs)
        print_msg "LOGS" "Showing training logs..."
        docker-compose logs -f sagin
        ;;

    *)
        echo "============================================================================="
        echo "SAGIN Intelligent Routing - Docker Helper"
        echo "============================================================================="
        echo ""
        echo "Usage: $0 {command}"
        echo ""
        echo "Commands:"
        echo "  build       - Build Docker image"
        echo "  train       - Start training (default config)"
        echo "  shell       - Open interactive bash shell"
        echo "  test        - Test GPU and imports"
        echo "  jupyter     - Start Jupyter Lab (port 8888)"
        echo "  tensorboard - Start TensorBoard (port 6006)"
        echo "  all         - Start training + Jupyter + TensorBoard"
        echo "  stop        - Stop all containers"
        echo "  clean       - Remove all containers and images"
        echo "  logs        - Show training logs"
        echo ""
        echo "Examples:"
        echo "  $0 build"
        echo "  $0 train"
        echo "  $0 jupyter"
        echo "============================================================================="
        ;;
esac
