#!/bin/bash
# Test runner script for GLaDOS TTS

set -e

echo "GLaDOS TTS Test Runner"
echo "======================"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected. Consider using a venv."
    echo ""
fi

# Install dependencies if needed
if ! python -c "import pytest" 2>/dev/null; then
    echo "📦 Installing test dependencies..."
    pip install -q -r requirements-dev.txt
fi

# Parse arguments
TEST_TYPE="${1:-all}"
MARKERS=""

case $TEST_TYPE in
    unit)
        echo "🧪 Running unit tests (no GPU required)..."
        MARKERS="-m 'not gpu'"
        ;;
    gpu)
        echo "🎮 Running GPU integration tests..."
        MARKERS="-m gpu"
        # Check for CUDA
        if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            echo "❌ Error: GPU tests require CUDA. No CUDA device found."
            exit 1
        fi
        ;;
    all)
        echo "🧪 Running all tests..."
        MARKERS=""
        ;;
    *)
        echo "Usage: $0 [unit|gpu|all]"
        echo ""
        echo "  unit - Run only unit tests (no GPU required)"
        echo "  gpu  - Run only GPU integration tests (requires CUDA)"
        echo "  all  - Run all tests (default)"
        exit 1
        ;;
esac

# Run tests
echo ""
eval "pytest tests/ -v --tb=short $MARKERS"

# Show coverage summary
echo ""
echo "✅ Tests completed successfully!"
echo ""
echo "📊 Coverage report available at: htmlcov/index.html"
