# Automated Testing Setup Guide

This document provides information about the automated testing infrastructure implemented for the GLaDOS TTS project.

## Overview

The project now includes:
- **Unit tests** for utility functions (no GPU required)
- **Integration tests** for TTS engine (GPU required)
- **GitHub Actions workflows** for automated testing
- **Test runner script** for local development

## Test Structure

```
tests/
├── __init__.py              # Package marker
├── conftest.py             # Shared fixtures and configuration
├── test_utils.py           # Unit tests for text utilities (6 tests)
├── test_tts_integration.py # GPU integration tests (11 tests)
└── test_engine_api.py      # API tests (placeholder)
```

## Running Tests Locally

### Quick Start
```bash
# Run all tests
./run_tests.sh

# Run only unit tests (no GPU)
./run_tests.sh unit

# Run only GPU tests
./run_tests.sh gpu
```

### Manual Execution
```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run specific test types
pytest tests/test_utils.py -v -m "not gpu"  # Unit tests
pytest tests/test_tts_integration.py -v -m gpu  # GPU tests
pytest tests/ -v  # All tests
```

## GitHub Actions Workflows

The `.github/workflows/tests.yml` workflow includes two jobs:

### 1. GPU Integration Tests
- Runs on: `self-hosted` (requires setup)
- Tests: Full TTS pipeline tests
- Purpose: Validate GPU functionality and model inference

### 2. All Tests Summary
- Runs on: `self-hosted` (requires setup)
- Tests: Complete test suite with coverage
- Purpose: Generate comprehensive coverage reports

## Setting Up Self-Hosted Runner

To run GPU tests, you need to set up a self-hosted GitHub Actions runner with CUDA support.

### Prerequisites
- Ubuntu 20.04+ or compatible Linux distribution
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.x or 12.x installed
- Docker (optional but recommended)

### Setup Steps

1. **Navigate to Repository Settings**
   - Go to: `https://github.com/Jonah-May-OSS/glados-tts/settings/actions/runners`
   - Click "New self-hosted runner"

2. **Follow GitHub's Instructions**
   ```bash
   # Download and extract the runner
   mkdir actions-runner && cd actions-runner
   curl -o actions-runner-linux-x64-2.311.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
   tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz
   
   # Configure the runner
   ./config.sh --url https://github.com/Jonah-May-OSS/glados-tts --token YOUR_TOKEN
   
   # Install as a service (optional)
   sudo ./svc.sh install
   sudo ./svc.sh start
   ```

3. **Install Dependencies on Runner**
   ```bash
   # Install Python and CUDA
   sudo apt update
   sudo apt install python3 python3-pip nvidia-cuda-toolkit
   
   # Verify CUDA installation
   nvidia-smi
   ```

4. **Place Model Files**
   - Download models from Google Drive
   - Extract to: `~/glados-tts-models/`
   - The workflow expects models at: `models/` relative to repo root
   - Consider symlinking: `ln -s ~/glados-tts-models models`

5. **Install torch_tensorrt (Optional)**
   ```bash
   pip install torch-tensorrt
   ```
   This enables TensorRT optimizations. Tests work without it but performance is better with it.

### Runner Labels
The workflow uses the `self-hosted` label. No additional labels are required, but you can add custom labels during setup if needed.

## Optional: Codecov Integration

To enable coverage reporting to Codecov:

1. Sign up at https://codecov.io
2. Add your repository
3. Get the upload token
4. Add as repository secret:
   - Go to: Repository Settings → Secrets → Actions
   - Create new secret: `CODECOV_TOKEN`
   - Paste your Codecov token

If this secret is not set, coverage upload will be skipped (workflow will still pass).

## Test Development

### Adding New Tests

1. **Unit Tests** (no GPU):
   - Add to `tests/test_utils.py` or create new file
   - No special markers needed
   - Should run fast (< 1 second per test)

2. **GPU Tests**:
   - Add to `tests/test_tts_integration.py` or create new file
   - Mark with `@pytest.mark.gpu` decorator
   - Can be slower (model loading + inference)

### Example Test
```python
import pytest

class TestMyFeature:
    def test_basic_functionality(self):
        # Unit test - runs on CPU
        result = my_function("input")
        assert result == "expected"
    
    @pytest.mark.gpu
    def test_gpu_functionality(self, tts_runner):
        # Integration test - needs GPU
        audio = tts_runner.run_tts("test text")
        assert audio is not None
```

## Troubleshooting

### Tests Fail with "No module named torch_tensorrt"
- This is expected on CPU-only systems
- The code handles this gracefully
- GPU tests will skip TensorRT optimization

### GPU Tests Fail with "Models not found"
- Ensure model files are present in `models/` directory
- Download from: [Google Drive link]
- Required files:
  - `models/glados-new.pt`
  - `models/vocoder-gpu.pt`
  - `models/emb/glados_p2.pt`
  - `models/en_us_cmudict_ipa_forward.pt`

### Runner Can't Find CUDA
- Verify with: `nvidia-smi`
- Install CUDA toolkit: `sudo apt install nvidia-cuda-toolkit`
- Set environment variables:
  ```bash
  export CUDA_HOME=/usr/local/cuda
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ```

## Maintenance

### Updating Dependencies
```bash
# Update test dependencies
pip install --upgrade pytest pytest-cov pytest-mock
pip freeze | grep pytest > requirements-dev.txt
```

### Running Specific Tests
```bash
# Run single test
pytest tests/test_utils.py::TestTextPreparation::test_prepare_text_basic -v

# Run with specific markers
pytest -m "gpu and not slow" -v

# Run with coverage
pytest --cov=. --cov-report=html tests/
```

## Performance Benchmarks

Typical test execution times:
- Unit tests: ~10 seconds (CPU only)
- GPU integration tests: ~2-5 minutes (includes model loading)
- Full suite: ~5-7 minutes (GPU + model warm-up)

## Future Enhancements

Potential improvements for the testing infrastructure:
- [ ] Add Flask API integration tests
- [ ] Add performance regression tests
- [ ] Add audio quality validation tests
- [ ] Implement test data caching for faster CI
- [ ] Add Docker-based testing environment
- [ ] Set up matrix testing for different CUDA versions
