# GLaDOS Text-to-speech (TTS) Voice Generator
Neural network based TTS Engine.

If you want to just play around with the TTS, this works as stand-alone.
```console
python3 glados-tts/glados.py
```

the TTS Engine can also be used remotely on a machine more powerful then the Pi to process in house TTS: (executed from glados-tts directory
```console
python3 engine-remote.py
```

Default port is 8124
Be sure to update settings.env variable in your main Glados-voice-assistant directory:
```
TTS_ENGINE_API			= http://192.168.1.3:8124/synthesize/
```


## Training (New Model)
The Tacotron and ForwardTacotron models were trained as multispeaker models on two datasets separated into three speakers. LJSpeech (13,100 lines), and then on the heavily modified version of the Ellen McClain dataset, separated into Portal 1 and 2 voices (with punctuation and corrections added manually). The lines from the end of Portal 1 after the cores get knocked off were counted as Portal 2 lines.


## Training (Old Model)
The initial, regular Tacotron model was trained first on LJSpeech, and then on a heavily modified version of the Ellen McClain dataset (all non-Portal 2 voice lines removed, punctuation added).

* The Forward Tacotron model was only trained on about 600 voice lines.
* The HiFiGAN model was generated through transfer learning from the sample.
* All models have been optimized and quantized.



## Installation Instruction
If you want to install the TTS Engine on your machine, please follow the steps
below.

1. Download the model files from [`Google Drive`](https://drive.google.com/file/d/1TRJtctjETgVVD5p7frSVPmgw8z8FFtjD/view?usp=sharing) and unzip into the repo folder
2. Install the required Python packages, e.g., by running `pip install -r
   requirements.txt`

## Testing

This project includes automated tests to ensure code quality and functionality.

### Running Tests Locally

**Quick Start:**
Use the provided test runner script:
```bash
./run_tests.sh          # Run all tests
./run_tests.sh unit     # Run only unit tests (no GPU)
./run_tests.sh gpu      # Run only GPU tests
```

**Manual Test Execution:**

1. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. Run all tests:
   ```bash
   pytest tests/ -v
   ```

3. Run only unit tests (no GPU required):
   ```bash
   pytest tests/test_utils.py -v -m "not gpu"
   ```

4. Run GPU integration tests (requires CUDA and model files):
   ```bash
   pytest tests/test_tts_integration.py -v -m gpu
   ```

5. Run tests with coverage report:
   ```bash
   pytest tests/ -v --cov=. --cov-report=html
   ```

### Continuous Integration

The project uses GitHub Actions for automated testing:

- **GPU Integration Tests**: Run on self-hosted runners with CUDA support
- **Coverage Reports**: Generated and uploaded to Codecov

Tests run automatically on:
- Push to main branch
- Pull requests to main branch
- Manual workflow dispatch

### Test Organization

- `tests/test_utils.py`: Unit tests for utility functions
- `tests/test_tts_integration.py`: Integration tests for TTS engine (requires GPU)
- `tests/test_engine_api.py`: Tests for Flask API endpoints

Tests marked with `@pytest.mark.gpu` require a CUDA-capable GPU and model files.
