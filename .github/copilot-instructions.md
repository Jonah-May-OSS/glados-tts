# Copilot Instructions for GLaDOS TTS

## Project Overview
This is a neural network-based Text-to-Speech (TTS) engine that generates GLaDOS voice from Portal games. The project uses PyTorch for deep learning models including Tacotron and HiFiGAN vocoder.

## Key Technologies
- **Python 3.x**: Primary language
- **PyTorch**: Deep learning framework for TTS models
- **Flask**: Web API framework for remote TTS processing
- **Deep Phonemizer**: Text preprocessing and phoneme conversion
- **NLTK**: Natural language processing

## Architecture
- `glados.py`: Main TTS engine and runner
- `engine.py`: Remote API server
- `utils/`: Text processing and utility functions
- `models/`: Pre-trained neural network models (Tacotron, HiFiGAN)

## Code Style and Patterns
- Use type hints for function parameters and return values
- Follow PEP 8 style guidelines
- Use logging module for debug and info messages instead of print statements
- Prefer pathlib.Path over os.path for file operations
- Use torch.device for device-agnostic code (CPU/CUDA/MPS)

## Testing and Validation
- Test TTS generation with various text inputs
- Verify model loading and initialization
- Check device compatibility (CPU, CUDA, MPS)
- Validate API endpoints if modifying engine.py

## Dependencies
- Manage dependencies via `requirements.txt`
- Pin versions with `~=` for stability
- Check for Python version-specific dependencies (e.g., audioop-lts for Python 3.13+)

## Common Tasks
- **Model optimization**: Use torch.jit for model compilation
- **Audio processing**: Use pydub for audio manipulation
- **Text preprocessing**: Use deep_phonemizer and custom utils
- **Device handling**: Support CPU, CUDA, and MPS (Apple Silicon)

## Important Notes
- Models are large and loaded from external sources (Google Drive)
- The engine supports both standalone and remote API modes
- Multispeaker models trained on LJSpeech and Ellen McClain datasets
- Portal 1 and Portal 2 voice variants are supported
