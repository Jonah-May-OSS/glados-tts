"""Integration tests for GLaDOS TTS engine (requires GPU and models)."""

import pytest
import torch
from pathlib import Path
from glados import TTSRunner
from pydub import AudioSegment


# Mark all tests in this module as requiring GPU
pytestmark = pytest.mark.gpu


class TestTTSRunner:
    """Integration tests for TTSRunner."""

    @pytest.fixture(scope="class")
    def tts_runner(self, models_dir):
        """Create a TTSRunner instance for testing."""
        # Skip if models are not available
        if not models_dir.exists():
            pytest.skip("Models directory not found")

        required_files = [
            models_dir / "glados-new.pt",
            models_dir / "vocoder-gpu.pt",
            models_dir / "emb" / "glados_p2.pt",
        ]

        for file_path in required_files:
            if not file_path.exists():
                pytest.skip(f"Required model file not found: {file_path}")

        return TTSRunner(use_p1=False, log=True, models_dir=models_dir)

    def test_tts_runner_initialization(self, tts_runner):
        """Test that TTSRunner initializes correctly."""
        assert tts_runner is not None
        assert tts_runner.initialized is True
        assert tts_runner.device is not None
        assert tts_runner.glados is not None
        assert tts_runner.vocoder is not None
        assert tts_runner.emb is not None

    def test_tts_runner_device_selection(self, tts_runner):
        """Test that appropriate device is selected."""
        device = tts_runner.device
        # Should be CUDA if available, MPS if on Apple Silicon, otherwise CPU
        if torch.cuda.is_available():
            assert device.type == "cuda"
        elif torch.backends.mps.is_available():
            assert device.type == "mps"
        else:
            assert device.type == "cpu"

    def test_run_tts_basic(self, tts_runner):
        """Test basic TTS generation."""
        text = "Hello world."
        audio = tts_runner.run_tts(text)

        assert isinstance(audio, AudioSegment)
        assert audio.frame_rate == 22050
        assert audio.sample_width == 2
        assert audio.channels == 1
        assert len(audio) > 0

    def test_run_tts_various_texts(self, tts_runner, sample_texts):
        """Test TTS generation with various text inputs."""
        for text in sample_texts:
            audio = tts_runner.run_tts(text)
            assert isinstance(audio, AudioSegment)
            assert len(audio) > 0
            # Longer text should generally produce longer audio
            assert len(audio) > 100  # At least 100ms

    def test_run_tts_with_alpha(self, tts_runner):
        """Test TTS generation with different alpha values."""
        text = "Testing different alpha values."

        for alpha in [0.8, 1.0, 1.2]:
            audio = tts_runner.run_tts(text, alpha=alpha)
            assert isinstance(audio, AudioSegment)
            assert len(audio) > 0

    def test_run_tts_long_text(self, tts_runner):
        """Test TTS generation with longer text."""
        text = (
            "This is a longer piece of text that should test the model's "
            "ability to handle extended input. The quick brown fox jumps "
            "over the lazy dog. Testing, testing, one two three."
        )
        audio = tts_runner.run_tts(text)

        assert isinstance(audio, AudioSegment)
        assert len(audio) > 1000  # Should be at least 1 second

    def test_run_tts_special_characters(self, tts_runner):
        """Test TTS with special characters and punctuation."""
        texts = [
            "Hello, world!",
            "What is your name?",
            "Testing... one, two, three.",
            "Numbers: 1, 2, 3, 4, 5.",
        ]

        for text in texts:
            audio = tts_runner.run_tts(text)
            assert isinstance(audio, AudioSegment)
            assert len(audio) > 0

    def test_tts_runner_p1_vs_p2(self, models_dir):
        """Test both Portal 1 and Portal 2 voice variants."""
        # Check if P1 embedding exists
        p1_emb = models_dir / "emb" / "glados_p1.pt"
        if not p1_emb.exists():
            pytest.skip("Portal 1 embedding not found")

        runner_p1 = TTSRunner(use_p1=True, log=True, models_dir=models_dir)
        runner_p2 = TTSRunner(use_p1=False, log=True, models_dir=models_dir)

        text = "Testing voice variants."
        audio_p1 = runner_p1.run_tts(text)
        audio_p2 = runner_p2.run_tts(text)

        # Both should produce valid audio
        assert isinstance(audio_p1, AudioSegment)
        assert isinstance(audio_p2, AudioSegment)
        assert len(audio_p1) > 0
        assert len(audio_p2) > 0

    def test_tts_consistency(self, tts_runner):
        """Test that same text produces consistent output length."""
        text = "Consistency test."

        audio1 = tts_runner.run_tts(text)
        audio2 = tts_runner.run_tts(text)

        # Length should be similar (within 10%)
        len_diff = abs(len(audio1) - len(audio2))
        max_allowed_diff = max(len(audio1), len(audio2)) * 0.1
        assert len_diff <= max_allowed_diff

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="TensorRT compilation requires CUDA"
    )
    def test_tensorrt_compilation(self, tts_runner):
        """Test that TensorRT compilation works (if applicable)."""
        # This test checks if TRT engines are created/loaded
        # The taco_trt and voco_trt flags indicate if TRT is being used
        assert hasattr(tts_runner, "taco_trt")
        assert hasattr(tts_runner, "voco_trt")
        # Just ensure the flags are boolean
        assert isinstance(tts_runner.taco_trt, bool)
        assert isinstance(tts_runner.voco_trt, bool)

    def test_workspace_size_calculation(self, tts_runner):
        """Test workspace size calculation for TensorRT."""
        workspace_size = tts_runner._get_workspace_size()

        assert isinstance(workspace_size, int)
        assert workspace_size > 0
        # Should be between 1GB and 4GB
        assert workspace_size >= 1 * 1024 * 1024 * 1024
        assert workspace_size <= 4 * 1024 * 1024 * 1024
