"""Core GLaDOS TTS model runner and CLI test harness."""

import logging
import time
from pathlib import Path
from typing import Any, cast

import torch
import torch_tensorrt
from pydub import AudioSegment, playback
from dp.preprocessing.text import Preprocessor, LanguageTokenizer, SequenceTokenizer

from .utils.tools import (
    get_cleaner_and_tokenizer,
    prepare_text,
)

_LOGGER = logging.getLogger(__name__)

# Bucket sizes for Tacotron warm-ups

BUCKET_SIZES = [8, 16, 32]


class TTSRunner:
    """Text-to-Speech runner for GLaDOS TTS with streaming support and optional TRT Vocoder."""

    def __init__(
        self,
        use_p1: bool = False,
        log: bool = False,
        models_dir: Path = Path("models"),
    ):
        """
        Initialize TTS: load embedding, cast Tacotron to FP16,
        and load or compile TRT Vocoder.
        """
        self.log = log
        self.models_dir = models_dir
        self.fp16 = False
        self.taco_trt = False
        self.voco_trt = False
        self.initialized = False
        self.glados: Any = None
        self.vocoder: Any = None
        self.emb: torch.Tensor = torch.empty(0)

        # Device selection

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        _LOGGER.info("Using device: %s", self.device)

        # Only initialize if not already initialized

        if not self.initialized:
            self.initialize_models(use_p1)

    def _get_workspace_size(self) -> int:
        """
        Calculate TensorRT workspace size dynamically based on available VRAM.
        Returns workspace size in bytes.
        """
        default_workspace = 2 * 1024 * 1024 * 1024  # 2GB default
        min_workspace = 1 * 1024 * 1024 * 1024  # 1GB minimum
        max_workspace = 4 * 1024 * 1024 * 1024  # 4GB maximum

        if self.device.type != "cuda":
            _LOGGER.info(
                f"Non-CUDA device ({self.device.type}), using default workspace size: 2GB"
            )
            return default_workspace

        try:
            # Get total VRAM from the specific device being used
            device_index = self.device.index if self.device.index is not None else 0
            total_memory = torch.cuda.get_device_properties(device_index).total_memory
            # Use 40% of total VRAM as a conservative workspace size
            calculated_workspace = int(total_memory * 0.4)

            # Clamp to min/max bounds
            workspace_size = max(
                min_workspace, min(calculated_workspace, max_workspace)
            )

            _LOGGER.info(
                f"CUDA device {device_index}: Total VRAM: {total_memory / (1024**3):.2f}GB, "
                f"Setting workspace size to {workspace_size / (1024**3):.2f}GB"
            )
            return workspace_size
        except Exception as e:
            _LOGGER.warning(f"Failed to detect VRAM, using default workspace size: {e}")
            return default_workspace

    def initialize_models(self, use_p1: bool = False):
        """Initialize models and perform warm-up."""
        _LOGGER.info("Initializing GLaDOS TTS models...")

        # Safe globals for embedding deserialization

        torch.serialization.add_safe_globals(
            [Preprocessor, LanguageTokenizer, SequenceTokenizer]
        )

        # Initialize Cleaner and Tokenizer only once

        self.cleaner, self.tokenizer = get_cleaner_and_tokenizer(
            str(self.models_dir), str(self.device), "english_cleaners", "en_us", True
        )

        # Load speaker embedding

        emb_filename = "glados_p1.pt" if use_p1 else "glados_p2.pt"
        emb_path = self.models_dir / "emb" / emb_filename
        if not emb_path.is_file():
            raise FileNotFoundError(f"Embedding not found at {emb_path}")
        self.emb = torch.load(str(emb_path), map_location=self.device).to(self.device)

        # Model file paths

        tacotron_path = self.models_dir / "glados-new.pt"
        vocoder_path = self.models_dir / "vocoder-gpu.pt"
        trt_tacotron_path = self.models_dir / "tacotron-trt.ts"
        _LOGGER.debug("Looking for Tacotron at: %s", tacotron_path)
        trt_vocoder_path = self.models_dir / "vocoder-trt.ts"
        _LOGGER.debug("Looking for TRT vocoder at: %s", trt_vocoder_path)

        # Load TorchScript models

        base_tacotron = torch.jit.load(str(tacotron_path), map_location=self.device)
        base_vocoder = torch.jit.load(str(vocoder_path), map_location=self.device)

        # Load or compile TRT Tacotron with extended profile

        self.taco_trt = False
        if trt_tacotron_path.exists():
            _LOGGER.info("Loading TRT tacotron engine...")
            try:
                self.glados = (
                    torch.jit.load(str(trt_tacotron_path)).to(self.device).eval()
                )
                self.taco_trt = True
            except Exception as e:
                _LOGGER.error("Failed to load TRT tacotron: %s", e)
        if not self.taco_trt:
            _LOGGER.info("Compiling TRT tacotron...")
            try:
                # Compile with TensorRT backend through torch.compile.
                trt_mod = cast(
                    Any,
                    torch.compile(
                        base_tacotron.eval().to(self.device),
                        backend="tensorrt",
                    ),
                )
                trt_mod.save(str(trt_tacotron_path))
                self.glados = trt_mod.eval().to(self.device)
                self.taco_trt = True
            except Exception as e:
                _LOGGER.error("Failed to compile TRT tacotron: %s", e)
                self.glados = base_tacotron.to(self.device).eval()
        _LOGGER.info("Tacotron engine ready. TRT=%s", self.taco_trt)

        # Load or compile TRT Vocoder with extended profile

        self.voco_trt = False
        if trt_vocoder_path.exists():
            _LOGGER.info("Loading TRT vocoder engine...")
            try:
                self.vocoder = (
                    torch.jit.load(str(trt_vocoder_path)).to(self.device).eval()
                )
                self.voco_trt = True
            except Exception as e:
                _LOGGER.error("Failed to load TRT vocoder: %s", e)
        if not self.voco_trt:
            _LOGGER.info("Compiling TRT vocoder...")
            try:
                trt_mod = cast(
                    Any,
                    torch_tensorrt.compile(
                        base_vocoder.eval().to(self.device),
                        inputs=[
                            torch_tensorrt.Input(
                                min_shape=[1, 80, 1],
                                opt_shape=[1, 80, 500],
                                max_shape=[1, 80, 2000],
                                dtype=torch.float32,
                            )
                        ],
                        enabled_precisions={torch.float16},
                        truncate_long_and_double=True,
                        calibrator=None,
                    ),
                )
                trt_mod.save(str(trt_vocoder_path))
                self.vocoder = trt_mod.eval().to(self.device)
                self.voco_trt = True
            except Exception as e:
                _LOGGER.error("Failed to compile TRT vocoder: %s", e)
                self.vocoder = base_vocoder.to(self.device).eval()
        _LOGGER.info("Vocoder engine ready. TRT=%s", self.voco_trt)

        # Warm up models

        self._warmup_models()
        # Set initialized flag

        self.initialized = True

    def quantize_model(self, model: torch.jit.ScriptModule) -> torch.jit.ScriptModule:
        """Retained for API compatibility; returns eval model on current device."""
        return model.eval().to(self.device)

    def _warmup_models(self):
        _LOGGER.info("Priming TRT engines with a minimal dummy run…")
        with torch.no_grad():
            # 1) Tacotron dummy: “Warmup” text → minimal mel
            dummy_x = prepare_text("Warmup", self.device, self.cleaner, self.tokenizer)
            start = time.time()
            _ = cast(Any, self.glados).generate_jit(dummy_x, self.emb.half(), 1.0)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            _LOGGER.debug("Dummy Tacotron took %.1f ms", (time.time() - start) * 1000)

        # Now do your existing bucket warm-up:
        _LOGGER.info("Warming up with buckets: %s", BUCKET_SIZES)
        with torch.no_grad():
            for bucket in BUCKET_SIZES:
                warmup_text = "Hello " * bucket
                x = prepare_text(warmup_text, self.device, self.cleaner, self.tokenizer)

                # Tacotron timing
                start_taco = time.time()
                mel_out = cast(
                    torch.Tensor,
                    cast(
                        dict[str, torch.Tensor],
                        cast(Any, self.glados).generate_jit(x, self.emb.half(), 1.0),
                    )["mel_post"],
                ).to(self.device)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                _LOGGER.debug(
                    "Warmup Tacotron bucket %s took %.1f ms",
                    bucket,
                    (time.time() - start_taco) * 1000,
                )

                # Vocoder timing
                mel = mel_out.float()
                start_voc = time.time()
                _ = cast(Any, self.vocoder)(mel)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                _LOGGER.debug(
                    "Warmup Vocoder bucket %s took %.1f ms",
                    bucket,
                    (time.time() - start_voc) * 1000,
                )

        _LOGGER.info("Model warm-ups complete.")

    def run_tts(self, text: str, alpha: float = 1.0) -> AudioSegment:
        """Generate a full utterance audio segment with timing logs."""
        x = prepare_text(text, self.device, self.cleaner, self.tokenizer)
        emb = self.emb.half() if self.fp16 else self.emb
        with torch.no_grad():
            # Tacotron

            start_taco = time.time()
            out = cast(
                dict[str, torch.Tensor],
                cast(Any, self.glados).generate_jit(x, emb, alpha),
            )
            n_frames = out["mel_post"].shape[-1]
            _LOGGER.debug(
                "Tacotron generated mel with %s frames in %.1f ms",
                n_frames,
                (time.time() - start_taco) * 1000,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize()

            # Vocoder
            mel = out["mel_post"]
            n_frames = mel.shape[-1]
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            start_voc = time.time()
            audio_wave = cast(
                torch.Tensor, cast(Any, self.vocoder)(mel.float()).squeeze()
            )
            voc_ms = (time.time() - start_voc) * 1000
            _LOGGER.debug(
                "Vocoder generated audio from mel with %s frames in %.1f ms",
                n_frames,
                voc_ms,
            )

        pcm = (audio_wave * 32768.0).cpu().numpy().astype("int16").tobytes()
        return AudioSegment(pcm, frame_rate=22050, sample_width=2, channels=1)

    def play_audio(self, audio: AudioSegment):
        """Play a generated audio segment to the default output device."""
        playback.play(audio)

    def speak(self, text: str, alpha: float = 1.0):
        """Synthesize text and immediately play it."""
        self.play_audio(self.run_tts(text, alpha))


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_p1", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--models_dir", type=str, default="models")
    args = parser.parse_args()
    runner = TTSRunner(
        use_p1=args.use_p1, log=args.log, models_dir=Path(args.models_dir)
    )
    while True:
        try:
            user_text = input("Input: ")
            if user_text.strip():
                runner.speak(user_text)
        except KeyboardInterrupt:
            print("Exiting...")
            break
