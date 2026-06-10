"""Core GLaDOS TTS model runner and CLI test harness."""

import logging
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import torch
import torch_tensorrt
from dp.preprocessing.text import LanguageTokenizer, Preprocessor, SequenceTokenizer
from pydub import AudioSegment, playback

from .utils.tools import (
    get_cleaner_and_tokenizer,
    prepare_text,
)

_LOGGER = logging.getLogger(__name__)

# Bucket sizes for Tacotron warm-ups

BUCKET_SIZES = [8, 16, 32]

# Output audio format produced by the vocoder.

SAMPLE_RATE = 22050
SAMPLE_WIDTH = 2
CHANNELS = 1

# Long mels are vocoded in windows so the first audio bytes can be streamed
# before the whole utterance is vocoded. Each window carries extra context
# frames on both sides that are trimmed from the output, so HiFiGAN's
# receptive field never sees a hard edge inside the kept region. The window
# size matches the TRT profile's opt_shape (500 frames); window + context
# stays well under its max_shape (2000 frames), which also lets utterances
# longer than the profile maximum synthesize instead of failing.

VOCODER_CHUNK_FRAMES = 500
VOCODER_CONTEXT_FRAMES = 32

# Script used to probe-load a serialized TRT engine in a throwaway subprocess.
# A stale/incompatible engine can abort the interpreter with a native segfault
# on deserialization, which a try/except in this process cannot catch.

_TRT_PROBE_SRC = (
    "import sys\n"
    "import torch\n"
    "import torch_tensorrt  # noqa: F401 - registers the TRT runtime/ops\n"
    "module = torch.jit.load(sys.argv[1])\n"
    "if sys.argv[2] != 'cpu':\n"
    "    module = module.to(sys.argv[2])\n"
    "module.eval()\n"
)


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
        # TRT execution contexts are not safe for concurrent execution, so all
        # inference is serialized behind this lock (the GPU would serialize the
        # work anyway).
        self._infer_lock = threading.Lock()

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
                "Non-CUDA device (%s), using default workspace size: 2GB",
                self.device.type,
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
                "CUDA device %s: Total VRAM: %.2fGB, Setting workspace size to %.2fGB",
                device_index,
                total_memory / (1024**3),
                workspace_size / (1024**3),
            )
            return workspace_size
        except Exception as e:
            _LOGGER.warning(
                "Failed to detect VRAM, using default workspace size: %s", e
            )
            return default_workspace

    @staticmethod
    def _is_stale_trt_engine_error(err: Exception) -> bool:
        message = str(err).lower()
        stale_markers = (
            "version tag does not match",
            "serialized engine version",
            "deserializecudaengine",
            "unable to deserialize the tensorrt engine",
            "serialization assertion",
        )
        return any(marker in message for marker in stale_markers)

    def _remove_trt_engine(self, engine_path: Path, engine_name: str) -> None:
        try:
            engine_path.unlink(missing_ok=True)
            _LOGGER.warning(
                "Removed stale TRT %s engine cache at %s.",
                engine_name,
                engine_path,
            )
        except Exception as cleanup_err:
            _LOGGER.warning(
                "Failed to remove stale TRT %s engine cache at %s: %s",
                engine_name,
                engine_path,
                cleanup_err,
            )

    def _prune_stale_trt_engine(
        self, engine_path: Path, err: Exception, engine_name: str
    ) -> None:
        if not self._is_stale_trt_engine_error(err):
            return
        self._remove_trt_engine(engine_path, engine_name)

    def _trt_engine_loads_safely(self, engine_path: Path) -> bool:
        """Return True if the engine deserializes without crashing the process.

        The engine is loaded in a throwaway subprocess first so that a native
        crash (segfault on a stale/incompatible TensorRT engine) is contained
        there instead of taking down the server. Only when the probe succeeds
        do we load the engine for real in this process.
        """
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    _TRT_PROBE_SRC,
                    str(engine_path),
                    self.device.type,
                ],
                capture_output=True,
                timeout=180,
                check=False,
            )
        except subprocess.TimeoutExpired:
            _LOGGER.error(
                "TRT engine probe for %s timed out; treating engine as unusable.",
                engine_path,
            )
            return False
        if result.returncode != 0:
            _LOGGER.error(
                "TRT engine probe for %s failed (exit %s): %s",
                engine_path,
                result.returncode,
                result.stderr.decode(errors="replace").strip(),
            )
            return False
        return True

    def _flatten_tacotron_rnn(self) -> None:
        """Flatten Tacotron RNN parameters when available for better cuDNN performance."""
        try:
            glados_model = cast(Any, self.glados)
            rnn = getattr(glados_model, "rnn", None)
            if rnn is None and hasattr(glados_model, "module"):
                rnn = getattr(glados_model.module, "rnn", None)

            if rnn is None or not hasattr(rnn, "flatten_parameters"):
                _LOGGER.debug("No Tacotron RNN flatten hook available.")
                return

            rnn.flatten_parameters()
            _LOGGER.debug("Flattened Tacotron RNN parameters.")
        except Exception as err:
            _LOGGER.debug("Tacotron RNN flatten skipped: %s", err)

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
        emb = torch.load(str(emb_path), map_location=self.device).to(self.device)
        # Warm-up and real inference must use the same dtype: the TRT graph
        # specializes on the dtypes of its first call, so a mismatch makes the
        # first real request pay a re-specialization instead of hitting the
        # warmed engine.
        self.fp16 = self.device.type == "cuda"
        self.emb = emb.half() if self.fp16 else emb.float()

        # Model file paths

        tacotron_path = self.models_dir / "glados-new.pt"
        vocoder_path = self.models_dir / "vocoder-gpu.pt"
        trt_tacotron_path = self.models_dir / "tacotron-trt.ts"
        _LOGGER.debug("Looking for Tacotron at: %s", tacotron_path)
        trt_vocoder_path = self.models_dir / "vocoder-trt.ts"
        _LOGGER.debug("Looking for TRT vocoder at: %s", trt_vocoder_path)

        # Load or compile TRT Tacotron with extended profile

        self.taco_trt = False
        if trt_tacotron_path.exists():
            if self._trt_engine_loads_safely(trt_tacotron_path):
                _LOGGER.info("Loading TRT tacotron engine...")
                try:
                    self.glados = (
                        torch.jit.load(str(trt_tacotron_path)).to(self.device).eval()
                    )
                    self.taco_trt = True
                except Exception as e:
                    _LOGGER.error("Failed to load TRT tacotron: %s", e)
                    self._prune_stale_trt_engine(trt_tacotron_path, e, "tacotron")
            else:
                _LOGGER.error(
                    "TRT tacotron engine at %s crashed a load probe; "
                    "removing it and recompiling.",
                    trt_tacotron_path,
                )
                self._remove_trt_engine(trt_tacotron_path, "tacotron")
        if not self.taco_trt:
            # The base TorchScript model is only loaded when there is no
            # usable cached engine, keeping startup time and peak VRAM down.
            base_tacotron = torch.jit.load(str(tacotron_path), map_location=self.device)
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
            del base_tacotron
        _LOGGER.info("Tacotron engine ready. TRT=%s", self.taco_trt)

        # Load or compile TRT Vocoder with extended profile

        self.voco_trt = False
        if trt_vocoder_path.exists():
            if self._trt_engine_loads_safely(trt_vocoder_path):
                _LOGGER.info("Loading TRT vocoder engine...")
                try:
                    self.vocoder = (
                        torch.jit.load(str(trt_vocoder_path)).to(self.device).eval()
                    )
                    self.voco_trt = True
                except Exception as e:
                    _LOGGER.error("Failed to load TRT vocoder: %s", e)
                    self._prune_stale_trt_engine(trt_vocoder_path, e, "vocoder")
            else:
                _LOGGER.error(
                    "TRT vocoder engine at %s crashed a load probe; "
                    "removing it and recompiling.",
                    trt_vocoder_path,
                )
                self._remove_trt_engine(trt_vocoder_path, "vocoder")
        if not self.voco_trt:
            base_vocoder = torch.jit.load(str(vocoder_path), map_location=self.device)
            _LOGGER.info("Compiling TRT vocoder...")
            try:
                trt_mod = cast(
                    Any,
                    torch_tensorrt.compile(
                        base_vocoder.eval().to(self.device),
                        ir="torchscript",
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
                    ),
                )
                trt_mod.save(str(trt_vocoder_path))
                self.vocoder = trt_mod.eval().to(self.device)
                self.voco_trt = True
            except Exception as e:
                _LOGGER.error("Failed to compile TRT vocoder: %s", e)
                self.vocoder = base_vocoder.to(self.device).eval()
            del base_vocoder
        _LOGGER.info("Vocoder engine ready. TRT=%s", self.voco_trt)

        self._flatten_tacotron_rnn()

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
            _ = cast(Any, self.glados).generate_jit(dummy_x, self.emb, 1.0)
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
                        cast(Any, self.glados).generate_jit(x, self.emb, 1.0),
                    )["mel_post"],
                ).to(self.device)
                _LOGGER.debug(
                    "Warmup Tacotron bucket %s took %.1f ms",
                    bucket,
                    (time.time() - start_taco) * 1000,
                )

                # Vocoder timing
                mel = mel_out.float()
                start_voc = time.time()
                _ = cast(Any, self.vocoder)(mel)
                _LOGGER.debug(
                    "Warmup Vocoder bucket %s took %.1f ms",
                    bucket,
                    (time.time() - start_voc) * 1000,
                )

        if self.device.type == "cuda":
            # One-time release of compilation/warm-up scratch memory. Never do
            # this on the request path: it forces the caching allocator to
            # re-allocate from scratch and adds latency to every request.
            torch.cuda.empty_cache()
        _LOGGER.info("Model warm-ups complete.")

    @staticmethod
    def _pcm_bytes(audio_wave: torch.Tensor) -> bytes:
        """Convert a [-1, 1] float waveform tensor to int16 PCM bytes."""
        return (
            (audio_wave.float().clamp(-1.0, 1.0) * 32767.0)
            .cpu()
            .numpy()
            .astype("int16")
            .tobytes()
        )

    def run_tts_stream(self, text: str, alpha: float = 1.0) -> Iterator[bytes]:
        """Synthesize text, yielding int16 PCM as the vocoder produces it.

        The mel-spectrogram is vocoded in windows (see VOCODER_CHUNK_FRAMES)
        so the first audio bytes are available before the whole utterance has
        been vocoded. The inference lock is held until the generator is
        exhausted or closed; consumers must drain or close it promptly.
        """
        x = prepare_text(text, self.device, self.cleaner, self.tokenizer)
        debug = _LOGGER.isEnabledFor(logging.DEBUG)
        with self._infer_lock:
            # Tacotron

            with torch.no_grad():
                start_taco = time.time()
                out = cast(
                    dict[str, torch.Tensor],
                    cast(Any, self.glados).generate_jit(x, self.emb, alpha),
                )
                mel = out["mel_post"].float()
            n_frames = mel.shape[-1]
            if debug:
                if self.device.type == "cuda":
                    # Sync only when timing is reported; it stalls the
                    # pipeline for no benefit otherwise.
                    torch.cuda.synchronize()
                _LOGGER.debug(
                    "Tacotron generated mel with %s frames in %.1f ms",
                    n_frames,
                    (time.time() - start_taco) * 1000,
                )

            # Vocoder, windowed over the mel frames

            start_voc = time.time()
            for start in range(0, n_frames, VOCODER_CHUNK_FRAMES):
                end = min(start + VOCODER_CHUNK_FRAMES, n_frames)
                ctx_start = max(0, start - VOCODER_CONTEXT_FRAMES)
                ctx_end = min(n_frames, end + VOCODER_CONTEXT_FRAMES)
                with torch.no_grad():
                    audio = cast(
                        torch.Tensor,
                        cast(Any, self.vocoder)(mel[:, :, ctx_start:ctx_end]),
                    ).squeeze()
                # Trim the context regions; assumes the vocoder upsamples by
                # an integer hop per frame (true for HiFiGAN).
                hop = audio.shape[-1] // (ctx_end - ctx_start)
                front = (start - ctx_start) * hop
                back = (ctx_end - end) * hop
                yield self._pcm_bytes(audio[front : audio.shape[-1] - back])
            if debug:
                _LOGGER.debug(
                    "Vocoder generated audio from mel with %s frames in %.1f ms",
                    n_frames,
                    (time.time() - start_voc) * 1000,
                )

    def run_tts(self, text: str, alpha: float = 1.0) -> AudioSegment:
        """Generate a full utterance audio segment with timing logs."""
        pcm = b"".join(self.run_tts_stream(text, alpha))
        return AudioSegment(
            pcm,
            frame_rate=SAMPLE_RATE,
            sample_width=SAMPLE_WIDTH,
            channels=CHANNELS,
        )

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
