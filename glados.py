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

# Token-length profile for the Tacotron TRT engine. Inputs longer than the
# max fall back to the TorchScript path at runtime.

TACOTRON_TRT_MIN_TOKENS = 1
TACOTRON_TRT_OPT_TOKENS = 128
TACOTRON_TRT_MAX_TOKENS = 512

# Script used to probe-load a serialized TRT engine in a throwaway subprocess.
# A stale/incompatible engine can abort the interpreter with a native segfault
# on deserialization, which a try/except in this process cannot catch.

_trt_probe_src = (
    "import sys\n"
    "import torch\n"
    "import torch_tensorrt  # noqa: F401 - registers the TRT runtime/ops\n"
    "module = torch.jit.load(sys.argv[1])\n"
    "if sys.argv[2] != 'cpu':\n"
    "    module = module.to(sys.argv[2])\n"
    "module.eval()\n"
)


class _TacotronGenerate(torch.nn.Module):
    """Expose Tacotron's generate_jit() as forward() for TensorRT.

    The TorchScript-IR TensorRT frontend only compiles a module's forward
    method, while Tacotron inference goes through the custom generate_jit()
    method - which is also invisible to torch.compile (it only wraps
    forward/__call__). Wrapping generate_jit in a forward makes the graph
    reachable for partial TRT conversion. The speaker embedding is baked in
    as a buffer and the speech rate is fixed at alpha=1.0 (other alphas use
    the TorchScript path at runtime), so forward only takes the token tensor.
    """

    def __init__(self, inner: torch.nn.Module, emb: torch.Tensor) -> None:
        super().__init__()
        self.inner = inner
        self.emb: torch.Tensor
        self.register_buffer("emb", emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner.generate_jit(x, self.emb, 1.0)  # type: ignore[operator]
        return out["mel_post"]


class TTSRunner:
    """Text-to-Speech runner for GLaDOS TTS with streaming support and optional TRT acceleration."""

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
        # generate_jit-capable TorchScript Tacotron, loaded lazily when the
        # TRT wrapper is active (only needed for alpha != 1.0 or over-length
        # inputs).
        self.glados: Any = None
        # TRT-compiled wrapper: forward(x) -> mel_post, alpha = 1.0 baked in.
        self.glados_trt: Any = None
        self.vocoder: Any = None
        self.emb: torch.Tensor = torch.empty(0)
        self.tacotron_path: Path | None = None
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
                    _trt_probe_src,
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

    def _load_tacotron_jit(self) -> Any:
        """Load the base TorchScript Tacotron onto the device."""
        assert self.tacotron_path is not None
        return (
            torch.jit.load(str(self.tacotron_path), map_location=self.device)
            .to(self.device)
            .eval()
        )

    def _optimize_tacotron_jit(self, base_tacotron: Any) -> Any:
        """Apply torch.jit inference optimizations to generate_jit."""
        try:
            return torch.jit.optimize_for_inference(
                base_tacotron, other_methods=["generate_jit"]
            )
        except Exception as err:
            _LOGGER.warning(
                "torch.jit.optimize_for_inference failed for Tacotron, "
                "using plain TorchScript: %s",
                err,
            )
            return base_tacotron

    def _ensure_tacotron_jit(self) -> Any:
        """Return the generate_jit-capable Tacotron, loading it lazily.

        When a cached TRT engine is in use, the base model is not loaded at
        startup; it is only needed for alpha != 1.0 or inputs longer than
        the TRT profile.
        """
        if self.glados is None:
            _LOGGER.info("Lazily loading TorchScript Tacotron...")
            self.glados = self._optimize_tacotron_jit(self._load_tacotron_jit())
            self._flatten_tacotron_rnn()
        return self.glados

    def _generate_mel(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        """Generate a mel-spectrogram, preferring the TRT engine.

        The TRT wrapper bakes in alpha=1.0 and is compiled for inputs up to
        TACOTRON_TRT_MAX_TOKENS; other requests use the TorchScript path.
        """
        if self.taco_trt and alpha == 1.0 and x.shape[-1] <= TACOTRON_TRT_MAX_TOKENS:
            return cast(torch.Tensor, cast(Any, self.glados_trt)(x))
        out = cast(
            dict[str, torch.Tensor],
            self._ensure_tacotron_jit().generate_jit(x, self.emb, alpha),
        )
        return out["mel_post"]

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
        # One embedding dtype everywhere (fp16 on CUDA): it is baked into the
        # TRT Tacotron wrapper at compile time and passed to the TorchScript
        # path at runtime, so warm-up and inference exercise the same kernels.
        self.fp16 = self.device.type == "cuda"
        self.emb = emb.half() if self.fp16 else emb.float()

        # Model file paths

        self.tacotron_path = self.models_dir / "glados-new.pt"
        vocoder_path = self.models_dir / "vocoder-gpu.pt"
        # The engine bakes in the speaker embedding, so the cache is
        # per-voice.
        trt_tacotron_path = (
            self.models_dir / f"tacotron-trt-{'p1' if use_p1 else 'p2'}.ts"
        )
        _LOGGER.debug("Looking for Tacotron at: %s", self.tacotron_path)
        trt_vocoder_path = self.models_dir / "vocoder-trt.ts"
        _LOGGER.debug("Looking for TRT vocoder at: %s", trt_vocoder_path)

        # Drop the legacy engine cache: in older releases the Tacotron "TRT
        # compile" never ran (inference used generate_jit, which torch.compile
        # does not intercept, and the engine was saved before the lazy
        # compilation could trigger), so tacotron-trt.ts is just a mislabeled
        # copy of glados-new.pt.

        legacy_trt_tacotron_path = self.models_dir / "tacotron-trt.ts"
        if legacy_trt_tacotron_path.exists():
            self._remove_trt_engine(legacy_trt_tacotron_path, "legacy tacotron")

        # Load or compile the TRT Tacotron wrapper (forward(x) -> mel_post,
        # alpha=1.0 baked in; see _TacotronGenerate). Runtime falls back to
        # the TorchScript generate_jit path for alpha != 1.0 or over-length
        # inputs.

        self.taco_trt = False
        if trt_tacotron_path.exists():
            if self._trt_engine_loads_safely(trt_tacotron_path):
                _LOGGER.info("Loading TRT tacotron engine...")
                try:
                    self.glados_trt = (
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
            base_tacotron = self._load_tacotron_jit()
            _LOGGER.info("Compiling TRT tacotron...")
            try:
                wrapper = torch.jit.script(_TacotronGenerate(base_tacotron, self.emb))
                trt_mod = cast(
                    Any,
                    torch_tensorrt.compile(
                        wrapper.eval().to(self.device),
                        ir="torchscript",
                        inputs=[
                            torch_tensorrt.Input(
                                min_shape=[1, TACOTRON_TRT_MIN_TOKENS],
                                opt_shape=[1, TACOTRON_TRT_OPT_TOKENS],
                                max_shape=[1, TACOTRON_TRT_MAX_TOKENS],
                                dtype=torch.int64,
                            )
                        ],
                        enabled_precisions={torch.float16},
                        truncate_long_and_double=True,
                        # The autoregressive decoder loop cannot run inside
                        # TRT; convertible subgraphs become TRT engines and
                        # the rest stays TorchScript.
                        require_full_compilation=False,
                    ),
                )
                trt_mod.save(str(trt_tacotron_path))
                self.glados_trt = trt_mod.eval().to(self.device)
                self.taco_trt = True
            except Exception as e:
                _LOGGER.error("Failed to compile TRT tacotron: %s", e)
            if self.taco_trt:
                # Free the base model; alpha != 1.0 or over-length requests
                # reload it lazily via _ensure_tacotron_jit().
                del base_tacotron
            else:
                self.glados = self._optimize_tacotron_jit(base_tacotron)
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

        if self.glados is not None:
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
            _ = self._generate_mel(dummy_x, 1.0)
            _LOGGER.debug("Dummy Tacotron took %.1f ms", (time.time() - start) * 1000)

        # Now do your existing bucket warm-up:
        _LOGGER.info("Warming up with buckets: %s", BUCKET_SIZES)
        with torch.no_grad():
            for bucket in BUCKET_SIZES:
                warmup_text = "Hello " * bucket
                x = prepare_text(warmup_text, self.device, self.cleaner, self.tokenizer)

                # Tacotron timing
                start_taco = time.time()
                mel_out = self._generate_mel(x, 1.0).to(self.device)
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
                mel = self._generate_mel(x, alpha).float()
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
