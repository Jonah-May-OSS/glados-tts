import logging
import time
from pathlib import Path
from typing import Generator, Tuple

import torch
import torch_tensorrt
from pydub import AudioSegment, playback
from dp.preprocessing.text import Preprocessor, LanguageTokenizer, SequenceTokenizer

from .utils.tools import prepare_text

_LOGGER = logging.getLogger(__name__)

# Bucket sizes for VOCODER warm-ups
BUCKET_SIZES = [16, 32, 64]


class TTSRunner:
    """Text-to-Speech runner for GLaDOS TTS with streaming support and optional TRT Vocoder."""

    def __init__(
        self,
        use_p1: bool = False,
        log: bool = False,
        models_dir: Path = Path("models"),
    ):
        """
        Initialize TTS: load embedding, cast Tacotron to FP16, and load or compile TRT Vocoder.
        """
        self.log = log
        self.models_dir = models_dir

        # Device selection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        _LOGGER.info(f"Using device: {self.device}")

        # Safe globals for embedding deserialization
        torch.serialization.add_safe_globals(
            [Preprocessor, LanguageTokenizer, SequenceTokenizer]
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
        trt_vocoder_path = (
            (self.models_dir / "vocoder.trt")
            if (self.models_dir / "vocoder.trt").exists()
            else (self.models_dir / "vocoder-trt.ts")
        )
        _LOGGER.debug(f"Looking for TRT vocoder at: {trt_vocoder_path}")

        # Load TorchScript models
        base_tacotron = torch.jit.load(str(tacotron_path), map_location=self.device)
        base_vocoder = torch.jit.load(str(vocoder_path), map_location=self.device)

        # Convert Tacotron weights to FP16 for speed
        _LOGGER.info("Converting Tacotron to FP16...")
        self.glados = base_tacotron.half().to(self.device).eval()
        # Compact RNN weights
        try:
            self.glados.flatten_parameters()
        except Exception:
            for m in self.glados.modules():
                if hasattr(m, "flatten_parameters"):
                    m.flatten_parameters()
        self.fp16 = True
        # Warm up Tacotron model to avoid first-chunk overhead
        _LOGGER.info("Warming up Tacotron model... (initial dummy run)")
        dummy_text = "Test"
        dummy_input = prepare_text(dummy_text, self.models_dir, self.device)
        _ = self.glados.generate_jit(dummy_input, self.emb.half(), 1.0)

        # Load or compile TRT Vocoder with extended profile
        self.trt = False
        if trt_vocoder_path.exists():
            _LOGGER.info("Loading TRT vocoder engine...")
            try:
                self.vocoder = (
                    torch.jit.load(str(trt_vocoder_path)).to(self.device).eval()
                )
                self.trt = True
            except Exception as e:
                _LOGGER.error(f"Failed to load TRT vocoder: {e}")
        if not self.trt:
            _LOGGER.info("Compiling TRT vocoder...")
            trt_mod = torch_tensorrt.compile(
                base_vocoder.eval().to(self.device),
                inputs=[
                    torch_tensorrt.Input(
                        min_shape=[1, 80, 1],
                        opt_shape=[1, 80, 500],
                        max_shape=[1, 80, 4000],
                        dtype=torch.float32,
                    )
                ],
                enabled_precisions={torch.float32},
                truncate_long_and_double=True,
            )
            trt_mod.save(str(trt_vocoder_path))
            self.vocoder = trt_mod.eval().to(self.device)
            self.trt = True

        # Warm up models
        self._warmup_models()

    def _warmup_models(self):
        _LOGGER.info(f"Warming up with buckets: {BUCKET_SIZES}")
        with torch.no_grad():
            for bucket in BUCKET_SIZES:
                text = "Hello " * bucket
                x = prepare_text(text, self.models_dir, self.device)
                mel_out = self.glados.generate_jit(x, self.emb.half(), 1.0)[
                    "mel_post"
                ].to(self.device)
                mel = mel_out.half() if self.trt else mel_out.float()
                _ = self.vocoder(mel)
                _LOGGER.debug(f"Completed warmup bucket size {bucket}")
        _LOGGER.info("Model warm-ups complete.")

    def run_tts(self, text: str, alpha: float = 1.0) -> AudioSegment:
        """Generate a full utterance audio segment."""
        x = prepare_text(text, self.models_dir, self.device)
        emb = self.emb.half() if self.fp16 else self.emb
        with torch.no_grad():
            if self.log:
                start = time.time()
            out = self.glados.generate_jit(x, emb, alpha)
            if self.log:
                _LOGGER.debug(f"Tacotron took {(time.time()-start)*1000:.1f} ms")
            mel = out["mel_post"].to(self.device)
            mel = mel.half() if self.trt else mel.float()
            if self.log:
                start = time.time()
            audio_wave = self.vocoder(mel).squeeze()
            if self.log:
                _LOGGER.debug(f"Vocoder took {(time.time()-start)*1000:.1f} ms")
        pcm = (audio_wave * 32768.0).cpu().numpy().astype("int16").tobytes()
        return AudioSegment(pcm, frame_rate=22050, sample_width=2, channels=1)

    def stream_tts(
        self, text: str, alpha: float = 1.0, samples_per_chunk: int = 1024
    ) -> Generator[Tuple[bytes, int, int, int], None, None]:
        """Pipelined TTS streaming generator with debug timing."""
        rate, width, channels = 22050, 2, 1
        yield (b"__AUDIO_START__", rate, width, channels)
        for sentence in filter(None, (s.strip() for s in text.split("."))):
            x = prepare_text(sentence, self.models_dir, self.device)
            emb = self.emb.half() if self.fp16 else self.emb
            with torch.no_grad():
                if self.log:
                    start_taco = time.time()
                out = self.glados.generate_jit(x, emb, alpha)
                if self.log:
                    _LOGGER.debug(
                        f"Tacotron chunk took {(time.time()-start_taco)*1000:.1f} ms"
                    )
                mel_post = out["mel_post"].to(self.device)
                mel_post = mel_post.half() if self.trt else mel_post.float()
                if self.log:
                    start_voc = time.time()
                audio_wave = self.vocoder(mel_post).squeeze()
                if self.log:
                    _LOGGER.debug(
                        f"Vocoder chunk took {(time.time()-start_voc)*1000:.1f} ms"
                    )
            raw = (audio_wave * 32768.0).cpu().numpy().astype("int16").tobytes()
            for i in range(0, len(raw), samples_per_chunk * width * channels):
                yield (
                    raw[i : i + samples_per_chunk * width * channels],
                    rate,
                    width,
                    channels,
                )
        yield (b"__AUDIO_STOP__", rate, width, channels)
        yield (b"__SYNTH_STOPPED__", rate, width, channels)

    def play_audio(self, audio: AudioSegment):
        playback.play(audio)

    def speak(self, text: str, alpha: float = 1.0):
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
            text = input("Input: ")
            if text.strip():
                runner.speak(text)
        except KeyboardInterrupt:
            print("Exiting...")
            break
