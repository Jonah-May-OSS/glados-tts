"""Standalone Flask wrapper for the GLaDOS TTS runner."""

import os
import shutil
import sys
import time
import urllib.parse
from pathlib import Path

from flask import Flask, abort, request, send_file

from glados import TTSRunner

print("\033[1;94mINFO:\033[;97m Initializing TTS Engine...")

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
PORT = 8124
CACHE = True

runner = TTSRunner(use_p1=False, log=True, models_dir=BASE_DIR / "models")


def glados_tts(text: str, key: str | None = None, alpha: float = 1.0) -> bool:
    """Synthesize text to a temporary wav file."""
    if key:
        output_file = AUDIO_DIR / f"GLaDOS-tts-temp-output-{key}.wav"
    else:
        output_file = AUDIO_DIR / "GLaDOS-tts-temp-output.wav"

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    runner.run_tts(text, alpha).export(str(output_file), format="wav")
    return True


def create_app() -> Flask:
    """Create a Flask application for TTS synthesis."""
    app = Flask(__name__)

    @app.route("/synthesize/", defaults={"text": ""})
    @app.route("/synthesize/<path:text>")
    def synthesize(text: str):
        if text == "":
            return "No input"

        line = urllib.parse.unquote(request.url[request.url.find("synthesize/") + 11 :])
        filename = "GLaDOS-tts-" + line.replace(" ", "-")
        filename = filename.replace("!", "")
        filename = filename.replace("°c", "degrees celcius")
        filename = filename.replace(",", "") + ".wav"
        audio_root = AUDIO_DIR.resolve()
        cached_file = (AUDIO_DIR / filename).resolve(strict=False)
        try:
            cached_file.relative_to(audio_root)
        except ValueError:
            return "Invalid input", 400

        audio_root = AUDIO_DIR.resolve()
        try:
            cached_file.resolve().relative_to(audio_root)
        except ValueError:
            abort(400, description="Invalid filename")

        if cached_file.is_file():
            os.utime(cached_file, None)
            print("\033[1;94mINFO:\033[;97m The audio sample sent from cache.")
            return send_file(cached_file)

        key = str(time.time())[7:]
        if not glados_tts(line, key):
            return "TTS Engine Failed"

        tempfile = AUDIO_DIR / f"GLaDOS-tts-temp-output-{key}.wav"
        if len(line) < 200 and CACHE:
            shutil.move(tempfile, cached_file)
            return send_file(cached_file)

        try:
            return send_file(tempfile)
        finally:
            tempfile.unlink(missing_ok=True)

    return app


if __name__ == "__main__":
    print("\033[1;94mINFO:\033[;97m Initializing TTS Server...")
    flask_app = create_app()

    cli = sys.modules.get("flask.cli")
    if cli is not None:
        setattr(cli, "show_server_banner", lambda *_args, **_kwargs: None)

    flask_app.run(host="0.0.0.0", port=PORT)
