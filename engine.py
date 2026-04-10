"""Standalone Flask wrapper for the GLaDOS TTS runner."""

import os
import shutil
import sys
import time
import urllib.parse
from pathlib import Path

from flask import Flask, abort, request, send_file
from werkzeug.utils import secure_filename

from glados import TTSRunner

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
PORT = 8124
CACHE = True


def create_app() -> Flask:
    """Create a Flask application for TTS synthesis."""
    print("\033[1;94mINFO:\033[;97m Initializing TTS Engine...")
    runner = TTSRunner(use_p1=False, log=True, models_dir=BASE_DIR / "models")
    app = Flask(__name__)

    @app.route("/synthesize/", defaults={"text": ""})
    @app.route("/synthesize/<path:text>")
    def synthesize(text: str):
        if text == "":
            return "No input"

        line = urllib.parse.unquote(request.url[request.url.find("synthesize/") + 11 :])
        filename_core = line.replace(" ", "-")
        filename_core = filename_core.replace("!", "")
        filename_core = filename_core.replace("°c", "degrees celcius")
        filename_core = filename_core.replace(",", "")
        filename = f"GLaDOS-tts-{secure_filename(filename_core)}.wav"
        audio_root = AUDIO_DIR.resolve()
        cached_file = (AUDIO_DIR / filename).resolve(strict=False)
        try:
            cached_file.relative_to(audio_root)
        except ValueError:
            abort(400, description="Invalid filename")

        if cached_file.is_file():
            os.utime(cached_file, None)
            print("\033[1;94mINFO:\033[;97m The audio sample sent from cache.")
            return send_file(cached_file)

        key = str(time.time())[7:]
        tempfile = AUDIO_DIR / f"GLaDOS-tts-temp-output-{key}.wav"
        try:
            AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            runner.run_tts(line).export(str(tempfile), format="wav")
        except Exception as exc:
            print(f"\033[1;91mERROR:\033[;97m TTS Engine Failed: {exc}")
            return "TTS Engine Failed"
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
