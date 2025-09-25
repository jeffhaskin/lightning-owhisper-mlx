"""CLI entrypoint for running the transcription server."""

from __future__ import annotations

import argparse
import uvicorn

from .app import create_app
from .config import DEFAULT_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Deepgram-compatible transcription server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8080, help="TCP port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development only)")
    args = parser.parse_args()

    app = create_app(DEFAULT_CONFIG)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":  # pragma: no cover
    main()
