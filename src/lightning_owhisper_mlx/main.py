"""CLI entrypoint for running the server."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .config import AppConfig, load_config
from .server import create_app


def _apply_cli_overrides(config: AppConfig, host: str | None, port: int | None) -> AppConfig:
    data = config.model_dump()
    if host is not None:
        data.setdefault("general", {})["host"] = host
    if port is not None:
        data.setdefault("general", {})["port"] = port
    return AppConfig.model_validate(data)


def app() -> None:
    parser = argparse.ArgumentParser(description="Run the Lightning OWhisper MLX server")
    parser.add_argument("--config", type=Path, default=None, help="Path to configuration YAML")
    parser.add_argument("--host", type=str, default=None, help="Override host from config")
    parser.add_argument("--port", type=int, default=None, help="Override port from config")
    args = parser.parse_args()

    config = load_config(args.config)
    config = _apply_cli_overrides(config, args.host, args.port)

    uvicorn.run(
        create_app(config),
        host=config.general.host,
        port=config.general.port,
        log_level="info",
    )


if __name__ == "__main__":  # pragma: no cover
    app()
