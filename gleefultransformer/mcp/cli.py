import argparse
import sys
from pathlib import Path

from .errors import ServiceError


def _workspace(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", required=True, type=Path, help="Directory containing registry.json.")


def _check_workspace(parser: argparse.ArgumentParser, workspace: Path) -> Path:
    if not workspace.is_dir():
        parser.error("--workspace must be an existing directory.")
    return workspace.resolve()


def _service_error(exc: ServiceError) -> int:
    print(f"{exc.code}: {exc}", file=sys.stderr)
    return 1


def server_main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Serve Gleeful Transformer MCP tools over stdio.")
    _workspace(parser)
    args = parser.parse_args(argv)
    workspace = _check_workspace(parser, args.workspace)

    from .adapter import create_server
    from .service import RunService

    try:
        create_server(RunService(workspace)).run("stdio")
    except ServiceError as exc:
        return _service_error(exc)
    return 0


def worker_main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the Gleeful Transformer MCP execution worker.")
    _workspace(parser)
    args = parser.parse_args(argv)
    workspace = _check_workspace(parser, args.workspace)

    from .worker import run_forever

    try:
        run_forever(workspace)
    except KeyboardInterrupt:
        return 130
    except ServiceError as exc:
        return _service_error(exc)
    return 0
