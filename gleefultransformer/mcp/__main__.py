import argparse
import sys

from .cli import server_main, worker_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Gleeful Transformer MCP commands.")
    parser.add_argument("command", choices=("serve", "worker"))
    args = parser.parse_args(sys.argv[1:2])
    command = server_main if args.command == "serve" else worker_main
    return command(sys.argv[2:])


raise SystemExit(main())
