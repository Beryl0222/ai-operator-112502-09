"""Command line entry: ``python -m operator_quant_groups``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .contract import handle_request


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="operator_quant_groups",
        description="Grouped quantized linear operator and eligibility report",
    )
    parser.add_argument(
        "request",
        nargs="?",
        help="JSON request string or path to a JSON request file",
    )
    parser.add_argument("--action", default="submit")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--input-json", type=Path)
    args = parser.parse_args(argv)

    if args.request:
        if Path(args.request).exists():
            request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        else:
            request = json.loads(args.request)
    else:
        request = {"action": args.action}

    if args.fixture:
        request.setdefault("action", "submit")
        request["payload"] = json.loads(args.fixture.read_text(encoding="utf-8"))
    if args.input_json:
        request["inputs"] = json.loads(args.input_json.read_text(encoding="utf-8"))

    response = handle_request(request)
    json.dump(response, sys.stdout, ensure_ascii=False, indent=2, default=str)
    sys.stdout.write("\n")
    return 0 if response.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
