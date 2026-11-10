"""moonglide — где встать, чтобы самолёт на заходе прошёл по диску Луны.

    python -m moonglide [--port 8090] [--bind 127.0.0.1] [--cache ~/.cache/moonglide]
"""
import argparse
import logging
import os
from pathlib import Path
from types import SimpleNamespace

from . import approaches
from .dem import Dem
from .runways import Airports, Runways
from .sky import Sky
from .web import serve


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bind", default="127.0.0.1", help="адрес (0.0.0.0 — для всей сети)")
    ap.add_argument("--port", type=int, default=8090)
    default_cache = os.environ.get("STATE_DIRECTORY") or str(Path.home() / ".cache" / "moonglide")
    ap.add_argument("--cache", type=Path, default=Path(default_cache),
                    help="эфемериды, полосы, рельеф (по умолчанию $STATE_DIRECTORY или ~/.cache/moonglide)")
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--approaches", type=Path, action="append",
                    help="параметры заходов, можно несколько — последний главнее "
                         "(по умолчанию approaches.toml и approaches.local.toml рядом с кодом, если есть)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    files = args.approaches or [root / "approaches.toml", root / "approaches.local.toml"]
    aps = approaches.load(*files)
    if aps:
        logging.info("заходы: %s", ", ".join(f"{a} {r} ({v['file']})" for (a, r), v in aps.items()))
    ctx = SimpleNamespace(sky=Sky(args.cache), dem=Dem(args.cache), runways=Runways(args.cache),
                          airports=Airports(args.cache), approaches=aps)
    serve(ctx, args.bind, args.port)


if __name__ == "__main__":
    main()
