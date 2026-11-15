"""Command line for the public Falcon package:  python -m tradingbot falcon select|final|all ...   and   python -m tradingbot fundhist"""
from __future__ import annotations

import argparse


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="tradingbot", description="Falcon: paper-trading models for TA-125 stocks (research only, no broker)")
    p.add_argument("--tickers", default="data/tase125.txt", help="the stocks to trade (default: the TA-125 list)")
    p.add_argument("--refresh", action="store_true", help="re-download prices instead of using the cache")
    sub = p.add_subparsers(dest="cmd", required=True)
    fa = sub.add_parser("falcon", help="choose the recipe and settings on validation years, then run the test periods once, with real-world costs; checkpointed")
    fa.add_argument("stage", choices=["select", "final", "all"])
    fa.add_argument("--out", default="runs/falcon", help="folder for checkpoints and results")
    fa.add_argument("--recipes", default="", help="comma-separated subset of candidate recipes ('base' is always included)")
    fa.add_argument("--val", default=None, help="validation window 'start,end' (default: calendar 2024)")
    fa.add_argument("--test-end", default=None, help="last day of the 'Past year' test window")
    fa.add_argument("--pit-check", action="store_true", help="also re-run restricted to point-in-time liquid stocks (downloads all ~527 TASE stocks)")
    fa.add_argument("--all-tickers", default="data/tase_all.txt")
    fh = sub.add_parser("fundhist", help="download quarterly company reports (public pages, 1 request/second) into data/fundamentals_hist")
    fh.add_argument("--delay", type=float, default=1.0); fh.add_argument("--refresh-reports", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "falcon":
        from .falcon import main as falcon_main
        falcon_main(a); return
    from . import fundhist as FH
    from .data import load_tickers
    tk = load_tickers(a.tickers); got = 0
    for i, t in enumerate(tk):
        d = FH.fetch_stock(t, delay=a.delay, refresh=a.refresh_reports); got += len(d) > 1
        if i % 10 == 0:
            print(f"[{i + 1}/{len(tk)}] {t}: {len(d)} quarters", flush=True)
    print(f"reports for {got} of {len(tk)} stocks in {FH.CACHE}")


if __name__ == "__main__":
    main()
