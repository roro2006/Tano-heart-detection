"""Backend command for the simulated device screen (heart_web.py, /device).

    python python/device_cmd.py listen                 # a random test recording
    python python/device_cmd.py listen --abnormal      # a random abnormal one
    python python/device_cmd.py listen a0001 --fast    # a named recording, as fast as the link

Starts a listen on every open device screen and prints the board's progress
and verdict here as it happens. The server must be running:
python python/heart_web.py COM4 (or --oracle).
"""
from __future__ import annotations

import argparse
import json
import threading
import urllib.error
import urllib.parse
import urllib.request


def events(base: str, ready: threading.Event):
    """Yield (event, data) from the device stream."""
    with urllib.request.urlopen(f"{base}/api/device/stream") as stream:
        event = None
        for raw in stream:
            line = raw.decode().rstrip("\n").rstrip("\r")
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event:
                if event == "hello":
                    ready.set()
                yield event, json.loads(line[6:])
                event = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["listen"])
    parser.add_argument("name", nargs="?", help="PhysioNet recording, e.g. a0001 (default: random test recording)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--abnormal", action="store_true")
    group.add_argument("--normal", action="store_true")
    parser.add_argument("--fast", action="store_true", help="stream as fast as the link instead of in real time")
    parser.add_argument("--server", default="http://127.0.0.1:8770")
    args = parser.parse_args()

    ready = threading.Event()
    stream = events(args.server, ready)
    first = next(stream)          # subscribe before starting, so no event is missed
    query = {"pace": "fast" if args.fast else "realtime"}
    if args.name:
        query["name"] = args.name
    if args.abnormal or args.normal:
        query["label"] = "abnormal" if args.abnormal else "normal"
    request = urllib.request.Request(f"{args.server}/api/device/listen?{urllib.parse.urlencode(query)}",
                                     data=b"", method="POST")
    try:
        with urllib.request.urlopen(request) as response:
            started = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"device refused: {json.loads(exc.read()).get('error')}")
    print(f"listening to {started['name']} (database {started['database']}); board {first[1]['board']}")

    windows = flagged = 0
    for event, data in stream:
        if event == "window":
            windows += 1
            flagged += (data.get("margin") or 0) > 0
            print(f"\r  {windows} windows analysed on the chip, {flagged} sounded abnormal", end="", flush=True)
        elif event == "result":
            verdict = "ABNORMAL heart sound: refer for echocardiogram" if data["abnormal"] else "normal heart sounds"
            print(f"\n  verdict: {verdict} (margin {data['margin']}, {data['audio_seconds']:.1f} s heard)")
            print(f"  dataset label: {'abnormal' if started['label'] else 'normal'}")
        elif event == "reference":
            print(f"  PC reference check: {'identical' if data['identical'] else 'MISMATCH'}")
        elif event == "error":
            print(f"\n  device error: {data['message']}")
        elif event == "end":
            break


if __name__ == "__main__":
    main()
