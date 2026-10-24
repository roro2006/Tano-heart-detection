"""Browser front end: watch the FPGA screen a heart recording as it streams.

    python python/heart_web.py COM4            # then open http://127.0.0.1:8770
    python python/heart_web.py --oracle        # no board: the PC reference plays its part

Pick a PhysioNet recording and the server streams its raw 2 kHz samples to
the board (AUDIO), 1024 at a time, reporting the board's running window count
and margin after every chunk. In real-time mode the chunks are paced at 2 kHz
so the page can play the recording while the board listens. When the
recording ends, the board's totals are checked against heart_features +
tpu_model.forward on the PC, as heart_host.py does.

Live mode takes the browser's microphone instead: the page resamples it to
2 kHz and posts it in AUDIO-sized chunks, the board screens it as it arrives,
and each session is saved to data/live/ as a 2 kHz WAV. The model was trained
on stethoscope recordings, so use a stethoscope microphone; a laptop
microphone is a new recording condition and its verdicts mean little.
"""
from __future__ import annotations

import argparse
import array
import io
import json
import sys
import threading
import time
import uuid
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).parent))
import heart_dataset
import heart_features
import tpu_model
from host import AUDIO_CHUNK, TpuLink

ROOT = Path(__file__).resolve().parents[1]
PAGE = Path(__file__).with_name("heart_web.html")
RATE = 2000
ENVELOPE = 20          # samples per waveform point sent to the page (100 points/s)
LIVE_DIR = ROOT / "data" / "live"   # every microphone session is saved here as a 2 kHz WAV
LIVE_IDLE = 5.0        # seconds without a chunk before another page may take the board


def playback_wav(samples: list[int], factor: int = 4) -> bytes:
    """The recording as an 8 kHz WAV: browsers will not decode 2 kHz audio.
    Linear interpolation; only the page's playback uses this, never the board."""
    out = []
    for a, b in zip(samples, samples[1:] + samples[-1:]):
        out += [a + (b - a) * k // factor for k in range(factor)]
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE * factor)
        handle.writeframes(array.array("h", out).tobytes())
    return buffer.getvalue()


def window_margins(model: tpu_model.TpuModel, windows: list) -> list[int]:
    """Per-window (pooled[1] - pooled[0]) from the PC reference."""
    out = []
    for features in windows:
        logits = tpu_model.forward(model, features)
        out.append(sum(row[1] for row in logits) - sum(row[0] for row in logits))
    return out


class OracleLink:
    """Stands in for TpuLink without a board: replays the PC reference chunk by chunk."""

    def __init__(self, model: tpu_model.TpuModel):
        self.model = model
        self.samples: list[int] = []

    def info(self) -> tuple[bytes, int, int]:
        return b"PC ", 0, 0

    def audio(self, samples, reset: bool = False) -> tuple[int, int, int]:
        if reset:
            self.samples = []
        self.samples += list(samples)
        windows = heart_features.recording_windows(self.samples)
        margins = window_margins(self.model, windows)
        last = tpu_model.predicted_class(tpu_model.forward(self.model, windows[-1])) if windows else 0
        return len(windows), sum(margins), last


class Screen:
    def __init__(self, link, model: tpu_model.TpuModel, port: str):
        self.link, self.model, self.port = link, model, port
        self.lock = threading.Lock()
        self.items = {r.name: r for r in heart_dataset.recordings()}
        self.split = heart_dataset.assign_splits(list(self.items.values()))
        magic, version, _ = link.info()
        self.board = {"port": port, "magic": magic.decode(errors="replace").strip(), "version": version}
        self.live: dict | None = None      # the microphone session that holds the lock

    def _acquire(self) -> bool:
        """Take the board, from a microphone session that went quiet if need be."""
        if self.lock.acquire(timeout=1):
            return True
        live = self.live
        if live and time.monotonic() - live["active"] > LIVE_IDLE:
            self.live = None           # the lock passes straight to the caller
            return True
        return False

    # Live microphone: the page posts 2 kHz samples, at most AUDIO_CHUNK at a time.

    def live_start(self) -> dict:
        if not self._acquire():
            raise RuntimeError("the board is busy with another recording")
        try:
            self.link.audio([], reset=True)
        except Exception:
            self.lock.release()
            raise
        self.live = {"id": uuid.uuid4().hex, "active": time.monotonic(), "samples": [],
                     "windows": 0, "margin": 0}
        return {"id": self.live["id"], "rate": RATE, "chunk": AUDIO_CHUNK, "frame": heart_features.FRAME}

    def _session(self, sid: str) -> dict:
        live = self.live
        if not live or live["id"] != sid:
            raise RuntimeError("this microphone session has ended; start a new one")
        live["active"] = time.monotonic()
        return live

    def live_chunk(self, sid: str, samples: list[int]) -> dict:
        live = self._session(sid)
        count, margin, last = self.link.audio(samples)
        live["samples"] += samples
        new = []
        if count > live["windows"]:
            # For the heatmap only: the latest window's features from the last
            # 18 frames (filters restart, so this is close to, not exactly, the
            # board's input). The board's own decision does not depend on it.
            tail = live["samples"][-(heart_features.FRAMES + 2) * heart_features.FRAME:]
            features = heart_features.recording_windows(tail)
            new.append({"index": count - 1, "margin": margin - live["margin"], "total": margin,
                        "class": last, "features": features[-1] if features else None})
        live["windows"], live["margin"] = count, margin
        return {"windows": count, "margin": margin, "last": last, "new": new}

    def live_stop(self, sid: str) -> dict:
        live = self._session(sid)
        self.live = None
        self.lock.release()
        samples = live["samples"]
        saved = None
        if samples:
            LIVE_DIR.mkdir(parents=True, exist_ok=True)
            saved = LIVE_DIR / time.strftime("live-%Y%m%d-%H%M%S.wav")
            with wave.open(str(saved), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(RATE)
                handle.writeframes(array.array("h", samples).tobytes())
        windows = heart_features.recording_windows(samples)
        pc = window_margins(self.model, windows)
        return {"windows": live["windows"], "margin": live["margin"], "abnormal": live["margin"] > 0,
                "seconds": len(samples) / RATE, "saved": str(saved.relative_to(ROOT)) if saved else None,
                "reference": {"margins": pc, "windows": len(pc), "margin": sum(pc),
                              "identical": (len(pc), sum(pc)) == (live["windows"], live["margin"])}}

    def catalog(self) -> list[dict]:
        return [{"name": r.name, "database": r.database, "label": r.label, "split": self.split[r.name]}
                for r in self.items.values()]

    def run(self, name: str, realtime: bool, send) -> None:
        record = self.items[name]
        samples = heart_dataset.read_samples(record.path)
        windows = heart_features.recording_windows(samples)
        envelope = [[min(samples[i:i + ENVELOPE]), max(samples[i:i + ENVELOPE])]
                    for i in range(0, len(samples), ENVELOPE)]
        send("meta", {"name": name, "database": record.database, "label": record.label,
                      "split": self.split[name], "samples": len(samples), "rate": RATE,
                      "envelope_step": ENVELOPE, "envelope": envelope, "features": windows,
                      "frame": heart_features.FRAME, "realtime": realtime})

        # The PC reference runs alongside the board and reports when it is done.
        reference: dict = {}
        def compute():
            reference["margins"] = window_margins(self.model, windows)
        checker = threading.Thread(target=compute, daemon=True)
        checker.start()

        if not self._acquire():
            send("error", {"message": "the board is busy with another recording"})
            return
        try:
            start = time.perf_counter()
            board_seconds = 0.0
            seen, total = self.link.audio([], reset=True)[:2]
            for offset in range(0, len(samples), AUDIO_CHUNK):
                chunk = samples[offset:offset + AUDIO_CHUNK]
                t = time.perf_counter()
                count, margin, last = self.link.audio(chunk)
                board_seconds += time.perf_counter() - t
                sent = offset + len(chunk)
                for index in range(seen, count):
                    # One window per chunk at most (1024 samples < the 8-frame hop),
                    # so the change in the running sum is this window's margin.
                    send("window", {"index": index, "margin": margin - total if index == count - 1 else None,
                                    "total": margin, "class": last})
                seen, total = count, margin
                send("progress", {"sent": sent, "windows": count, "margin": margin})
                if realtime:
                    ahead = sent / RATE - (time.perf_counter() - start)
                    if ahead > 0:
                        time.sleep(ahead)
            elapsed = time.perf_counter() - start
        finally:
            self.lock.release()

        abnormal = total > 0
        send("result", {"windows": seen, "margin": total, "abnormal": abnormal,
                        "correct": abnormal == bool(record.label), "elapsed": elapsed,
                        "link_seconds": board_seconds, "audio_seconds": len(samples) / RATE})
        checker.join()
        pc = reference["margins"]
        send("reference", {"margins": pc, "windows": len(pc), "margin": sum(pc),
                           "identical": (len(pc), sum(pc)) == (seen, total)})


def handler_for(screen: Screen):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _json(self, body) -> None:
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path == "/":
                data = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif url.path == "/api/board":
                self._json(screen.board)
            elif url.path == "/api/recordings":
                self._json(screen.catalog())
            elif url.path == "/api/wav" and query.get("name") in screen.items:
                data = playback_wav(heart_dataset.read_samples(screen.items[query["name"]].path))
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif url.path == "/api/run" and query.get("name") in screen.items:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                def send(event: str, body) -> None:
                    self.wfile.write(f"event: {event}\ndata: {json.dumps(body)}\n\n".encode())
                    self.wfile.flush()
                try:
                    screen.run(query["name"], query.get("pace") == "realtime", send)
                    send("end", {})
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as exc:  # report board errors to the page
                    try:
                        send("error", {"message": f"{type(exc).__name__}: {exc}"})
                    except OSError:
                        pass
            else:
                self.send_error(404)

        def do_POST(self):
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            try:
                if url.path == "/api/live/start":
                    self._json(screen.live_start())
                elif url.path == "/api/live/chunk":
                    if len(body) % 2 or len(body) > 2 * AUDIO_CHUNK:
                        raise ValueError(f"send at most {AUDIO_CHUNK} int16 samples per chunk")
                    samples = array.array("h", body).tolist()
                    self._json(screen.live_chunk(query.get("id", ""), samples))
                elif url.path == "/api/live/stop":
                    self._json(screen.live_stop(query.get("id", "")))
                else:
                    self.send_error(404)
            except Exception as exc:
                data = json.dumps({"error": f"{exc}"}).encode()
                self.send_response(409)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("port", nargs="?", help="UART port, e.g. COM4")
    parser.add_argument("--oracle", action="store_true", help="no board: use the PC reference")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights" / "heart",
                        help="the model the board runs (baked into the bitstream)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8770)
    args = parser.parse_args()
    if not args.port and not args.oracle:
        parser.error("give a port or --oracle")
    model = tpu_model.load(args.weights)
    link = OracleLink(model) if args.oracle else TpuLink(args.port)
    screen = Screen(link, model, "PC oracle" if args.oracle else args.port)
    print(f"board: {screen.board}; open http://{args.host}:{args.http_port}", flush=True)
    ThreadingHTTPServer((args.host, args.http_port), handler_for(screen)).serve_forever()


if __name__ == "__main__":
    main()
