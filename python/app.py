"""Tkinter 16x16 drawing front end for the trained pattern denoiser.

    python python/app.py COM4          # whole model on the FPGA
    python python/app.py --oracle      # software oracle, no board needed

Draw on the left grid, press Infer, and the right grid shows the model's
reconstruction plus the nearest pattern class. The whole model runs on the
FPGA (one INFER per press) with the weights baked into the bitstream; pass
--load to send --weights first instead.
"""
from __future__ import annotations
import argparse
import sys
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import tpu_model
from evaluate import classify
from host import TpuLink

CELL = 20
WEIGHTS = Path(__file__).resolve().parents[1] / "weights"


class DrawingApp:
    def __init__(self, model: tpu_model.TpuModel, tpu: TpuLink | None):
        self.model, self.tpu = model, tpu
        self.grid = [[0] * 16 for _ in range(16)]
        self.root = tk.Tk()
        self.root.title(f"TPU denoiser ({'FPGA' if tpu else 'software oracle'})")
        self.canvas = tk.Canvas(self.root, width=16 * CELL * 2 + CELL, height=16 * CELL)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.paint)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.status = tk.Label(self.root, text="draw a cross, box or x")
        self.status.pack()
        tk.Button(self.root, text="Infer", command=self.infer).pack(side=tk.LEFT)
        tk.Button(self.root, text="Clear", command=self.clear).pack(side=tk.LEFT)
        self.cells = self._cells(0)
        self.outputs = self._cells(17 * CELL)

    def _cells(self, left: int):
        return [[self.canvas.create_rectangle(left + c * CELL, r * CELL, left + c * CELL + CELL,
                 r * CELL + CELL, fill="white") for c in range(16)] for r in range(16)]

    def paint(self, event):
        r, c = event.y // CELL, event.x // CELL
        if 0 <= r < 16 and 0 <= c < 16:
            self.grid[r][c] = 1
            self.canvas.itemconfigure(self.cells[r][c], fill="black")

    def clear(self):
        self.grid = [[0] * 16 for _ in range(16)]
        for row in self.cells + self.outputs:
            for cell in row:
                self.canvas.itemconfigure(cell, fill="white")

    def infer(self):
        features = tpu_model.grid_features(self.grid)
        if self.tpu:
            _, cycles, logits = self.tpu.infer(features)
            timing = f" ({cycles / 27e3:.2f} ms on chip)"
        else:
            logits, timing = tpu_model.forward(self.model, features), ""
        output = tpu_model.pixels(logits)
        for r in range(16):
            for c in range(16):
                self.canvas.itemconfigure(self.outputs[r][c], fill="black" if output[r][c] else "white")
        self.status.configure(text=f"looks like: {classify(output)}{timing}")

    def run(self):
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("port", nargs="?", help="UART port, e.g. COM4")
    parser.add_argument("--oracle", action="store_true", help="run the model in software")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--load", action="store_true", help="load --weights into the FPGA first")
    args = parser.parse_args()
    if not args.port and not args.oracle:
        parser.error("give a port or --oracle")
    model = tpu_model.load(args.weights)
    if args.oracle:
        DrawingApp(model, None).run()
        return
    with TpuLink(args.port) as tpu:
        if args.load:
            tpu.load_model(model)
        DrawingApp(model, tpu).run()


if __name__ == "__main__":
    main()
