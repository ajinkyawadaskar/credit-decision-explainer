#!/usr/bin/env python
"""Convert a screen recording (.mov/.mp4) to a README-sized GIF.

Uses the ffmpeg binary bundled with imageio-ffmpeg, so it needs no Homebrew,
no sudo, and no global install.

    python tools/mov2gif.py ~/Desktop/recording.mov docs/demo.gif

Two-pass palette encoding: pass 1 builds an optimal 256-colour palette for the
clip, pass 2 encodes against it. Single-pass GIF encoding of UI footage looks
muddy on text, which is most of what this demo shows.
"""
from __future__ import annotations

import argparse, pathlib, subprocess, sys, tempfile

import imageio_ffmpeg


def convert(src: pathlib.Path, dst: pathlib.Path, fps: int, width: int) -> None:
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = f"fps={fps},scale={width}:-1:flags=lanczos"

    with tempfile.TemporaryDirectory() as td:
        palette = pathlib.Path(td) / "palette.png"
        subprocess.run(
            [ff, "-y", "-i", str(src), "-vf", f"{vf},palettegen=stats_mode=diff",
             str(palette)],
            check=True, capture_output=True,
        )
        subprocess.run(
            [ff, "-y", "-i", str(src), "-i", str(palette), "-lavfi",
             f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
             "-loop", "0", str(dst)],
            check=True, capture_output=True,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=pathlib.Path, help="input .mov / .mp4")
    ap.add_argument("dst", type=pathlib.Path, nargs="?", default=pathlib.Path("docs/demo.gif"))
    ap.add_argument("--fps", type=int, default=12, help="12 is plenty for UI footage")
    ap.add_argument("--width", type=int, default=1000)
    a = ap.parse_args()

    if not a.src.exists():
        print(f"not found: {a.src}", file=sys.stderr)
        return 1

    convert(a.src, a.dst, a.fps, a.width)
    mb = a.dst.stat().st_size / 1_048_576
    print(f"wrote {a.dst}  {mb:.1f} MB  ({a.fps}fps, {a.width}px wide)")

    if mb > 10:
        print(f"\n{mb:.1f} MB is large for a README. Retry smaller:")
        print(f"  .venv/bin/python tools/mov2gif.py {a.src} {a.dst} --fps 8 --width 800")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
