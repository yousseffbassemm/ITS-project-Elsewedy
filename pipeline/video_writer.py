"""Browser-friendly H.264 video writer that pipes BGR frames straight to ffmpeg.

We avoid OpenCV's VideoWriter (its available codecs vary by build and often
produce MP4s browsers won't play) and instead stream raw frames to the ffmpeg
binary bundled with imageio-ffmpeg, encoding H.264 + yuv420p + faststart.
"""
from __future__ import annotations

import subprocess
import threading

import imageio_ffmpeg
import numpy as np


class FFmpegH264Writer:
    def __init__(self, path: str, width: int, height: int, fps: float):
        self.path = path
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            exe, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{max(fps, 1):.4f}",
            "-i", "-",
            "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            path,
        ]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        # Drain stderr continuously in a thread. If we only read it at close(),
        # a long encode that emits more than the pipe buffer (~64 KB) would block
        # ffmpeg on stderr while we block on stdin — a deadlock.
        self._err: list[bytes] = []
        self._err_thread = threading.Thread(target=self._drain, daemon=True)
        self._err_thread.start()

    def _drain(self) -> None:
        for line in iter(self.proc.stderr.readline, b""):
            self._err.append(line)

    def _stderr_text(self) -> str:
        return b"".join(self._err).decode(errors="ignore").strip()

    def write(self, frame: np.ndarray) -> None:
        # OpenCV frames are already uint8 and contiguous, but astype()+tobytes()
        # copies the whole frame twice on every write. Pass the buffer straight
        # through instead — both calls below are no-ops in the common case.
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).data)
        except (BrokenPipeError, OSError) as exc:
            # ffmpeg died mid-stream; its stderr says why, the pipe error doesn't.
            self.proc.wait()
            self._err_thread.join(timeout=5)
            raise RuntimeError(
                f"ffmpeg exited during encoding (exit {self.proc.returncode}): "
                f"{self._stderr_text()[:500]}"
            ) from exc

    def close(self) -> None:
        if self.proc.stdin and not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        code = self.proc.wait()
        self._err_thread.join(timeout=5)
        if code != 0:
            raise RuntimeError(f"ffmpeg encoding failed (exit {code}): {self._stderr_text()[:500]}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
