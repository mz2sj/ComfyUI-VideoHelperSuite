"""Silent-audio fallback for uploads that have no audio stream.

Run from the repo root:

    python tests/test_silent_audio.py
"""
import array
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class FakeTensor:
    def __init__(self, values, shape):
        self.values = list(values)
        self.shape = tuple(shape)
        if _numel(self.shape) != len(self.values):
            raise ValueError(f"shape {self.shape} does not match {len(self.values)} values")

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = list(shape)
        unknown = None
        known = 1
        for index, size in enumerate(shape):
            if size == -1:
                unknown = index
            else:
                known *= size
        if unknown is not None:
            shape[unknown] = 0 if known == 0 else len(self.values) // known
        return FakeTensor(self.values, shape)

    def transpose(self, dim0, dim1):
        if len(self.shape) != 2:
            raise ValueError("test stub only transposes 2D audio")
        rows, cols = self.shape
        transposed = []
        for col in range(cols):
            for row in range(rows):
                transposed.append(self.values[row * cols + col])
        return FakeTensor(transposed, (cols, rows))

    def unsqueeze(self, dim):
        shape = list(self.shape)
        shape.insert(dim, 1)
        return FakeTensor(self.values, shape)

    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]


def _numel(shape):
    count = 1
    for size in shape:
        count *= int(size)
    return count


def _install_import_stubs():
    if "server" not in sys.modules:
        try:
            import server  # noqa: F401
        except ImportError:
            server = types.ModuleType("server")

            class PromptServer:
                class instance:
                    class prompt_queue:
                        currently_running = {}

            server.PromptServer = PromptServer
            sys.modules["server"] = server
    if "folder_paths" not in sys.modules:
        try:
            import folder_paths  # noqa: F401
        except ImportError:
            folder_paths = types.ModuleType("folder_paths")
            folder_paths.get_temp_directory = lambda: tempfile.gettempdir()
            sys.modules["folder_paths"] = folder_paths
    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except ImportError:
            torch = types.ModuleType("torch")

            def frombuffer(buf, dtype=None):
                samples = array.array("f")
                samples.frombytes(bytes(buf))
                return FakeTensor(samples, (len(samples),))

            def zeros(shape, dtype=None):
                return FakeTensor([0.0] * _numel(shape), shape)

            torch.Tensor = FakeTensor
            torch.float32 = "float32"
            torch.frombuffer = frombuffer
            torch.zeros = zeros
            sys.modules["torch"] = torch


_install_import_stubs()

from videohelpersuite.utils import (  # noqa: E402
    _container_duration_seconds,
    _resolve_silence_seconds,
    _stream_kinds,
    get_audio,
    lazy_get_audio,
    missing_audio_seconds,
    silent_preview_args,
)
import videohelpersuite.utils as vhs_utils  # noqa: E402


USER_STDERR = """\
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'silent.mp4':
  Duration: 00:00:05.03, start: 0.000000, bitrate: 1667 kb/s
  Stream #0:0(und): Video: h264 (High), yuv420p, 928x960, 1643 kb/s, 32 fps
Output #0, f32le, to 'pipe:':
[out#0/f32le] Output file does not contain any stream
Error opening output file -.
Error opening output files: Invalid argument
"""

FFMPEG6_STDERR = """\
Duration: 00:00:05.03, start: 0.000000, bitrate: 9 kb/s
Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p, 320x240, 5 kb/s, 32 fps, 32 tbr, 16384 tbn (default)
Output #0, f32le, to 'pipe:':
[out#0/f32le] Output file does not contain any stream
"""

WITH_AUDIO_STDERR = """\
Duration: 00:00:01.00, start: 0.000000, bitrate: 102 kb/s
Stream #0:0[0x1](und): Video: h264 (High), yuv420p, 320x240, 10 kb/s, 32 fps
Stream #0:1[0x2](und): Audio: aac (LC), 44100 Hz, mono, fltp, 70 kb/s
"""


def wave_shape(waveform):
    return tuple(int(size) for size in waveform.shape)


def wave_values(waveform):
    if hasattr(waveform, "detach"):
        return waveform.detach().cpu().reshape(-1).tolist()
    return list(waveform.values)


def assert_close(actual, expected, tolerance, label):
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: {actual} != {expected} (tol {tolerance})")


def probe_duration(path):
    raw = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        text=True,
    )
    return float(raw.strip())


def test_user_stderr_is_video_only():
    assert _stream_kinds(USER_STDERR) == ["Video"]
    assert_close(_container_duration_seconds(USER_STDERR), 5.03, 1e-9, "duration")
    assert _stream_kinds(FFMPEG6_STDERR) == ["Video"]
    assert _stream_kinds(WITH_AUDIO_STDERR) == ["Video", "Audio"]
    assert _resolve_silence_seconds(5.03, 0, 56.25) == 5.03
    assert _resolve_silence_seconds(5.03, 1, 2) == 2
    assert_close(_resolve_silence_seconds(5.03, 4, 10), 1.03, 1e-9, "tail")
    assert _resolve_silence_seconds(5.03, 0, 0) == 5.03
    assert _resolve_silence_seconds(5.03, 10, 1) == 0
    assert _resolve_silence_seconds(None, 0, 0) is None
    assert _resolve_silence_seconds(None, 0, 3) == 3


def test_stderr_fallback_without_ffprobe():
    original = vhs_utils._ffprobe_path
    vhs_utils._ffprobe_path = lambda: None
    try:
        capped = missing_audio_seconds("unused.mp4", 0, 56.25, USER_STDERR)
        assert_close(capped, 5.03, 1e-9, "capped silence")
        full = missing_audio_seconds("unused.mp4", 0, 0, USER_STDERR)
        assert_close(full, 5.03, 1e-9, "full silence")
        assert missing_audio_seconds("unused.mp4", 0, 56.25, WITH_AUDIO_STDERR) is None
        assert missing_audio_seconds("unused.mp4", 0, 0, "Error opening input") is None
        past_end = missing_audio_seconds("unused.mp4", 10, 56.25, USER_STDERR)
        assert past_end is None
    finally:
        vhs_utils._ffprobe_path = original


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


def _expect_silence(audio, seconds):
    shape = wave_shape(audio["waveform"])
    if shape[0] != 1 or shape[1] != 2:
        raise AssertionError(f"expected stereo batch, got {shape}")
    if audio["sample_rate"] != 44100:
        raise AssertionError(audio["sample_rate"])
    assert_close(shape[2] / 44100, seconds, 1.5 / 44100, "silence length")
    values = wave_values(audio["waveform"])
    if any(value != 0 for value in values):
        raise AssertionError("silence waveform contains non-zero samples")


def test_get_audio_on_real_files():
    tmp = tempfile.mkdtemp(prefix="vhs-silent-")
    try:
        silent = os.path.join(tmp, "silent.mp4")
        tone = os.path.join(tmp, "tone.wav")
        broken = os.path.join(tmp, "broken.mp4")
        _ffmpeg("-f", "lavfi", "-i", "color=c=black:s=320x240:r=32:d=5.03",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", silent)
        _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=0.4",
                "-ac", "1", tone)
        with open(broken, "wb") as handle:
            handle.write(b"not a video")

        duration = probe_duration(silent)
        # Same window as the reported failure: -t 56.25 on a ~5s video-only mp4.
        audio = get_audio(silent, 0, 56.25)
        _expect_silence(audio, duration)

        mapped = lazy_get_audio(silent, 1, 2)
        assert list(mapped) == ["waveform", "sample_rate"]
        _expect_silence(dict(mapped), 2)

        tail = get_audio(silent, 4, 0)
        _expect_silence(tail, duration - 4)

        real = get_audio(tone, 0, 0)
        real_shape = wave_shape(real["waveform"])
        if real_shape[0] != 1 or real_shape[1] != 1:
            raise AssertionError(f"mono wav decoded as {real_shape}")
        if real["sample_rate"] != 44100:
            raise AssertionError(real["sample_rate"])
        if max(abs(value) for value in wave_values(real["waveform"])) < 0.1:
            raise AssertionError("wav with audio decoded as silence")
        assert missing_audio_seconds(tone, 0, 0) is None
        assert missing_audio_seconds(silent, 0, 56.25) is not None

        for path in (broken, os.path.join(tmp, "missing.mp4")):
            try:
                get_audio(path)
            except Exception as exc:
                if "VHS failed to extract audio" not in str(exc):
                    raise AssertionError(f"unexpected error for {path}: {exc}") from exc
            else:
                raise AssertionError(f"{path} should still fail")

        preview = subprocess.run(silent_preview_args(0.2), capture_output=True, check=True)
        if not preview.stdout.startswith(b"\x1a\x45\xdf\xa3"):
            raise AssertionError("silent preview is not a webm")
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", "-"],
            input=preview.stdout, capture_output=True, check=True)
        if probe.stdout.decode().strip() != "audio":
            raise AssertionError(probe.stdout)

        raw = struct.pack("<" + "f" * len(wave_values(audio["waveform"])),
                          *wave_values(audio["waveform"]))
        muxed = os.path.join(tmp, "with-silence.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", silent,
             "-f", "f32le", "-ar", "44100", "-ac", "2", "-i", "-",
             "-c:v", "copy", "-c:a", "aac", "-shortest", muxed],
            input=raw, check=True)
        streams = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", muxed],
            text=True,
        ).split()
        if streams != ["video", "audio"]:
            raise AssertionError(streams)
    finally:
        shutil.rmtree(tmp)


def main():
    test_user_stderr_is_video_only()
    test_stderr_fallback_without_ffprobe()
    test_get_audio_on_real_files()
    print("ok")


if __name__ == "__main__":
    main()
