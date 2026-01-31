import ffmpeg

def mp4_to_wav_bytes(mp4_bytes: bytes) -> bytes:
    """
    Extract audio from MP4 bytes and return WAV bytes
    """
    out, _ = (
        ffmpeg
        .input('pipe:0')
        .output(
            'pipe:1',
            format='wav',
            acodec='pcm_s16le',
            ac=1,
            ar=22050
        )
        .run(
            input=mp4_bytes,
            capture_stdout=True,
            capture_stderr=True
        )
    )
    return out