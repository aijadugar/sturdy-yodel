# utils/preprocessing.py
import librosa
import numpy as np
import io
def audio_to_logmel(audio_bytes, sr=22050, n_mels=128, max_len_sec=5):
    """
    Converts raw audio bytes into a normalized log-Mel spectrogram.
    Pads or trims audio to max_len_sec seconds.
    """
    # Load audio from bytes
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)
    
    # Pad or trim to fixed length
    target_len = sr * max_len_sec
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]
    
    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel)
    
    # Normalize
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    
    return log_mel  # shape: (128, time)