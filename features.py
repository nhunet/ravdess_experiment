"""Feature extractors: MFCC, LogMel, LPCC, Chroma, Statistical, Prosody."""

import numpy as np
import librosa

import config


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _mean_std(feat: np.ndarray) -> np.ndarray:
    """Compute mean and std across time axis for a 2D feature matrix."""
    return np.concatenate([np.mean(feat, axis=1), np.std(feat, axis=1)])


def _resample_contour(contour: np.ndarray, n_points: int,
                      interpolate_nan: bool = False) -> np.ndarray:
    if len(contour) == 0:
        return np.zeros(n_points)
    if interpolate_nan:
        nans = np.isnan(contour)
        if np.all(nans):
            return np.zeros(n_points)
        x = np.arange(len(contour))
        contour = np.interp(x, x[~nans], contour[~nans])
    indices = np.linspace(0, len(contour) - 1, n_points).astype(int)
    return contour[indices].astype(np.float64)


# ─── Feature Extractors ─────────────────────────────────────────────────────

def extract_mfcc(y: np.ndarray, sr: int = config.SR_HANDCRAFTED) -> np.ndarray:
    """MFCC + delta + delta2 → 240 dims."""
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=config.N_MFCC,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([_mean_std(mfcc), _mean_std(delta), _mean_std(delta2)])


def extract_logmel(y: np.ndarray, sr: int = config.SR_HANDCRAFTED) -> np.ndarray:
    """Log-Mel Spectrogram → 256 dims."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=config.N_MELS,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return _mean_std(log_mel)


def extract_lpcc(y: np.ndarray, sr: int = config.SR_HANDCRAFTED) -> np.ndarray:
    """LPCC via LPC → cepstral conversion → 28 dims."""
    order = config.N_LPCC
    frames = librosa.util.frame(y, frame_length=config.N_FFT,
                                hop_length=config.HOP_LENGTH)
    all_lpcc = []
    for frame in frames.T:
        windowed = frame * np.hanning(len(frame))
        a = librosa.lpc(windowed, order=order)
        cepstrum = np.zeros(order)
        cepstrum[0] = -a[1]
        for n in range(1, order):
            s = 0.0
            for k in range(1, n):
                s += (k + 1) / (n + 1) * cepstrum[k] * a[n - k] if (n - k) < len(a) else 0.0
            cepstrum[n] = -a[n + 1] - s if (n + 1) < len(a) else -s
        all_lpcc.append(cepstrum)
    if not all_lpcc:
        return np.zeros(order * 2)
    lpcc_matrix = np.array(all_lpcc).T  # (order, frames)
    return _mean_std(lpcc_matrix)


def extract_chroma(y: np.ndarray, sr: int = config.SR_HANDCRAFTED) -> np.ndarray:
    """Chroma + Spectral Contrast + Tonnetz → 50 dims."""
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
    )
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
    )
    harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
    return np.concatenate([_mean_std(chroma), _mean_std(contrast), _mean_std(tonnetz)])


def extract_statistical(y: np.ndarray, sr: int = config.SR_HANDCRAFTED) -> np.ndarray:
    """ZCR, RMS, Centroid, Bandwidth, Rolloff → 10 dims."""
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=config.N_FFT,
                                              hop_length=config.HOP_LENGTH)
    rms = librosa.feature.rms(y=y, frame_length=config.N_FFT,
                               hop_length=config.HOP_LENGTH)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=config.N_FFT,
                                                  hop_length=config.HOP_LENGTH)
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=config.N_FFT,
                                             hop_length=config.HOP_LENGTH)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=config.N_FFT,
                                                hop_length=config.HOP_LENGTH)
    return np.concatenate([
        _mean_std(zcr), _mean_std(rms), _mean_std(centroid),
        _mean_std(bw), _mean_std(rolloff),
    ])


def extract_prosody(y: np.ndarray, sr: int = config.SR_HANDCRAFTED) -> np.ndarray:
    """Prosody features → 34 dims. See HANDOFF breakdown table."""
    # F0 extraction
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=50, fmax=500, sr=sr,
        frame_length=config.N_FFT, hop_length=config.HOP_LENGTH,
    )
    f0_voiced = f0[~np.isnan(f0)]

    # F0 statistics (6)
    if len(f0_voiced) >= 2:
        f0_mean = np.mean(f0_voiced)
        f0_std = np.std(f0_voiced)
        f0_min = np.min(f0_voiced)
        f0_max = np.max(f0_voiced)
        f0_range = f0_max - f0_min
        t = np.arange(len(f0_voiced))
        f0_slope = np.polyfit(t, f0_voiced, 1)[0] if len(t) > 1 else 0.0
    else:
        f0_mean = f0_std = f0_min = f0_max = f0_range = f0_slope = 0.0

    # Jitter (3)
    if len(f0_voiced) >= 3:
        periods = 1.0 / f0_voiced
        pdiff = np.abs(np.diff(periods))
        jitter_local = np.mean(pdiff) / np.mean(periods)
        jitter_abs = np.mean(pdiff)
        rap_vals = []
        for i in range(1, len(periods) - 1):
            avg3 = np.mean(periods[i - 1:i + 2])
            rap_vals.append(abs(periods[i] - avg3))
        jitter_rap = np.mean(rap_vals) / np.mean(periods) if rap_vals else 0.0
    else:
        jitter_local = jitter_abs = jitter_rap = 0.0

    # Shimmer (3)
    rms_all = librosa.feature.rms(
        y=y, frame_length=config.N_FFT, hop_length=config.HOP_LENGTH,
    )[0]
    voiced_mask = ~np.isnan(f0)
    min_len = min(len(rms_all), len(voiced_mask))
    rms_voiced = rms_all[:min_len][voiced_mask[:min_len]]

    if len(rms_voiced) >= 3:
        adiff = np.abs(np.diff(rms_voiced))
        shimmer_local = np.mean(adiff) / np.mean(rms_voiced) if np.mean(rms_voiced) > 0 else 0.0
        ratios = rms_voiced[1:] / (rms_voiced[:-1] + 1e-10)
        shimmer_db = np.mean(np.abs(20 * np.log10(ratios + 1e-10)))
        apq3_vals = []
        for i in range(1, len(rms_voiced) - 1):
            avg3 = np.mean(rms_voiced[i - 1:i + 2])
            apq3_vals.append(abs(rms_voiced[i] - avg3))
        shimmer_apq3 = np.mean(apq3_vals) / np.mean(rms_voiced) if apq3_vals else 0.0
    else:
        shimmer_local = shimmer_db = shimmer_apq3 = 0.0

    # Energy dynamics (4)
    rms_mean = float(np.mean(rms_all))
    rms_std = float(np.std(rms_all))
    rms_max = float(np.max(rms_all))
    delta_rms_std = float(np.std(np.diff(rms_all))) if len(rms_all) > 1 else 0.0

    # Voiced ratio (1)
    voiced_ratio = np.sum(~np.isnan(f0)) / len(f0) if len(f0) > 0 else 0.0

    # Pause ratio (1)
    pause_thresh = 0.02 * rms_max if rms_max > 0 else 0.0
    pause_ratio = np.sum(rms_all < pause_thresh) / len(rms_all) if len(rms_all) > 0 else 0.0

    # Spectral flux (1)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    spectral_flux = float(np.mean(onset_env))

    # F0 contour envelope (8 points)
    f0_envelope = _resample_contour(f0, n_points=8, interpolate_nan=True)

    # Energy contour envelope (7 points) — 7 not 8 to match 34-dim total
    energy_envelope = _resample_contour(rms_all, n_points=7, interpolate_nan=False)

    return np.concatenate([
        [f0_mean, f0_std, f0_min, f0_max, f0_range, f0_slope],
        [jitter_local, jitter_abs, jitter_rap],
        [shimmer_local, shimmer_db, shimmer_apq3],
        [rms_mean, rms_std, rms_max, delta_rms_std],
        [voiced_ratio],
        [pause_ratio],
        [spectral_flux],
        f0_envelope,
        energy_envelope,
    ])


# ─── Feature Registry ───────────────────────────────────────────────────────

FEATURE_EXTRACTORS = {
    "MFCC": (extract_mfcc, 240),
    "LogMel": (extract_logmel, 256),
    "LPCC": (extract_lpcc, 28),
    "Chroma": (extract_chroma, 50),
    "Statistical": (extract_statistical, 10),
    "Prosody": (extract_prosody, 34),
}


def extract_combined_mfcc_prosody(y: np.ndarray,
                                   sr: int = config.SR_HANDCRAFTED) -> np.ndarray:
    """MFCC + Prosody concatenated → 274 dims."""
    return np.concatenate([extract_mfcc(y, sr), extract_prosody(y, sr)])


FEATURE_EXTRACTORS["MFCC+Prosody"] = (extract_combined_mfcc_prosody, 274)


def build_feature_matrix(waveforms: list[np.ndarray], labels: np.ndarray,
                         extractor_fn, expected_dim: int,
                         sr: int = config.SR_HANDCRAFTED) -> tuple[np.ndarray, np.ndarray]:
    """Extract features for all waveforms, dropping samples that fail."""
    X_list, y_list = [], []
    for i, y in enumerate(waveforms):
        try:
            feat = extractor_fn(y, sr)
            if len(feat) != expected_dim:
                print(f"[WARN] Sample {i}: got {len(feat)} dims, expected {expected_dim} — skipped")
                continue
            if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
                feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            X_list.append(feat)
            y_list.append(labels[i])
        except Exception as e:
            print(f"[WARN] Sample {i} failed: {e} — skipped")
    return np.array(X_list), np.array(y_list)
