#Feature extraction
import numpy as np

def extract_features(window, feature_cols, sampling_rate):
    """
    window: a pandas DataFrame window
    feature_cols: list of column names to compute features on (e.g. )
    sampling_rate: Hz
    returns: dict of features for this window
    """
    features = {}  # ✅ FIX: define dict

    for axis in feature_cols:
        signal = window[axis].to_numpy()

        # Time-domain
        features[f'{axis}_std'] = float(np.std(signal))
        features[f'{axis}_mean'] = float(np.mean(signal))
        features[f'{axis}_min'] = float(np.min(signal))
        features[f'{axis}_max'] = float(np.max(signal))
        features[f'{axis}_range'] = float(np.max(signal) - np.min(signal))

        signal = np.asarray(signal, dtype=float)
        signal = signal[np.isfinite(signal)]
        if len(signal) < 2:
            continue

        # Frequency-domain
        signal_d = signal - np.mean(signal)
        fft_vals = np.fft.fft(signal_d)
        fft_freqs = np.fft.fftfreq(len(signal), d=1.0 / sampling_rate)

        # Dominant frequency (ignore 0 Hz)
        idx = np.argmax(np.abs(fft_vals[1:])) + 1
        features[f'{axis}_dominant_freq'] = float(fft_freqs[idx])

        # Spectral energy
        features[f'{axis}_spectral_energy'] = float(np.sum(np.abs(fft_vals) ** 2))

        # Skewness & kurtosis
        std = np.std(signal)
        if std == 0:
            features[f'{axis}_skewness'] = 0.0
            features[f'{axis}_kurtosis'] = 0.0
        else:
            z = (signal - np.mean(signal)) / std
            features[f'{axis}_skewness'] = float(np.mean(z ** 3))
            features[f'{axis}_kurtosis'] = float(np.mean(z ** 4) - 3)



    return features
