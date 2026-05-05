import numpy as np

TARGET_KEYS = [
    "LEFTARM_AccY_mean",
    "LEFTARM_AccZ_mean",
    "RIGHTLEG_AccY_mean",
    "RIGHTLEG_AccZ_mean",
    "LEFTARM_AccX_dominant_freq",
    "LEFTARM_AccY_dominant_freq",
    "LEFTARM_AccZ_dominant_freq",
    "LEFTARM_AccMag_dominant_freq",
    "RIGHTLEG_AccX_dominant_freq",
    "RIGHTLEG_AccY_dominant_freq",
    "RIGHTLEG_AccZ_dominant_freq",
    "RIGHTLEG_AccMag_dominant_freq",
]

def features_to_target(feature_dict):
    """
    Extracts the 10 regression target values from a feature dictionary.
    Returns a numpy array of shape (10,) or None if any key is missing
    or contains a non-finite value.
    """
    try:
        arr = np.array([feature_dict[k] for k in TARGET_KEYS], dtype=float)
        if not np.all(np.isfinite(arr)):
            return None
        return arr
    except KeyError:
        return None