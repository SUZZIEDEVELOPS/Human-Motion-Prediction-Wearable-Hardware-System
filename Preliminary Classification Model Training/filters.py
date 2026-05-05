#from scipy.signal import butter, filtfilt
#def butter_lowpass_filter(data, cutoff, fs, order=4):
 #   nyq = 0.5 * fs
  #  normal_cutoff = cutoff / nyq
   # b, a = butter(order, normal_cutoff, btype='low', analog=False)
    #y = filtfilt(b, a, data)
    #return y


import numpy as np
from scipy.signal import butter, filtfilt, lfilter

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    Low-pass filter with a safe fallback when the signal is too short for filtfilt.
    """
    x = np.asarray(data, dtype=float)

    # Handle NaN/inf early (filtfilt will break / propagate NaNs)
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=np.nanmedian(x), posinf=0.0, neginf=0.0)

    # Validate fs and cutoff
    if fs is None or fs <= 0:
        return x
    nyq = 0.5 * fs
    if cutoff is None or cutoff <= 0 or cutoff >= nyq:
        return x

    normal_cutoff = cutoff / nyq  # must be in (0,1)
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    # filtfilt requires len(x) > padlen
    padlen = 3 * (max(len(a), len(b)) - 1)  # SciPy rule of thumb
    if len(x) <= padlen:
        # Fallback: one-pass filter (works for short segments, but adds phase delay)
        return lfilter(b, a, x)

    return filtfilt(b, a, x)
