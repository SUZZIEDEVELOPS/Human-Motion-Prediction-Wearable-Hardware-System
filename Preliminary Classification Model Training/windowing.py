import numpy as np

def sliding_window(signal, size, step):
    windows = []
    for i in range(0, len(signal)-size, step):
        windows.append(signal[i:i+size])
    return windows
