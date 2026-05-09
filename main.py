import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# ============================================
# CONFIG
# ============================================

fs = 50e6
count = 1_000_000

files = {
    "Clear Sky": "data/clear_sky_100mb.bin",
    "Spoof 1": "data/spoof_1_100mb.bin",
    "Spoof 2": "data/spoof_2_100mb.bin",
    "Spoof 4": "data/spoof_4_100mb.bin",
}

# ============================================
# LOAD IQ
# ============================================

def load_iq(path, count):

    raw = np.fromfile(path, dtype=np.int16, count=count*2)

    I = raw[0::2].astype(np.float32)
    Q = raw[1::2].astype(np.float32)

    iq = I + 1j * Q

    print("dtype:", iq.dtype)
    print("finite:", np.isfinite(iq).all())
    print("mean abs:", np.mean(np.abs(iq)))

    iq = iq - np.mean(iq)  # DC removal

    return iq

# ============================================
# ANALYSIS
# ============================================

for label, path in files.items():

    print("\n" + "="*60)
    print(label)
    print("="*60)

    iq = load_iq(path, count)

    # ----------------------------------------
    # BASIC STATS
    # ----------------------------------------

    print("Samples:", len(iq))

    print("Mean I:", np.mean(np.real(iq)))
    print("Mean Q:", np.mean(np.imag(iq)))

    print("STD I:", np.std(np.real(iq)))
    print("STD Q:", np.std(np.imag(iq)))

    print("Min:", np.min(np.real(iq)))
    print("Max:", np.max(np.real(iq)))

    # ----------------------------------------
    # NAN / INF
    # ----------------------------------------

    print("NaNs:", np.isnan(iq).sum())
    print("Infs:", np.isinf(iq).sum())

    # ----------------------------------------
    # ZERO SAMPLES
    # ----------------------------------------

    zero_count = np.sum(iq == 0)

    print("Zero samples:", zero_count)

    # ----------------------------------------
    # CLIPPING
    # ----------------------------------------

    clipping = np.sum(
        (np.real(iq) >= 32767) |
        (np.real(iq) <= -32768)
    )

    print("Clipped samples:", clipping)

    # ----------------------------------------
    # SIGNAL POWER
    # ----------------------------------------

    power = np.mean(np.abs(iq)**2)

    print("Average power:", power)

    # ========================================
    # PLOTS
    # ========================================

    # ----------------------------------------
    # HISTOGRAM
    # ----------------------------------------

    plt.figure(figsize=(10,4))

    plt.hist(
        np.real(iq[:100000]),
        bins=200
    )

    plt.title(f"{label} - Amplitude Histogram")
    plt.xlabel("Amplitude")
    plt.ylabel("Count")

    plt.show()

    # ----------------------------------------
    # TIME DOMAIN
    # ----------------------------------------

    plt.figure(figsize=(12,4))

    plt.plot(np.real(iq[:5000]))

    plt.title(f"{label} - I Component")

    plt.show()

    # ----------------------------------------
    # PSD
    # ----------------------------------------

    f, Pxx = welch(
        iq,
        fs=fs,
        nperseg=4096
    )

    plt.figure(figsize=(12,4))

    plt.semilogy(f, Pxx)

    plt.title(f"{label} - PSD")
    plt.xlabel("Frequency")
    plt.ylabel("PSD")

    plt.show()

    # ----------------------------------------
    # CONSTELLATION
    # ----------------------------------------

    plt.figure(figsize=(6,6))

    plt.scatter(
        np.real(iq[:50000]),
        np.imag(iq[:50000]),
        s=1,
        alpha=0.3
    )

    plt.title(f"{label} - IQ Constellation")
    plt.xlabel("I")
    plt.ylabel("Q")

    plt.show()