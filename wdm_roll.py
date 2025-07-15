import numpy as np
from scipy.special import betainc
from numpy import fft


def Phi_unit(f, A, d):
    """
    Meyer window function for the WDM wavelet transform.

    See Eq. (10) of Cornish (2020).
    `f` and half-width `A` are in units of Δf; `d` controls the smoothness.
    """
    B = 1.0 - 2.0 * A
    if B <= 0:
        if A >= 0.5:
            raise ValueError("A must be < 0.5 so that B = 1 − 2A > 0.")

    f_arr = np.asarray(f)
    result = np.zeros_like(f_arr, dtype=float)

    # Region 1: |f| < A → φ = 1
    mask1 = np.abs(f_arr) < A
    result[mask1] = 1.0

    # Region 2: A ≤ |f| < A + B → φ = cos(π/2 · p), p = I((|f| − A)/B; d, d)
    mask2 = (np.abs(f_arr) >= A) & (np.abs(f_arr) < (A + B))
    if np.any(mask2):
        z = (np.abs(f_arr[mask2]) - A) / B
        z = np.clip(z, 0.0, 1.0)
        p = betainc(d, d, z)
        result[mask2] = np.cos(np.pi * p / 2.0)

    return result.item() if np.isscalar(f) else result


def wdm_dT_dF(nt, nf, dt):
    """
    Returns (ΔT, ΔF) for WDM with nt time bins, nf freq bins, and input sampling dt.
    """
    return nf * dt, 1.0 / (2.0 * nf * dt)


def wdm_times_frequencies(nt, nf, dt):
    """
    Returns (ts, fs) for WDM:
    """
    ΔT, ΔF = wdm_dT_dF(nt, nf, dt)
    return np.arange(nt) * ΔT, np.arange(nf) * ΔF


def wdm_transform(x, nt, nf, A, d):
    n_total = nt * nf

    if x.shape[-1] != n_total:
        raise ValueError(f"len(x) must be nt*nf = {n_total}")
    if nt % 2 or nf % 2 or not (0 < A < 0.5) or d <= 0:
        raise ValueError("nt,nf even; 0<A<0.5; d>0 required.")

    # full FFT
    X_fft = fft.fft(x)

    # build phi window
    fs_full = fft.fftfreq(n_total)
    half = nt // 2
    fs_phi = np.concatenate([fs_full[:half], fs_full[-half:]])
    phi = Phi_unit(fs_phi / (1.0 / (2.0 * nf)), A, d) / np.sqrt(1.0 / (2.0 * nf))

    W = np.zeros((nt, nf), dtype=float)
    center = n_total // 2
    start = center - half

    # Handle m=1 to nf-1 (the regular frequency bands)
    for m in range(1, nf):
        shift = center - m * half
        rolled = np.roll(X_fft, shift)
        sl = rolled[start:start + nt]
        block = np.concatenate([sl[half:], sl[:half]])
        xnm = fft.ifft(block * phi)
        # parity factor: swap real/imag mapping
        n = np.arange(nt)
        parity = (n + m) % 2
        # even indices use imaginary, odd use real
        C = np.where(parity == 0, 1, 1j)
        W[:, m] = (np.sqrt(2.0) / nf) * np.real(C * xnm)

    # Handle m=0 (DC components) - store in even indices of column 0
    shift = center  # No shift for DC
    rolled = np.roll(X_fft, shift)
    sl = rolled[start:start + nt]
    block = np.concatenate([sl[half:], sl[:half]])
    xnm = fft.ifft(block * phi)
    # DC components go to even indices with sqrt(2) normalization and factor of 1/2
    for n in range(0, nt, 2):
        W[n, 0] = np.real(xnm[n]) * np.sqrt(2.0) / (2.0 * nf)

    # Handle m=nf (Nyquist components) - store in odd indices of column 0
    shift = center - nf * half
    rolled = np.roll(X_fft, shift)
    sl = rolled[start:start + nt]
    block = np.concatenate([sl[half:], sl[:half]])
    xnm = fft.ifft(block * phi)
    # Nyquist components go to odd indices with sqrt(2) normalization and factor of 1/2
    for n in range(1, nt, 2):
        W[n, 0] = np.real(xnm[n - 1]) * np.sqrt(2.0) / (2.0 * nf)

    return W




def wdm_inverse_transform(W, A, d):
    nt, nf = W.shape
    n_total = nt * nf
    # validation omitted for brevity
    fs_full = fft.fftfreq(n_total)
    half = nt // 2
    fs_phi = np.concatenate([fs_full[:half], fs_full[-half:]])
    phi = Phi_unit(fs_phi / (1.0/(2.0*nf)), A, d)

    n = np.arange(nt)[:, None]
    m = np.arange(nf)[None, :]
    parity = (n + m) % 2
    C = np.where(parity == 0, 1, 1j)
    ylm = np.zeros((nt, nf), complex)
    ylm[:,1:] = (C[:,1:] * W[:,1:] / np.sqrt(2.0)) * nf

    Y = fft.fft(ylm, axis=0)
    X_rec = np.zeros(n_total, complex)
    center = n_total // 2
    start = center - half

    for m in range(1, nf):
        block = Y[:, m] * phi
        pos, neg = block[:half], block[half:]
        temp = np.zeros(n_total, complex)
        temp[start:start+nt] = np.concatenate([neg, pos])
        X_rec += np.roll(temp, m * half - center)

    return np.real(fft.ifft(X_rec))
