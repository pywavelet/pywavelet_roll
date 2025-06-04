import numpy as np
from scipy.special import betainc
from numpy.fft import fft, ifft, fftfreq

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
    if np.any(mask2) and B > 1e-12:
        z = (np.abs(f_arr[mask2]) - A) / B
        z = np.clip(z, 0.0, 1.0)
        p = betainc(d, d, z)
        result[mask2] = np.cos(np.pi * p / 2.0)

    return result.item() if np.isscalar(f) else result


def wdm_dT_dF(nt, nf, dt):
    """
    Returns (ΔT, ΔF) for WDM with nt time bins, nf freq bins, and input sampling dt.
      ΔT = nf · dt
      ΔF = 1 / (2 · nf · dt)
    """
    ΔT = nf * dt
    ΔF = 1.0 / (2.0 * nf * dt)
    return (ΔT, ΔF)


def wdm_times_frequencies(nt, nf, dt):
    """
    Returns (ts, fs) for WDM: 
      ts = ΔT · [0..nt−1],  fs = ΔF · [0..nf−1],
    where (ΔT, ΔF) = wdm_dT_dF(nt,nf,dt).
    """
    ΔT, ΔF = wdm_dT_dF(nt, nf, dt)
    ts = np.arange(nt) * ΔT
    fs = np.arange(nf) * ΔF
    return ts, fs


def wdm_transform(x, nt, nf, A, d):
    """
    Forward WDM transform using np.roll + slice + reorder.

    Args:
      x   : 1D real array of length n_total = nt*nf
      nt  : number of time bins (even)
      nf  : number of frequency bins (even)
      A,d : Meyer window parameters (0 < A < 0.5, d > 0)

    Returns:
      W : real array of shape (nt, nf) with WDM coefficients.

    Implementation notes:
      1) Compute full FFT X_fft of length n_total.
      2) Build a single φ-window of length nt by sampling fftfreq(n_total).
      3) For each m = 1..nf−1:
         a) Compute shift = (n_total//2) − (m*(nt//2)).
         b) rolled = np.roll(X_fft, shift).
         c) slice_full = rolled[start : start+nt], where start = center − (nt//2).
         d) Reorder slice_full → [ positive_half, negative_half ].
         e) Multiply by φ-window, IFFT to get complex xnm_time of length nt.
         f) Multiply by C(n,m) = (1 if (n+m)%2==0 else 1j), take real, scale by √2/nf.
      4) Column m=0 is zero.
    """

    n_total = nt * nf
    if nt % 2 != 0 or nf % 2 != 0:
        raise ValueError("nt and nf must both be even.")
    if x.shape[-1] != n_total:
        raise ValueError(f"len(x)={x.shape[-1]} must equal nt*nf={n_total}.")
    if not (0 < A < 0.5):
        raise ValueError("A must be in (0, 0.5).")

    # 1) Compute FFT of full signal
    X_fft = fft(x)  # length = n_total

    # 2) Build φ-window of length=nt
    _, dF_phi = wdm_dT_dF(nt, nf, 1.0)
    fs_full = fftfreq(n_total)  # length = n_total
    half = nt // 2
    fs_phi = np.concatenate([fs_full[:half], fs_full[-half:]])  # length = nt
    phi_window = Phi_unit(fs_phi / dF_phi, A, d) / np.sqrt(dF_phi)  # length = nt

    # 3) Prepare output array
    W = np.zeros((nt, nf), dtype=float)

    center_idx = n_total // 2
    start = center_idx - half  # starting index in rolled array

    # 4) For each sub-band m=1..nf-1:
    for m in range(1, nf):
        freq_bin = m * half
        shift = center_idx - freq_bin
        rolled = np.roll(X_fft, shift)

        # Slice exactly nt samples around center
        slice_full = rolled[start : start + nt]  # length = nt

        # slice_full is ordered [ negative_half | positive_half ]
        neg_half = slice_full[:half]
        pos_half = slice_full[half:]
        # We need [pos_half | neg_half] 
        block = np.concatenate([pos_half, neg_half])

        # Multiply by φ-window and IFFT
        xnm_time = ifft(block * phi_window)  # length = nt, complex

        # Build parity factor C(n,m) = 1 if (n+m)%2==0 else 1j
        n_idx = np.arange(nt)
        parity = (n_idx + m) % 2
        C_col = np.where(parity == 0, 1.0 + 0.0j, 0.0 + 1.0j)

        # Real part of conj(C)·xnm_time, scaled by √2/nf
        W[:, m] = (np.sqrt(2.0) / nf) * np.real(np.conj(C_col) * xnm_time)

    # Column m=0 remains zero
    return W


def wdm_inverse_transform(W, A, d):
    """
    Inverse WDM using the “roll + reorder + sum” approach:

    Args:
      W : real array (nt, nf) of WDM coefficients
      A,d : Meyer window parameters

    Returns:
      1D real signal of length n_total = nt * nf.
    """

    nt, nf = W.shape
    if nt % 2 != 0 or nf % 2 != 0:
        raise ValueError("nt and nf must both be even.")
    n_total = nt * nf
    if not (0 < A < 0.5):
        raise ValueError("A must be in (0, 0.5).")

    # 1) Build φ-window (same as forward)
    _, dF_phi = wdm_dT_dF(nt, nf, 1.0)
    fs_full = fftfreq(n_total)
    half = nt // 2
    fs_phi = np.concatenate([fs_full[:half], fs_full[-half:]])
    phi_window = Phi_unit(fs_phi / dF_phi, A, d) / np.sqrt(dF_phi)

    # 2) Build parity matrix C(n,m) and form ylm = C · W / sqrt(2) · nf
    n_idx = np.arange(nt)[:, None]  # shape (nt,1)
    m_idx = np.arange(nf)[None, :]  # shape (1,nf)
    parity = (n_idx + m_idx) % 2
    Cmat = np.where(parity == 0, 1.0 + 0.0j, 0.0 + 1.0j)  # shape (nt,nf)

    # ylm shape = (nt,nf)
    # (for m=0 we keep column zero)
    ylm = np.zeros((nt, nf), dtype=complex)
    ylm[:, 1:] = (Cmat[:, 1:] * W[:, 1:] / np.sqrt(2.0)) * nf

    # 3) FFT each column along axis=0
    Y = np.fft.fft(ylm, axis=0)  # shape (nt,nf)

    # 4) Reconstruct full-spectrum X_recon of length n_total
    X_recon = np.zeros(n_total, dtype=complex)
    center_idx = n_total // 2
    start = center_idx - half

    for m in range(1, nf):
        # Build the nt-length “block” in frequency space: Y[:,m] * φ-window
        block = Y[:, m] * phi_window  # length = nt

        pos_half = block[:half]
        neg_half = block[half:]
        block_full = np.concatenate([neg_half, pos_half])  
        # → this arranges [ negative_half | positive_half ] at indices [start:start+nt]

        # Place block_full into an otherwise-zero length-n_total array, then roll:
        temp_full = np.zeros(n_total, dtype=complex)
        temp_full[start : start + nt] = block_full

        # Roll so that the “center” (index=center_idx) goes to freq_bin=m*half
        freq_bin = m * half
        shift = freq_bin - center_idx
        X_recon += np.roll(temp_full, shift)

    # 5) IFFT back to time domain
    x_time = ifft(X_recon)
    return np.real(x_time)




def wdm_inverse_direct(W, A, d):
    """
    Inverse WDM transform (direct indexing, no roll -- i think i might have a bug :( )). Given W shape=(nt,nf),
    returns the real 1D signal x of length = nt*nf.

    Algorithm:
      nt, nf must be even, W[:,0]=0. For m=1..nf-1:
      1) Build ylm[n,m] = C(n,m)*W[n,m]/√2 * nf, where C(n,m)=1 or 1j by parity.
      2) Y[:,m] = FFT(ylm[:,m]) (length=nt).
      3) block = Y[:,m] * φ_window (length=nt, with φ_window from fftfreq(nt*nf)).
      4) Split block = [pos_half|neg_half], where pos_half = block[:nt//2], neg_half=block[nt//2:].
      5) l0 = m*(nt//2).  Add pos_half into X[l0 : l0+half]; add neg_half into X[l0-half : l0].
      6) After looping m=1..nf-1, IFFT(X) → time, return real part.
    """
    nt, nf = W.shape
    if nt % 2 != 0 or nf % 2 != 0:
        raise ValueError("nt and nf must both be even.")
    n_total = nt * nf
    if not (0 < A < 0.5):
        raise ValueError("A must be in (0,0.5).")

    # 1) Build φ_window of length=nt
    _, dF_phi = wdm_dT_dF(nt, nf, 1.0)
    fs_full = fftfreq(n_total)
    half = nt // 2
    fs_phi = np.concatenate((fs_full[:half], fs_full[-half:]))  # length=nt
    phi_window = Phi_unit(fs_phi / dF_phi, A, d) / np.sqrt(dF_phi)

    # 2) Build parity matrix C(n,m) and compute ylm
    n_idx = np.arange(nt)[:, None]  # shape (nt,1)
    m_idx = np.arange(nf)[None, :]  # shape (1,nf)
    parity = (n_idx + m_idx) % 2
    Cmat = np.where(parity == 0, 1.0 + 0.0j, 0.0 + 1.0j)  # shape (nt,nf)

    # ylm[n,m] = C(n,m) * W[n,m] / sqrt(2) * nf  (for m>=1; ylm[:,0]=0)
    ylm = np.zeros((nt, nf), dtype=complex)
    ylm[:, 1:] = (Cmat[:, 1:] * W[:, 1:] / np.sqrt(2.0)) * nf

    # 3) FFT each column of ylm along axis=0 → Y (shape (nt,nf))
    Y = np.fft.fft(ylm, axis=0)

    # 4) Reconstruct full-spectrum X_recon of length n_total by adding each band’s contributions
    X_recon = np.zeros(n_total, dtype=complex)
    half = nt // 2

    for m in range(1, nf):
        # Build the nt-length “block” = [pos_half | neg_half]
        block = Y[:, m] * phi_window  # length = nt

        # Split block into pos/neg
        pos_half = block[:half]
        neg_half = block[half:]

        # l0 = m*(nt/2)
        l0 = m * half

        # Add pos_half into X_recon[l0 : l0 + half]
        X_recon[l0 : l0 + half] += pos_half

        # Add neg_half into X_recon[l0 - half : l0]
        X_recon[l0 - half : l0] += neg_half

        # (No extra “conj” handling needed because ylm was built with the correct parity factor)

    # 5) IFFT back to time domain
    x_time = ifft(X_recon)
    return np.real(x_time)

