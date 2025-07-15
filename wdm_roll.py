import numpy as np
from scipy.special import betainc
from numpy import fft


def Phi_unit(f, A, d):
    """
    Meyer window function Φ(ω) from Cornish Eq. (11).

    Φ(ω) = 1/√ΔΩ for |ω| < A
    Φ(ω) = (1/√ΔΩ) cos[νd(π/2 * (|ω|-A)/B)] for A ≤ |ω| ≤ A+B

    where νd(x) is the normalized incomplete Beta function (Eq. 12)
    and B = ΔΩ - 2A with constraint 2A + B = ΔΩ.

    Args:
        f: frequency in units of ΔF
        A: half-width parameter (0 < A < 0.5)
        d: steepness parameter controlling edge smoothness
    """
    B = 1.0 - 2.0 * A  # From constraint 2A + B = ΔΩ = 1 (normalized)
    if B <= 0:
        if A >= 0.5:
            raise ValueError("A must be < 0.5 so that B = 1 − 2A > 0.")

    f_arr = np.asarray(f)
    result = np.zeros_like(f_arr, dtype=float)

    # Region 1: |ω| < A → Φ = 1/√ΔΩ (normalized to 1)
    mask1 = np.abs(f_arr) < A
    result[mask1] = 1.0

    # Region 2: A ≤ |ω| ≤ A + B → Φ = (1/√ΔΩ) cos[νd(π/2 * (|ω|-A)/B)]
    mask2 = (np.abs(f_arr) >= A) & (np.abs(f_arr) < (A + B))
    if np.any(mask2):
        z = (np.abs(f_arr[mask2]) - A) / B
        z = np.clip(z, 0.0, 1.0)
        # νd(x) from Eq. (12): normalized incomplete Beta function
        nu_d = betainc(d, d, z)
        result[mask2] = np.cos(np.pi * nu_d / 2.0)

    return result.item() if np.isscalar(f) else result


def wdm_dT_dF(Nt, Nf, delta_t):
    """
    WDM time-frequency grid parameters.

    From Cornish: ΔT = Nf * Δt, ΔF = 1/(2*Nf*Δt)
    where ΔT is time pixel width, ΔF is frequency pixel width.
    """
    Delta_T = Nf * delta_t
    Delta_F = 1.0 / (2.0 * Nf * delta_t)
    return Delta_T, Delta_F


def wdm_times_frequencies(Nt, Nf, delta_t):
    """
    Generate WDM time-frequency coordinate arrays.
    """
    Delta_T, Delta_F = wdm_dT_dF(Nt, Nf, delta_t)
    t_n = np.arange(Nt) * Delta_T  # Time coordinates t_n
    f_m = np.arange(Nf) * Delta_F  # Frequency coordinates f_m
    return t_n, f_m


def wdm_transform(x, Nt, Nf, A, d):
    """
    Forward WDM transform implementing Cornish Eqs. (16) and (17).

    Eq. (16): w_nm = √2 (-1)^nm ℜ[C_nm * x_m[n]]
    Eq. (17): x_m[n] = Σ_{l=-Nt/2}^{Nt/2-1} exp(-2πiln/Nt) X[l + mNt/2] Φ[l]

    Args:
        x: Input time series of length N = Nt*Nf
        Nt: Number of time bins (must be even)
        Nf: Number of frequency bins (must be even)
        A, d: Meyer window parameters from Eqs. (11-12)

    Returns:
        w_nm: WDM coefficients, shape (Nt, Nf)
              Regular bands in columns 1 to Nf-1
              DC components in even rows of column 0
              Nyquist components in odd rows of column 0
    """
    N = Nt * Nf  # Total data length

    # Validation
    if x.shape[-1] != N:
        raise ValueError(f"len(x) must be Nt*Nf = {N}")
    if Nt % 2 or Nf % 2 or not (0 < A < 0.5) or d <= 0:
        raise ValueError("Nt,Nf even; 0<A<0.5; d>0 required.")

    # Step 1: Compute X[l] = FFT(x) (Cornish notation)
    X = fft.fft(x) # [Dc, +ive freqs, -ive freqs, nyquist]

    # Step 2: Build frequency domain window Φ[l] from Eq. (11)
    # Note: "discrete Fourier samples are evaluated at f = l*Δf" (Cornish text)
    l_freqs = fft.fftfreq(N)  # l = 0,1,...,N-1 mapped to [-0.5, 0.5)
    half = Nt // 2

    # Reorder for l = -Nt/2, ..., Nt/2-1 as in Eq. (17)
    l_indices = np.concatenate([l_freqs[:half], l_freqs[-half:]])  # length Nt
    Delta_F_norm = 1.0 / (2.0 * Nf)  # Normalized ΔF for unit sampling
    Phi = Phi_unit(l_indices / Delta_F_norm, A, d) / np.sqrt(Delta_F_norm)

    # Step 3: Initialize output w_nm
    w_nm = np.zeros((Nt, Nf), dtype=float)

    # Grid indices for parity calculations
    center_idx = N // 2
    start_idx = center_idx - half

    # Step 4: Process regular frequency bands m = 1, 2, ..., Nf-1
    # Following Eqs. (16) and (17)
    for m in range(1, Nf):
        # Eq. (17): Extract X[l + mNt/2] using frequency shift
        freq_shift = m * half  # mNt/2 term from Eq. (17)
        roll_amount = center_idx - freq_shift
        X_shifted = np.roll(X, roll_amount)

        # Extract Nt samples: X[l + mNt/2] for l = -Nt/2, ..., Nt/2-1
        X_slice = X_shifted[start_idx:start_idx + Nt]

        # Reorder to match IFFT convention: [positive freqs, negative freqs]
        X_reordered = np.concatenate([X_slice[half:], X_slice[:half]])

        # Apply window Φ[l] and compute IFFT to get x_m[n] from Eq. (17)
        x_m = fft.ifft(X_reordered * Phi)

        # Eq. (16): Apply parity factors C_nm and phase (-1)^nm
        n_indices = np.arange(Nt)
        parity = (n_indices + m) % 2
        # C_nm from Eq. (10): C_nm = 1 for (n+m) even, C_nm = i for (n+m) odd
        C_nm = np.where(parity == 0, 1.0 + 0.0j, 0.0 + 1.0j)

        # Final formula: w_nm = √2 (-1)^nm ℜ[C_nm * x_m[n]]
        # Note: (-1)^nm = 1 for even parity, -1 for odd parity (but absorbed in C_nm here)
        w_nm[:, m] = (np.sqrt(2.0) / Nf) * np.real(C_nm * x_m)

    # Step 5: Handle m = 0 (DC components) - Cornish Ref. [19] special case
    # Store in even time indices of column 0
    m = 0
    roll_amount = center_idx  # No frequency shift for DC
    X_shifted = np.roll(X, roll_amount)
    X_slice = X_shifted[start_idx:start_idx + Nt]
    X_reordered = np.concatenate([X_slice[half:], X_slice[:half]])

    # Zero negative frequencies for DC component
    X_dc = X_reordered.copy()
    X_dc[half:] = 0.0  # Zero negative frequencies
    X_dc[0] /= 2.0  # Divide DC bin by 2

    x_0 = fft.ifft(X_dc * Phi)
    # Store DC components in even time indices with √2 normalization
    even_indices = np.arange(0, Nt, 2)
    w_nm[even_indices, 0] = np.real(x_0[even_indices]) * np.sqrt(2.0) / Nf

    # Step 6: Handle m = Nf (Nyquist components) - store in odd indices of column 0
    m = Nf
    freq_shift = Nf * half
    roll_amount = center_idx - freq_shift
    X_shifted = np.roll(X, roll_amount)
    X_slice = X_shifted[start_idx:start_idx + Nt]
    X_reordered = np.concatenate([X_slice[half:], X_slice[:half]])

    # Zero positive frequencies for Nyquist component
    X_nyquist = X_reordered.copy()
    X_nyquist[:half] = 0.0  # Zero positive frequencies
    X_nyquist[half] /= 2.0  # Divide Nyquist bin by 2

    x_Nf = fft.ifft(X_nyquist * Phi)
    # Store Nyquist components in odd time indices
    odd_indices = np.arange(1, Nt, 2)
    even_source_indices = np.arange(0, Nt, 2)  # Take even indices from x_Nf
    w_nm[odd_indices, 0] = np.real(x_Nf[even_source_indices]) * np.sqrt(2.0) / Nf

    return w_nm


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
