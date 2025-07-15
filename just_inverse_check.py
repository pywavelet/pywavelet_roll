import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.special
from numba import njit
from numpy import fft
from numpy.typing import NDArray

OUTDIR = 'out'
os.makedirs(OUTDIR, exist_ok=True)


### ORIGINAL CODE ### (dont change)
def wdm_inverse_transform(wave_in: NDArray[np.float64], Nf: int, Nt: int, nx: float = 4.0) -> NDArray[np.float64]:
    rfft_data = inverse_wavelet_freq_helper_fast(wave_in, phitilde_vec_norm(Nf, Nt, nx), Nf, Nt)
    return np.fft.irfft(rfft_data)


@njit()
def unpack_wave_inverse(
        m: int,
        Nt: int,
        Nf: int,
        phif: NDArray[np.float64],
        fft_prefactor2s: NDArray[np.complex128],
        res: NDArray[np.complex128],
) -> None:
    """Helper for unpacking results of frequency domain inverse transform"""
    if m in (0, Nf):
        for i_ind in range(int(Nt // 2)):
            i = int(np.abs(m * int(Nt // 2) - i_ind))  # i_off+i_min2
            ind3 = (2 * i) % Nt
            res[i] += fft_prefactor2s[ind3] * phif[i_ind]
        if m == Nf:
            i_ind = int(Nt // 2)
            i = int(np.abs(m * int(Nt // 2) - i_ind))  # i_off+i_min2
            ind3 = 0
            res[i] += fft_prefactor2s[ind3] * phif[i_ind]
    else:
        ind31 = (int(Nt // 2) * m) % Nt
        ind32 = (int(Nt // 2) * m) % Nt
        for i_ind in range(int(Nt // 2)):
            i1 = int(Nt // 2) * m - i_ind
            i2 = int(Nt // 2) * m + i_ind
            res[i1] += fft_prefactor2s[ind31] * phif[i_ind]
            res[i2] += fft_prefactor2s[ind32] * phif[i_ind]
            ind31 -= 1
            ind32 += 1
            if ind31 < 0:
                ind31 = Nt - 1
            if ind32 == Nt:
                ind32 = 0

        res[Nt // 2 * m] = fft_prefactor2s[(Nt // 2 * m) % Nt] * phif[0]


@njit()
def pack_wave_inverse(
        m: int,
        Nt: int,
        Nf: int,
        prefactor2s: NDArray[np.complex128],
        wave_in: NDArray[np.float64],
) -> None:
    """Helper for fast frequency domain inverse transform to prepare for fourier transform"""
    if m == 0:
        for n in range(Nt):
            prefactor2s[n] = 1 / np.sqrt(2) * wave_in[(2 * n) % Nt, 0]
    elif m == Nf:
        for n in range(Nt):
            prefactor2s[n] = 1 / np.sqrt(2) * wave_in[(2 * n) % Nt + 1, 0]
    else:
        for n in range(Nt):
            val = float(wave_in[n, m])
            if (n + m) % 2:
                mult2 = -1j
            else:
                mult2 = 1

            prefactor2s[n] = mult2 * val


# @njit()
def inverse_wavelet_freq_helper_fast(
        wave_in: NDArray[np.float64],
        phif: NDArray[np.float64],
        Nf: int,
        Nt: int,
) -> NDArray[np.complex128]:
    """Jit compatible loop for wdm_inverse_transform"""
    ND = Nf * Nt

    prefactor2s = np.zeros(Nt, np.complex128)
    res = np.zeros(ND // 2 + 1, dtype=np.complex128)

    for m in range(Nf + 1):
        pack_wave_inverse(m, Nt, Nf, prefactor2s, wave_in)
        # with numba.objmode(fft_prefactor2s="complex128[:]"):
        fft_prefactor2s = fft.fft(prefactor2s)
        unpack_wave_inverse(m, Nt, Nf, phif, fft_prefactor2s, res)

    return res


def phitilde_vec_norm(Nf: int, Nt: int, nx: float) -> NDArray[np.float64]:
    """Normalize phitilde as needed for inverse frequency domain transform"""
    ND: int = Nf * Nt
    om: NDArray[np.float64] = np.asarray(2 * np.pi / ND * np.arange(0, Nt // 2 + 1), dtype=np.float64)

    OM: float = np.pi  # Nyquist angular frequency
    DOM: float = float(OM / Nf)  # 2 pi times DF
    insDOM: float = float(1.0 / np.sqrt(DOM))
    B = OM / (2 * Nf)
    A = (DOM - B) / 2
    phif = np.zeros(om.size, dtype=np.float64)

    mask = (np.abs(om) >= A) & (np.abs(om) < A + B)

    x = (np.abs(om[mask]) - A) / B
    y = scipy.special.betainc(nx, nx, x)
    phif[mask] = insDOM * np.cos(np.pi / 2.0 * y)

    phif[np.abs(om) < A] = insDOM

    # nrm should be 1
    nrm: float = float(
        np.sqrt((2 * np.sum(phif[1:] ** 2) + phif[0] ** 2) * 2 * np.pi / ND) / (np.pi ** (3 / 2) / np.pi),
    )
    return phif / nrm


### END OF ORIGINAL CODE ###

### NEW CODE ###


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
        p = scipy.special.betainc(d, d, z)
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


def wdm_inverse_transform_new(W, A, d):
    nt, nf = W.shape
    n_total = nt * nf
    half = nt // 2

    # Build phi window
    fs_full = fft.fftfreq(n_total)
    fs_phi = np.concatenate([fs_full[:half], fs_full[-half:]])
    _, dF = wdm_dT_dF(nt, nf, 1.0)
    phi = Phi_unit(fs_phi / dF, A, d) / np.sqrt(dF)

    # Step 1: Apply parity factors and normalize
    ylm = np.zeros((nt, nf), dtype=complex)
    for n in range(nt):
        for m in range(1, nf):  # Julia starts from m=2, Python from m=1
            C = 1 if (n + m) % 2 == 0 else 1j
            ylm[n, m] = C * W[n, m] / np.sqrt(2.0)

    # Step 2: FFT over time dimension (axis=0)
    ylm_fft = fft.fft(ylm, axis=0)

    # Step 3: Build frequency domain array
    X = np.zeros(n_total, dtype=complex)

    for m in range(1, nf):
        l0 = m * half

        # First contribution: G[l - m*Nt/2] * Ylm
        # This maps ylm_fft[:, m] to frequencies around l0
        temp1 = np.zeros(n_total, dtype=complex)
        temp1[:half] = ylm_fft[:half, m] * phi[:half]  # Positive frequencies
        temp1[half:2 * half] = ylm_fft[half:, m] * phi[half:]  # Negative frequencies
        X += np.roll(temp1, l0 - half)

        # Second contribution: G[l + m*Nt/2] * conj(Y(-l)m)
        l1 = n_total - l0

        temp2 = np.zeros(n_total, dtype=complex)
        # Zero frequency of conjugate
        temp2[0] = np.conj(ylm_fft[0, m]) * phi[0]

        # Positive frequencies of conjugate (reversed)
        if half > 1:
            temp2[1:half] = np.conj(ylm_fft[nt - 1:half:-1, m]) * phi[1:half]

        # Negative frequencies of conjugate (reversed)
        temp2[half:2 * half] = np.conj(ylm_fft[half:0:-1, m]) * phi[half:]

        X += np.roll(temp2, l1 - half)

    # Step 4: Inverse FFT to get time domain
    x_reconstructed = np.real(fft.ifft(X))

    return x_reconstructed


### end of NEW CODE ###

def run_single_element_inverse_reconstruction_test():
    nf = nt = 6
    t = np.arange(nf * nt)
    x_original_wdm = np.zeros((nt, nf))
    x_original_wdm[3, 3] = 1.0

    data = wdm_inverse_transform(x_original_wdm, nf, nt, nx=4.0)
    data_new = wdm_inverse_transform_new(x_original_wdm, 0.25, 4.0)
    print("Reconstructed data:", data)
    print("Reconstructed data (new):", data_new)
    print("data / data(new):", data/data_new)
    if not np.allclose(data, data_new, atol=1e-7):
        print("Reconstruction failed for the original test case.")
        fig = plt.figure()
        plt.plot(t, data, label='Reconstructed Data', color='orange')
        plt.plot(t, data_new, label='Reconstructed Data (new)', color='blue', linestyle='--')
        plt.legend()
        plt.title('Reconstruction Failure for Original Test Case')
        plt.savefig(os.path.join(OUTDIR, 'reconstruction_failure_original.png'), dpi=300)
        plt.close(fig)
    else:
        print("Original test case passed.")


if __name__ == "__main__":
    run_single_element_inverse_reconstruction_test()
