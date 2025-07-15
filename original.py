import matplotlib.pyplot as plt
import numpy as np
import scipy.special
from numba import njit
from numpy import fft
from numpy.typing import NDArray


def wdm_inverse_transform(wave_in: NDArray[np.float64], Nf: int, Nt: int, nx: float = 4.0) -> NDArray[np.float64]:
    rfft_data = inverse_wavelet_freq_helper_fast(wave_in, phitilde_vec_norm(Nf, Nt, nx), Nf, Nt)
    return np.fft.irfft(rfft_data)


def wdm_transform(data: NDArray[np.float64], Nf: int, Nt: int, nx: float = 4.0) -> NDArray[np.float64]:
    return transform_wavelet_freq_helper(np.fft.rfft(data), Nf, Nt, 2 / Nf * phitilde_vec_norm(Nf, Nt, nx))


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


@njit()
def DX_assign_loop(
        m: int,
        Nt: int,
        Nf: int,
        DX: NDArray[np.complex128],
        data: NDArray[np.complex128],
        phif: NDArray[np.float64],
) -> None:
    """Helper for assigning DX in the main loop"""
    assert len(DX.shape) == 1, 'Storage array must be 1D'
    assert len(data.shape) == 1, 'Data must be 1D'
    assert len(phif.shape) == 1, 'Phi array must be 1D'

    i_base: int = int(Nt // 2)
    jj_base: int = int(m * Nt // 2)

    if m in (0, Nf):
        # NOTE this term appears to be needed to recover correct constant (at least for m=0) but was previously missing
        DX[Nt // 2] = phif[0] * data[int(m * Nt // 2)] / 2.0
    else:
        DX[Nt // 2] = phif[0] * data[int(m * Nt // 2)]

    for jj in range(jj_base + 1 - int(Nt // 2), jj_base + int(Nt // 2)):
        j: int = int(np.abs(jj - jj_base))
        i: int = i_base - jj_base + jj
        if (m == Nf and jj > jj_base) or (m == 0 and jj < jj_base):
            DX[i] = 0.0
        elif j == 0:
            continue
        else:
            DX[i] = phif[j] * data[jj]


@njit()
def DX_unpack_loop(m: int, Nt: int, Nf: int, DX_trans: NDArray[np.complex128], wave: NDArray[np.float64]) -> None:
    """Helper for unpacking fftd DX in main loop"""
    assert len(DX_trans.shape) == 1, 'Data array must be 1D'
    assert len(wave.shape) == 2, 'Output array must be 2D'
    if m == 0:
        # half of lowest and highest frequency bin pixels are redundant
        # so store them in even and odd components of m=0 respectively
        for n in range(0, Nt, 2):
            wave[n, 0] = DX_trans[n].real * np.sqrt(2.0)
    elif m == Nf:
        for n in range(0, Nt, 2):
            wave[n + 1, 0] = DX_trans[n].real * np.sqrt(2.0)
    else:
        for n in range(Nt):
            if m % 2:
                if (n + m) % 2:
                    wave[n, m] = -DX_trans[n].imag
                else:
                    wave[n, m] = DX_trans[n].real
            elif (n + m) % 2:
                wave[n, m] = DX_trans[n].imag
            else:
                wave[n, m] = DX_trans[n].real


def transform_wavelet_freq_helper(
        data: NDArray[np.complex128],
        Nf: int,
        Nt: int,
        phif: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Helper to do the wavelet transform using the fast wavelet domain transform"""
    assert len(data.shape) == 1, 'Only support 1D Arrays currently'
    assert len(phif.shape) == 1, 'phif must be 1D'
    wave = np.zeros((Nt, Nf))  # wavelet wavepacket transform of the signal

    DX = np.zeros(Nt, dtype=np.complex128)
    for m in range(Nf + 1):
        DX_assign_loop(m, Nt, Nf, DX, data, phif)
        DX_trans = fft.ifft(DX, Nt)
        DX_unpack_loop(m, Nt, Nf, DX_trans, wave)
    return wave


def check_roundtrip():
    f0, dt = 1, 0.125  # Frequency and time step
    Nf = Nt = 8
    nx = 4.0
    t = np.arange(0, Nt * Nf) * dt
    data = np.sin(2 * np.pi * f0 * t)
    wave = wdm_transform(data, Nf, Nt, nx)
    data_reconstructed = wdm_inverse_transform(wave, Nf, Nt, nx)

    # plotting
    T_bins = np.arange(Nt) * dt  # Time axis for plots
    F_bins = np.arange(Nf) / (2 * Nf * dt)  # Frequency axis for plots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(t, data, label='Original Signal', color='blue')
    axes[0].plot(t, data_reconstructed, label='Reconstructed Signal', color='orange', linestyle='--')
    pmc= axes[1].pcolormesh(T_bins, F_bins, wave.T, shading='auto', lw=0.5, edgecolor='white')
    fig.colorbar(pmc, ax=axes[1], label='Wavelet Coefficients', orientation='vertical')
    plt.tight_layout()
    plt.show()

    # Check if the original and reconstructed data match
    assert np.allclose(
        data, data_reconstructed,
        atol=1e-6), "Roundtrip failed: Original and reconstructed data do not match."


if __name__ == "__main__":
    check_roundtrip()
