import matplotlib.pyplot as plt
import numpy as np
import scipy.special
from numba import njit
from numpy import fft
from numpy.typing import NDArray


### ORIGNAL CODE ### (dont change this part)

def wdm_transform(data: NDArray[np.float64], Nf: int, Nt: int, nx: float = 4.0) -> NDArray[np.float64]:
    return transform_wavelet_freq_helper(np.fft.rfft(data), Nf, Nt, 2 / Nf * phitilde_vec_norm(Nf, Nt, nx))


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


#### END OF ORIGINAL CODE ###

## NEW VERSION ## (change this part)


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


def wdm_transform_roll(x, nt, nf, A, d):
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


###


def check_transform():
    f0, dt = 1, 0.125  # Frequency and time step
    Nf = Nt = 8
    t = np.arange(0, Nt * Nf) * dt
    data = np.sin(2 * np.pi * f0 * t)
    wave = wdm_transform(data, Nf, Nt, nx=4.0)
    wave_roll = wdm_transform_roll(data, Nt, Nf, A=0.25, d=4.0)
    wave_diff = wave - wave_roll

    # plotting
    T_bins = np.arange(Nt) * dt  # Time axis for plots
    F_bins = np.arange(Nf) / (2 * Nf * dt)  # Frequency axis for plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    def _plot_ax(fig, ax, w, title):
        pmc = ax.pcolormesh(T_bins, F_bins, w.T, shading='auto', lw=0.5, edgecolor='white')
        fig.colorbar(pmc, ax=ax, label=title, orientation='vertical')

    _plot_ax(fig, axes[0], wave, 'Wavelet Coefficients')
    _plot_ax(fig, axes[1], wave_roll, 'Roll Wavelet Coefficients')
    _plot_ax(fig, axes[2], wave_diff, 'Difference (Original - Roll)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    check_transform()
