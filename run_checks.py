import os

import matplotlib.pyplot as plt

from wdm_roll import *
from original import wdm_inverse_transform as wdm_inverse_transform_orig
from original import wdm_transform as wdm_transform_orig

plt.style.use('seaborn-v0_8-whitegrid')

HERE = __file__
OUTDIR = os.path.join(os.path.dirname(HERE), 'out')
os.makedirs(OUTDIR, exist_ok=True)

A_wavelet_param = 0.25  # Global for tests,
d_wavelet_param = 4  # Global for tests


# --- Test Functions and Classes ---
def chirp_signal(ts, Ac, As, f, fdot):
    """Generates a chirp signal."""
    phases = 2 * np.pi * ts * (f + fdot * ts / 2.0)
    return Ac * np.cos(phases) + As * np.sin(phases)


def run_monochromatic_wnm_test():
    """Runs a test for monochromatic WNM generation, printing results and plotting."""
    print("\n--- Running Monochromatic WNM Test ---")

    f0, dt = 1, 0.125  # Frequency and time step
    Nf = Nt = 8
    nx = 4.0
    A = 1
    N = Nt * Nf
    t = np.arange(0, N) * dt  # Time vector for original signal
    signal_time = A * np.sin(2 * np.pi * f0 * t)  # Monochromatic signal

    # Generate analytical WNM for a monochromatic signal
    T_bins = np.arange(Nt) * dt  # Time axis for plots
    F_bins = np.arange(Nf) / (2 * Nf * dt)  # Frequency axis for plots

    orig_wnm = wdm_transform_orig(signal_time, Nt, Nf, nx=4.0)

    # Perform WDM transform
    wnm = wdm_transform(signal_time, Nt, Nf, A_wavelet_param, d_wavelet_param)
    reconstructed_time = wdm_inverse_transform(orig_wnm, A_wavelet_param, d_wavelet_param)

    # Check if the WNM matches the analytical result
    diff_wnm = np.abs(wnm - orig_wnm)
    max_diff = np.max(diff_wnm)
    if max_diff > 1e-6:
        print(f"Monochromatic WNM Test FAILED: Max difference {max_diff:.2e} exceeds tolerance.")
    else:
        print(f"Monochromatic WNM Test PASSED: Max difference {max_diff:.2e} within tolerance.")

    # round-trip reconstruction of analytical WNM
    diff_time = np.abs(signal_time - reconstructed_time)
    max_diff_time = np.max(diff_time)
    if max_diff_time > 1e-6:
        print(f"Monochromatic WNM Test FAILED: Max time-domain difference {max_diff_time:.2e} exceeds tolerance.")
    else:
        print(f"Monochromatic WNM Test PASSED: Max time-domain difference {max_diff_time:.2e} within tolerance.")

    # Plotting ([[time, WDM transform, analytical]])
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # axes[0]: Time series plot
    axes[0].plot(t, signal_time, label='Original Signal', color='blue')
    axes[0].plot(t, reconstructed_time, label='Reconstructed Signal', color='orange')
    axes[0].set_title("Original vs. Reconstructed Signal (Time Domain)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()

    # twin plot fot phase
    ax2 = axes[0].twinx()
    ax2.plot(t, np.angle(reconstructed_time), color='green', linestyle='--', label='Phase of Reconstructed Signal')
    ax2.plot(t, np.angle(signal_time), color='red', linestyle='--', label='Phase of Original Signal')

    # axes[1]: WDM Transform
    xy = (T_bins, F_bins)
    kwgs = dict(cmap='viridis', shading='auto', edgecolors='white', linewidth=0.5)
    pcm1 = axes[1].pcolormesh(*xy, np.abs(wnm.T), **kwgs)
    fig.colorbar(pcm1, ax=axes[1], orientation='horizontal', pad=0.15)
    axes[1].set_title("WDM Transform Output")

    # axes[2]: Analytical WNM
    pcm2 = axes[2].pcolormesh(*xy, np.abs(orig_wnm.T), **kwgs)
    fig.colorbar(pcm2, ax=axes[2], orientation='horizontal', pad=0.15)
    axes[2].set_title("Analytical WNM")

    # axes[3]: Difference between WNM and Analytical WNM
    diff_wnm = np.abs(wnm - orig_wnm)
    pcm3 = axes[3].pcolormesh(*xy, diff_wnm.T, **kwgs)
    fig.colorbar(pcm3, ax=axes[3], orientation='horizontal', pad=0.15)
    axes[3].set_title("Difference (WNM - Analytical WNM)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'monochromatic_wnm_test.png'), dpi=300)


def run_parsevals_theorem_and_chirp_track_test():
    """Runs Parseval's theorem and chirp tracking tests, printing results and plotting."""
    print("\n--- Running Parseval's Theorem and Chirp Track Test ---")
    dt = 1.0 / np.pi  # Sampling interval of original signal
    fny = 1.0 / (2.0 * dt)  # Nyquist frequency

    nt = 32  # WDM time bins
    nf = 32  # WDM frequency bins
    n_total = nt * nf  # Total samples in original signal

    ts_signal = dt * np.arange(n_total)  # Time vector for original signal
    T_duration = n_total * dt  # Total duration of original signal

    f0 = fny / 5.0  # Initial frequency of chirp
    fdot = f0 / T_duration  # Rate of change of frequency

    Amplitude = 1.0
    rng = np.random.default_rng(seed=42)
    phi_chirp = np.arctan2(rng.standard_normal(), rng.standard_normal())
    Ac = Amplitude * np.cos(phi_chirp)
    As = Amplitude * np.sin(phi_chirp)

    f_time_domain = chirp_signal(ts_signal, Ac, As, f0, fdot)

    # Perform WDM transform
    f_tilde_wdm = wdm_transform(f_time_domain, nt, nf, A_wavelet_param, d_wavelet_param)

    # Perform inverse WDM transform for reconstruction plots
    f_reconstructed_time = wdm_inverse_transform(f_tilde_wdm, A_wavelet_param, d_wavelet_param)

    # --- Parseval's Theorem Test ---
    sum_f_sq = np.sum(f_time_domain ** 2)
    sum_f_tilde_sq = np.sum(f_tilde_wdm ** 2)

    parseval_check = np.isclose(sum_f_sq, sum_f_tilde_sq, rtol=1e-2, atol=0)
    if not parseval_check:
        print(f"Parseval's Theorem FAILED: sum(f^2)={sum_f_sq:.4e}, sum(f_tilde^2)={sum_f_tilde_sq:.4e} (rtol=1e-2)")
    else:
        print(f"Parseval's Theorem PASSED: sum(f^2)={sum_f_sq:.4e}, sum(f_tilde^2)={sum_f_tilde_sq:.4e} (rtol=1e-2)")

    # --- Chirp Track Test ---
    dT_wdm_bin, dF_wdm_bin = wdm_dT_dF(nt, nf, dt)  # WDM bin widths based on original signal dt

    max_power_indices_freq = np.argmax(np.abs(f_tilde_wdm), axis=1)

    times_for_chirp_track_pred = np.arange(nt) * dT_wdm_bin  # Time for each WDM time bin
    predicted_frequencies_chirp = f0 + fdot * times_for_chirp_track_pred
    predicted_max_power_indices_freq = predicted_frequencies_chirp / dF_wdm_bin

    diff_chirp_track = np.abs(max_power_indices_freq[1:-1] - predicted_max_power_indices_freq[1:-1])
    if diff_chirp_track.size > 0:
        chirp_track_check = np.all(diff_chirp_track <= 2.5)
        max_diff_val = np.max(diff_chirp_track)
        if not chirp_track_check:
            print(
                f"Chirp Track FAILED: Max deviation {max_diff_val:.2f} > 2.5. Failing diffs: {diff_chirp_track[diff_chirp_track > 2.5]}")
        else:
            print(f"Chirp Track PASSED: Max deviation {max_diff_val:.2f} <= 2.5")
    elif nt <= 2:
        print("Chirp Track SKIPPED: nt is too small to evaluate edges.")
    else:
        print("Chirp Track SKIPPED: Not enough data points after excluding edges.")

    # --- Plotting ---

    orig_kwgs = dict(color='tab:blue', alpha=0.5, label='original')
    recon_kwgs = dict(color='tab:orange', alpha=0.5, label='recon')

    # 1. Chirp in Frequency Domain (FFT of original signal)
    plt.figure(figsize=(12, 8))
    original_fft = fft.fft(f_time_domain)
    original_fft_freqs = fft.fftfreq(n_total, d=dt)
    reconstructed_fft = fft.fft(f_reconstructed_time)
    # Plot only positive frequencies for clarity
    positive_freq_mask = original_fft_freqs >= 0
    plt.subplot(2, 2, 1)

    ratio = np.abs(original_fft[positive_freq_mask]) / np.abs(reconstructed_fft[positive_freq_mask])

    plt.plot(original_fft_freqs[positive_freq_mask], np.abs(original_fft[positive_freq_mask]), **orig_kwgs)
    plt.plot(original_fft_freqs[positive_freq_mask], np.abs(reconstructed_fft[positive_freq_mask]), **recon_kwgs)
    plt.legend(fontsize='small', loc='upper right', frameon=True)
    # twinx for ratio
    ax2 = plt.gca().twinx()
    ax2.plot(original_fft_freqs[positive_freq_mask], ratio, color='tab:green', linestyle='--',
             label='Ratio (Original/Reconstructed)')
    ax2.legend(fontsize='small', loc='upper left', frameon=True)

    plt.title('Chirp in Frequency Domain (FFT of Original)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, fny)  # Show up to Nyquist

    # 2. WDM Transform (imshow)
    # Get actual time and frequency extents for WDM plot
    # wdm_times_xaxis are the start times of each WDM time bin
    # wdm_freqs_yaxis are the start frequencies of each WDM frequency bin
    wdm_times_xaxis, wdm_freqs_yaxis = wdm_times_frequencies(nt, nf, dt)

    plt.subplot(2, 2, 2)
    # extent: [left, right, bottom, top]
    # We want to show the full range covered by the bins
    # Time axis: from wdm_times_xaxis[0] to wdm_times_xaxis[-1] + dT_wdm_bin
    # Freq axis: from wdm_freqs_yaxis[0] to wdm_freqs_yaxis[-1] + dF_wdm_bin
    img_extent = [wdm_times_xaxis[0], wdm_times_xaxis[-1] + dT_wdm_bin,
                  wdm_freqs_yaxis[0], wdm_freqs_yaxis[-1] + dF_wdm_bin]

    # Transpose f_tilde_wdm because imshow's first index is rows (y-axis, frequency), second is columns (x-axis, time)
    # And WDM matrix is (nt_bins, nf_bins) = (time_bins, freq_bins)
    # So f_tilde_wdm is (time, freq). For imshow(M), M[row,col].
    # We want time on x-axis, freq on y-axis. So imshow(f_tilde_wdm.T)
    plt.imshow(np.abs(f_tilde_wdm.T), aspect='auto', origin='lower',
               extent=img_extent, cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.title('WDM Transform Output')
    plt.xlabel(f'Time (s) - WDM Bins (total {nt} bins)')
    plt.ylabel(f'Frequency (Hz) - WDM Bins (total {nf} bins)')
    # Plot the predicted chirp track on top
    plt.plot(times_for_chirp_track_pred, predicted_frequencies_chirp, 'r--', linewidth=1, label='Predicted Chirp Track')
    plt.legend(fontsize='small', loc='upper right', frameon=True)

    # 3. Reconstructed Chirp (Time Domain)
    plt.subplot(2, 2, 3)
    plt.plot(ts_signal, f_time_domain, **orig_kwgs)
    plt.plot(ts_signal, f_reconstructed_time, **recon_kwgs)
    plt.title('Original vs. Reconstructed Chirp (Time Domain)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.xlim(ts_signal[0], ts_signal[-1])
    plt.legend(fontsize='small', loc='upper right', frameon=True)

    # 4. Frequency-Domain Residuals
    freq_residuals = np.abs(original_fft - reconstructed_fft)
    plt.subplot(2, 2, 4)
    plt.plot(original_fft_freqs[positive_freq_mask], freq_residuals[positive_freq_mask])
    plt.title('Frequency-Domain Residuals (FFT(Original) - FFT(Reconstructed))')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude of Difference')
    plt.yscale('log')  # Residuals can be small, log scale helps
    plt.xlim(0, fny)
    # plt.ylim(bottom=max(1e-9, np.min(
    #     freq_residuals[positive_freq_mask & (freq_residuals > 0)]) * 0.1))  # Avoid zero for log scale

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'parseval_chirp_track_test.png'), dpi=300)


def run_single_element_inverse_reconstruction_test():
    """Runs round-trip transform for single element impulses, printing results."""
    print("\n--- Running Single Element Inverse Reconstruction Test ---")
    nt = 4
    nf = 4

    wdm_times_xaxis = np.arange(nt)  # Time bins for WDM
    wdm_freqs_yaxis = np.arange(nf)  # Frequency bins for WDM

    reconstruction_failures = []
    all_passed = True

    for i in range(nt):
        for j in range(nf):
            x_original_wdm = np.zeros((nt, nf))
            x_original_wdm[i, j] = 1.0

            time_signal_from_single_coeff = wdm_inverse_transform(x_original_wdm, A_wavelet_param, d_wavelet_param)
            x_reconstructed_wdm = wdm_transform(time_signal_from_single_coeff, nt, nf, A_wavelet_param, d_wavelet_param)

            # time_signal_from_single_coeff = np.fft.irfft(wdm_inverse_transform_orig(x_original_wdm, nf, nt, nx=4))
            # x_reconstructed_wdm = wdm_transform_orig(np.fft.rfft(time_signal_from_single_coeff), nf, nt, nx=4)


            if not np.allclose(x_original_wdm, x_reconstructed_wdm, atol=1e-7):
                diff_wdm = np.abs(x_original_wdm - x_reconstructed_wdm)
                diff_val = np.max(diff_wdm)
                reconstruction_failures.append(((i, j), diff_val))
                all_passed = False

                plt.close('all')  # Close any previous plots to avoid clutter
                fig, ax = plt.subplots(1, 3, figsize=(6, 4))

                # Use pcolormesh for exact grid plotting and add colorbars on top
                xy = (wdm_times_xaxis, wdm_freqs_yaxis)
                kwgs = dict(cmap='viridis', shading='auto', edgecolors='white', linewidth=0.5, )
                pcm0 = ax[0].pcolormesh(*xy, np.abs(x_original_wdm.T), **kwgs)
                pcm1 = ax[1].pcolormesh(*xy, np.abs(x_reconstructed_wdm.T), **kwgs)
                pcm2 = ax[2].pcolormesh(*xy, diff_wdm.T, **kwgs)
                fig.colorbar(pcm0, ax=ax[0], orientation='horizontal', pad=0.15)
                fig.colorbar(pcm1, ax=ax[1], orientation='horizontal', pad=0.15)
                fig.colorbar(pcm2, ax=ax[2], orientation='horizontal', pad=0.15)

                ax[0].set_title("Original WDM")
                ax[1].set_title("Orig->time->WDM")
                ax[2].set_title("diff")
                fig.savefig(os.path.join(OUTDIR, f'single_element_reconstruction_fail_{i}_{j}.png'), dpi=300)

    if reconstruction_failures:
        print(f"Single element reconstruction FAILED for {len(reconstruction_failures)} cases:")
        for k_idx in range(min(5, len(reconstruction_failures))):
            print(
                f"  Index (t,f)=({reconstruction_failures[k_idx][0][0]},{reconstruction_failures[k_idx][0][1]}), max_abs_diff={reconstruction_failures[k_idx][1]:.2e}")

    if all_passed:
        print("Single Element Inverse Reconstruction Test PASSED for all elements.")
    else:
        print("Single Element Inverse Reconstruction Test FAILED overall (see details above).")
    return all_passed


if __name__ == '__main__':
    run_monochromatic_wnm_test()
    run_parsevals_theorem_and_chirp_track_test()
    run_single_element_inverse_reconstruction_test()
