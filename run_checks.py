import matplotlib.pyplot as plt
import numpy as np
from wdm_roll import *

plt.style.use('seaborn-v0_8-whitegrid')

# --- Test Functions and Classes ---
def chirp_signal(ts, Ac, As, f, fdot):
    """Generates a chirp signal."""
    phases = 2 * np.pi * ts * (f + fdot * ts / 2.0)
    return Ac * np.cos(phases) + As * np.sin(phases)

A_wavelet_param = 0.25 # Global for tests, 
d_wavelet_param = 4    # Global for tests

def run_parsevals_theorem_and_chirp_track_test():
    """Runs Parseval's theorem and chirp tracking tests, printing results and plotting."""
    print("\n--- Running Parseval's Theorem and Chirp Track Test ---")
    dt = 1.0 / np.pi # Sampling interval of original signal
    fny = 1.0 / (2.0 * dt) # Nyquist frequency

    nt = 64 # WDM time bins
    nf = 64 # WDM frequency bins
    n_total = nt * nf # Total samples in original signal

    ts_signal = dt * np.arange(n_total) # Time vector for original signal
    T_duration = n_total * dt # Total duration of original signal

    f0 = fny / 5.0 # Initial frequency of chirp
    fdot = f0 / T_duration # Rate of change of frequency

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
    sum_f_sq = np.sum(f_time_domain**2)
    sum_f_tilde_sq = np.sum(f_tilde_wdm**2)
    
    parseval_check = np.isclose(sum_f_sq, sum_f_tilde_sq, rtol=1e-2, atol=0)
    if not parseval_check:
        print(f"Parseval's Theorem FAILED: sum(f^2)={sum_f_sq:.4e}, sum(f_tilde^2)={sum_f_tilde_sq:.4e} (rtol=1e-2)")
    else:
        print(f"Parseval's Theorem PASSED: sum(f^2)={sum_f_sq:.4e}, sum(f_tilde^2)={sum_f_tilde_sq:.4e} (rtol=1e-2)")

    # --- Chirp Track Test ---
    dT_wdm_bin, dF_wdm_bin = wdm_dT_dF(nt, nf, dt) # WDM bin widths based on original signal dt

    max_power_indices_freq = np.argmax(np.abs(f_tilde_wdm), axis=1)
    
    times_for_chirp_track_pred = np.arange(nt) * dT_wdm_bin # Time for each WDM time bin
    predicted_frequencies_chirp = f0 + fdot * times_for_chirp_track_pred
    predicted_max_power_indices_freq = predicted_frequencies_chirp / dF_wdm_bin
    
    diff_chirp_track = np.abs(max_power_indices_freq[1:-1] - predicted_max_power_indices_freq[1:-1])
    if diff_chirp_track.size > 0:
        chirp_track_check = np.all(diff_chirp_track <= 2.5)
        max_diff_val = np.max(diff_chirp_track)
        if not chirp_track_check:
            print(f"Chirp Track FAILED: Max deviation {max_diff_val:.2f} > 2.5. Failing diffs: {diff_chirp_track[diff_chirp_track > 2.5]}")
        else:
            print(f"Chirp Track PASSED: Max deviation {max_diff_val:.2f} <= 2.5")
    elif nt <=2 : 
        print("Chirp Track SKIPPED: nt is too small to evaluate edges.")
    else: 
        print("Chirp Track SKIPPED: Not enough data points after excluding edges.")

    # --- Plotting ---


    orig_kwgs = dict( color='tab:blue', alpha=0.5, label='original')
    recon_kwgs = dict(color='tab:orange', alpha=0.5, label='recon')

    # 1. Chirp in Frequency Domain (FFT of original signal)
    plt.figure(figsize=(12, 8))
    original_fft = fft(f_time_domain)
    original_fft_freqs = fftfreq(n_total, d=dt)
    reconstructed_fft = fft(f_reconstructed_time)
    # Plot only positive frequencies for clarity
    positive_freq_mask = original_fft_freqs >= 0
    plt.subplot(2, 2, 1)
    plt.plot(original_fft_freqs[positive_freq_mask], np.abs(original_fft[positive_freq_mask]), **orig_kwgs)
    plt.plot(original_fft_freqs[positive_freq_mask],np.abs(reconstructed_fft[positive_freq_mask]), **recon_kwgs )
    plt.legend()
    plt.title('Chirp in Frequency Domain (FFT of Original)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend(fontsize='small', loc='upper right', frameon=True)
    plt.xlim(0, fny) # Show up to Nyquist

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
    plt.plot(ts_signal, f_time_domain,  **orig_kwgs)
    plt.plot(ts_signal, f_reconstructed_time,  **recon_kwgs)
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
    plt.yscale('log') # Residuals can be small, log scale helps
    plt.xlim(0, fny)
    plt.ylim(bottom=max(1e-9, np.min(freq_residuals[positive_freq_mask & (freq_residuals > 0)])*0.1)) # Avoid zero for log scale

    plt.tight_layout()
    plt.show()


def run_single_element_inverse_reconstruction_test():
    """Runs round-trip transform for single element impulses, printing results."""
    print("\n--- Running Single Element Inverse Reconstruction Test ---")
    nt = 32 
    nf = 32

    reconstruction_failures = []
    all_passed = True

    for i in range(nt): 
        for j in range(1, nf): 
            x_original_wdm = np.zeros((nt, nf))
            x_original_wdm[i, j] = 1.0

            time_signal_from_single_coeff = wdm_inverse_direct(x_original_wdm, A_wavelet_param, d_wavelet_param)
            
            x_reconstructed_wdm = wdm_transform(time_signal_from_single_coeff, nt, nf, A_wavelet_param, d_wavelet_param)
            
            if not np.allclose(x_original_wdm, x_reconstructed_wdm, atol=1e-7): 
                diff_wdm = np.abs(x_original_wdm - x_reconstructed_wdm)
                diff_val = np.max(diff_wdm)
                reconstruction_failures.append(((i,j), diff_val))
                all_passed = False
                
                fig, ax = plt.subplots(1,3, figsize=(6,4))
                ax[0].imshow(np.abs(x_original_wdm.T), aspect='auto', origin='lower',  cmap='viridis')
                ax[1].imshow(np.abs(x_reconstructed_wdm.T), aspect='auto', origin='lower',  cmap='viridis')
                ax[2].imshow(diff_wdm.T, aspect='auto', origin='lower',  cmap='viridis')
                ax[0].set_title("Original WDM")
                ax[1].set_title("Orig->time->WDM")
                ax[2].set_title("diff")
                break
    
    if reconstruction_failures:
        print(f"Single element reconstruction FAILED for {len(reconstruction_failures)} cases:")
        for k_idx in range(min(5, len(reconstruction_failures))): 
            print(f"  Index (t,f)=({reconstruction_failures[k_idx][0][0]},{reconstruction_failures[k_idx][0][1]}), max_abs_diff={reconstruction_failures[k_idx][1]:.2e}")
    
    if all_passed:
        print("Single Element Inverse Reconstruction Test PASSED for all elements.")
    else:
        print("Single Element Inverse Reconstruction Test FAILED overall (see details above).")
    return all_passed




if __name__ == '__main__':
    run_parsevals_theorem_and_chirp_track_test()
    run_single_element_inverse_reconstruction_test()
    



