import matplotlib.pyplot as plt
import numpy as np


def plot_true_object(n_map, title, filepath):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    im1 = axs[0].imshow(np.abs(n_map), cmap="inferno", aspect="auto")
    axs[0].set_title(f"{title} (Modulus)")
    axs[0].set_box_aspect(1)
    plt.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(np.angle(n_map), cmap="inferno", aspect="auto")
    axs[1].set_title(f"{title} (Phase)")
    axs[1].set_box_aspect(1)
    plt.colorbar(im2, ax=axs[1])

    plt.savefig(filepath, bbox_inches="tight")
    plt.close()


def plot_exit_wave(u_field, filepath):
    plt.subplots(1, 2, figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.angle(u_field.T), cmap="inferno", aspect="auto")
    plt.title("Exit Wave Phase")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(u_field.T), cmap="inferno", aspect="auto")
    plt.title("Exit Wave Amplitude")
    plt.colorbar()
    plt.savefig(filepath)
    plt.close()


def plot_data_intensity(exit_wave_1d, measured_amp_1d, filepath):
    plt.figure(figsize=(10, 4))
    plt.plot(np.abs(exit_wave_1d), label="Magnitude (Near Field)")
    plt.plot(np.fft.fftshift(measured_amp_1d) / 100, label="Diffraction Amp (Scaled)")
    plt.title("Exit Wave Data (Central Probe)")
    plt.legend()
    plt.savefig(filepath)
    plt.close()


def plot_epoch_probe(probe_true, probe_est, epoch, filepath):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(np.abs(probe_true), "k--", label="Ground Truth", alpha=0.6)
    ax[0].plot(np.abs(probe_est), "r-", label=f"Epoch {epoch}")
    ax[0].set_title("Probe Amplitude")
    ax[0].legend()

    ax[1].plot(np.angle(probe_true), "k--", label="Ground Truth", alpha=0.6)
    ax[1].plot(np.angle(probe_est), "r-", label=f"Epoch {epoch}")
    ax[1].set_title("Probe Phase")
    plt.suptitle(f"Probe Reconstruction - Epoch {epoch}")
    plt.savefig(filepath)
    plt.close()


def plot_epoch_object(n_viz, epoch, filepath):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    im1 = axs[0].imshow(np.abs(n_viz.T), cmap="inferno", aspect="auto")
    axs[0].set_title(f"Modulus(n) - Epoch {epoch}")
    axs[0].set_box_aspect(1)
    plt.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(np.angle(n_viz.T), cmap="inferno", aspect="auto")
    axs[1].set_title(f"Phase(n) - Epoch {epoch}")
    axs[1].set_box_aspect(1)
    plt.colorbar(im2, ax=axs[1])

    plt.savefig(filepath, bbox_inches="tight")
    plt.close()


def plot_final_probe(probe_true, probe_history, mse, filepath_comp, filepath_conv):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(np.abs(probe_true), "k", linewidth=2, label="Ground Truth")
    axs[0].plot(np.abs(probe_history[0]), "g--", linewidth=1.5, label="Initial Guess")
    axs[0].plot(
        np.abs(probe_history[-1]), "r-.", linewidth=1.5, label="Final Reconstruction"
    )
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title(f"Probe Reconstruction Results (MSE: {mse:.2e})")
    axs[0].legend()

    axs[1].plot(np.angle(probe_true), "k", linewidth=2, label="Ground Truth")
    axs[1].plot(np.angle(probe_history[0]), "g--", linewidth=1.5, label="Initial Guess")
    axs[1].plot(
        np.angle(probe_history[-1]), "r-.", linewidth=1.5, label="Final Reconstruction"
    )
    axs[1].set_ylabel("Phase (rad)")
    axs[1].set_xlabel("Pixel Index")
    plt.tight_layout()
    plt.savefig(filepath_comp)
    plt.close()

    err_history = [np.mean(np.abs(p - probe_true) ** 2) for p in probe_history]
    plt.figure()
    plt.plot(err_history)
    plt.xlabel("Epoch")
    plt.ylabel("Probe Error (MSE vs GT)")
    plt.title("Probe Convergence")
    plt.savefig(filepath_conv)
    plt.close()


def plot_final_object(n_true, n_est, filepath):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].imshow(np.angle(n_true.T), cmap="inferno", aspect="auto")
    axs[0, 0].set_title("Ground Truth (Phase - Coarse)")

    axs[0, 1].imshow(np.angle(n_est.T), cmap="inferno", aspect="auto")
    axs[0, 1].set_title("Reconstruction (Phase)")

    axs[1, 0].imshow(np.abs(n_true.T), cmap="inferno", aspect="auto")
    axs[1, 0].set_title("Ground Truth (Modulus - Coarse)")

    axs[1, 1].imshow(np.abs(n_est.T), cmap="inferno", aspect="auto")
    axs[1, 1].set_title("Reconstruction (Modulus)")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
