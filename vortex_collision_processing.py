import contrast
from PIL import Image
from scipy import ndimage
from skimage import restoration
import numpy as np
import matplotlib.pyplot as plt
import pyfftw
import multiprocessing
from cycler import cycler
import tqdm
import time
import cv2
import os
import regex as re
import velocity
import matplotlib
import pickle
import faulthandler

faulthandler.enable()

matplotlib.use("Qt5Agg")

pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"
# try to load previous fftw wisdom
try:
    with open("fft.wisdom", "rb") as file:
        wisdom = pickle.load(file)
        pyfftw.import_wisdom(wisdom)
except FileNotFoundError:
    print("No FFT wisdom found, starting over ...")
# for dark theme
plt.style.use("dark_background")
plt.rcParams["figure.facecolor"] = "#00000080"
plt.rcParams["axes.facecolor"] = "#00000080"
plt.rcParams["savefig.facecolor"] = "#00000080"
# plt.rcParams['savefig.transparent'] = True
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Liberation Sans']
# for plots
tab_colors = [
    "tab:blue",
    "tab:orange",
    "forestgreen",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "teal",
]
fills = [
    "lightsteelblue",
    "navajowhite",
    "darkseagreen",
    "lightcoral",
    "violet",
    "indianred",
    "lavenderblush",
    "lightgray",
    "darkkhaki",
    "darkturquoise",
]
edges = tab_colors
custom_cycler = (
    (cycler(color=tab_colors))
    + (cycler(markeredgecolor=edges))
    + (cycler(markerfacecolor=fills))
)
plt.rc("axes", prop_cycle=custom_cycler)
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.size": 12
# })
# sshfs taladjidi@patriot.lkb.upmc.fr:/partages/EQ15B/LEON-15B /home/tangui/LEON
# path_leon = "/run/user/1000/gvfs/sftp:host=patriot.lkb.upmc.fr,user=taladjidi/partages/EQ15B/LEON-15B"
# path_leon = "/run/user/1000/gvfs/sftp:host=88.160.142.14,port=16384,user=aladjidi/home/aladjidi/Disk0/LEON"
path_leon = "/Volumes/partages/EQ15B/LEON-15B"
path_real = f"{path_leon}/DATA/Atoms/2024/Vortex_collision/Real"
path_dn = f"{path_leon}/DATA/Atoms/2024/Vortex_collision/Dn"
path_fourier = f"{path_leon}/DATA/Atoms/2024/Vortex_collision/Fourier"
path_anim = f"{path_leon}/DATA/Atoms/2024/Vortex_collision/Animations"
path_temp = f"{path_leon}/DATA/Atoms/2024/Vortex_collision/Temperature"
k0 = 2 * np.pi / 780e-9
L = 20e-2
d_real = 3.76e-6
d_def = 1.1e-6
Nx, Ny = 3008, 3008
Nx_def, Ny_def = 2048, 2048


def natural_sort(l):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(l, key=alphanum_key)


def compute_phase_time(scan: str, plot: bool = False):
    """Compute the phase.

    Computes the phase from the interferograms for a time scan
    dataset.

    Args:
        scan (str): Path to the dataset
        plot (bool, optional): Whether to plot the phase. Defaults to False.
    """
    # powers = np.load(f"{path_real}/{scan}/powers.npy")
    # N_times = powers.size
    input_power = np.load(f"{path_real}/{scan}/input_power.npy")
    N_times = input_power.size
    N_avg = np.load(f"{path_real}/{scan}/N_avg.npy")
    laser_powers = np.load(f"{path_real}/{scan}/laser_settings.npy")
    field = np.zeros((N_times, N_avg, Ny // 2, Nx // 2), dtype=np.complex64)
    field_ref = np.zeros(
        (N_times, N_avg, Ny // 2, Nx // 2), dtype=np.complex64
    )
    field_vortex = np.zeros(
        (N_times, N_avg, Ny // 2, Nx // 2), dtype=np.complex64
    )
    # ref_arm = np.array(Image.open(
    #     f"{path_real}/ref.TIF"), dtype=np.uint16).astype(np.float32)
    # ref_arm = ndimage.zoom(ref_arm, .5)
    taus = np.zeros(N_times)
    taus_err = np.zeros(N_times)
    # prepare plans
    a = pyfftw.empty_aligned((Ny, Nx), dtype=np.float32)
    c = pyfftw.empty_aligned((Ny // 2, Nx // 2), dtype=np.complex64)
    plan_fft = pyfftw.builders.rfft2(a)
    plan_ifft = pyfftw.builders.ifft2(c)
    with open("fft.wisdom", "wb") as file:
        wisdom = pyfftw.export_wisdom()
        pickle.dump(wisdom, file)
    plans = (plan_fft, plan_ifft)
    # del a, b
    pbar = tqdm.tqdm(total=N_times * N_avg, desc="Computing phase", position=1)
    for i in range(N_times):
        tau = np.zeros(N_avg)
        for j in range(N_avg):
            im_ref = np.array(
                Image.open(f"{path_dn}/{scan}/dn_ref_{i}_{j}.tiff"),
                dtype=np.uint16,
            )
            im = np.array(
                Image.open(f"{path_dn}/{scan}/dn_{i}_{j}.tiff"),
                dtype=np.uint16,
            )
            im_vortex = np.array(
                Image.open(f"{path_dn}/{scan}/dn_vortex_{i}_{j}.tiff"),
                dtype=np.uint16,
            )
            field_ref[i, j, :, :] = contrast.im_osc_fast_t(im_ref, plans=plans)
            field[i, j, :, :] = contrast.im_osc_fast_t(im, plans=plans)
            field_vortex[i, j, :, :] = contrast.im_osc_fast_t(
                im_vortex, plans=plans
            )
            phi_ref = contrast.angle_fast(field_ref[i, j, :, :])
            rho_ref = (
                field_ref[i, j, :, :].real * field_ref[i, j, :, :].real
                + field_ref[i, j, :, :].imag * field_ref[i, j, :, :].imag
            )
            threshold = 2e-2
            mask = rho_ref < threshold * np.max(rho_ref)
            phi_ref_masked = np.ma.array(phi_ref, mask=mask)
            phi_ref_unwrapped = restoration.unwrap_phase(
                phi_ref_masked, wrap_around=(True, True)
            )
            tau[j] = np.abs(
                np.nanmax(phi_ref_unwrapped) - np.nanmin(phi_ref_unwrapped)
            )
            pbar.update(1)
        taus[i] = np.mean(tau)
        taus_err[i] = np.std(tau)
    pbar.close()
    print("Saving data ...")
    t0 = time.perf_counter()
    np.save(f"{path_dn}/{scan}/field_ref.npy", field_ref)
    np.save(f"{path_dn}/{scan}/field.npy", field)
    np.save(f"{path_dn}/{scan}/field_vortex.npy", field_vortex)
    np.save(f"{path_dn}/{scan}/taus_reproc.npy", taus)
    np.save(f"{path_dn}/{scan}/taus_err_reproc.npy", taus_err)
    t = time.perf_counter() - t0
    sz = field_ref.nbytes + field.nbytes + field_vortex.nbytes + taus.nbytes
    rate = sz / t
    print(
        f"Saved {sz*1e-6:.0f} MB of data in {t:.2f} s / {rate*1e-6:.2f} MB/s"
    )
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(np.abs(field[0, -1, :, :]) ** 2)
    ax[1].imshow(np.angle(field[0, -1, :, :]), cmap="twilight_shifted")
    fig.savefig(f"{path_dn}/{scan}/field.svg", dpi=300)
    taus0 = np.load(f"{path_dn}/{scan}/taus.npy")
    taus0_err = np.load(f"{path_dn}/{scan}/taus_err.npy")
    fig1, ax1 = plt.subplots()
    ax1.errorbar(
        laser_powers,
        taus0,
        yerr=taus0_err,
        marker="o",
        label="During acquisition",
    )
    ax1.errorbar(
        laser_powers, taus, yerr=taus_err, marker="o", label="Reprocessed"
    )
    ax1.legend()
    ax1.set_xlabel("Power in a.u.")
    ax1.set_ylabel(r"$\delta\phi = \tau$ in rad")
    ax1.set_title("Phase shift vs power")
    fig1.savefig(f"{path_dn}/{scan}/taus.svg", dpi=300)
    if plot:
        plt.show()
    plt.close("all")


def compute_phase_full_time(scan: str, plot: bool = False):
    """Compute the phase.

    Computes the phase from the interferograms for a time scan
    dataset.

    Args:
        scan (str): Path to the dataset
        plot (bool, optional): Whether to plot the phase. Defaults to False.
    """
    # powers = np.load(f"{path_real}/{scan}/powers.npy")
    # N_times = powers.size
    input_power = np.load(f"{path_real}/{scan}/input_power.npy")
    N_times = input_power.size
    N_avg = np.load(f"{path_real}/{scan}/N_avg.npy")
    laser_powers = np.load(f"{path_real}/{scan}/laser_settings.npy")
    field = np.zeros((N_times, N_avg, Ny, Nx), dtype=np.complex64)
    field_ref = np.zeros((N_times, N_avg, Ny, Nx), dtype=np.complex64)
    field_vortex = np.zeros((N_times, N_avg, Ny, Nx), dtype=np.complex64)
    # ref_arm = np.array(Image.open(
    #     f"{path_real}/ref.TIF"), dtype=np.uint16).astype(np.float32)
    # ref_arm = ndimage.zoom(ref_arm, .5)
    taus = np.zeros(N_times)
    taus_err = np.zeros(N_times)
    # prepare plans
    a = pyfftw.empty_aligned((Ny, Nx), dtype=np.float32)
    c = pyfftw.empty_aligned((Ny, Nx), dtype=np.complex64)
    plan_fft = pyfftw.builders.rfft2(a)
    plan_ifft = pyfftw.builders.ifft2(c)
    with open("fft.wisdom", "wb") as file:
        wisdom = pyfftw.export_wisdom()
        pickle.dump(wisdom, file)
    plans = (plan_fft, plan_ifft)
    # del a, b
    pbar = tqdm.tqdm(total=N_times * N_avg, desc="Computing phase", position=1)
    for i in range(N_times):
        tau = np.zeros(N_avg)
        for j in range(N_avg):
            im_ref = np.array(
                Image.open(f"{path_dn}/{scan}/dn_ref_{i}_{j}.tiff"),
                dtype=np.uint16,
            )
            im = np.array(
                Image.open(f"{path_dn}/{scan}/dn_{i}_{j}.tiff"),
                dtype=np.uint16,
            )
            im_vortex = np.array(
                Image.open(f"{path_dn}/{scan}/dn_vortex_{i}_{j}.tiff"),
                dtype=np.uint16,
            )
            field_ref[i, j, :, :] = contrast.im_osc_fast(
                im_ref, plans=plans, radius=min(im_ref.shape[-2:]) // 8
            )
            field[i, j, :, :] = contrast.im_osc_fast(
                im, plans=plans, radius=min(im.shape[-2:]) // 8
            )
            field_vortex[i, j, :, :] = contrast.im_osc_fast(
                im_vortex, plans=plans, radius=min(im_vortex.shape[-2:]) // 8
            )
            phi_ref = np.angle(field_ref[i, j, :, :])
            rho_ref = (
                field_ref[i, j, :, :].real * field_ref[i, j, :, :].real
                + field_ref[i, j, :, :].imag * field_ref[i, j, :, :].imag
            )
            threshold = 2e-2
            mask = rho_ref < threshold * np.nanmax(rho_ref)
            phi_ref_masked = np.ma.array(phi_ref, mask=mask)
            phi_ref_unwrapped = restoration.unwrap_phase(
                phi_ref_masked, wrap_around=(True, True)
            )
            tau[j] = np.abs(
                np.nanmax(phi_ref_unwrapped) - np.nanmin(phi_ref_unwrapped)
            )
            pbar.update(1)
        taus[i] = np.mean(tau)
        taus_err[i] = np.std(tau)
    pbar.close()
    print("Saving data ...")
    t0 = time.perf_counter()
    np.save(f"{path_dn}/{scan}/field_ref_full.npy", field_ref)
    np.save(f"{path_dn}/{scan}/field_full.npy", field)
    np.save(f"{path_dn}/{scan}/field_vortex_full.npy", field_vortex)
    np.save(f"{path_dn}/{scan}/taus_reproc.npy", taus)
    np.save(f"{path_dn}/{scan}/taus_err_reproc.npy", taus_err)
    t = time.perf_counter() - t0
    sz = field_ref.nbytes + field.nbytes + field_vortex.nbytes + taus.nbytes
    rate = sz / t
    print(
        f"Saved {sz*1e-6:.0f} MB of data in {t:.2f} s / {rate*1e-6:.2f} MB/s"
    )
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(np.abs(field[0, -1, :, :]) ** 2)
    ax[1].imshow(np.angle(field[0, -1, :, :]), cmap="twilight_shifted")
    fig.savefig(f"{path_dn}/{scan}/field.svg", dpi=300)
    taus0 = np.load(f"{path_dn}/{scan}/taus.npy")
    taus0_err = np.load(f"{path_dn}/{scan}/taus_err.npy")
    fig1, ax1 = plt.subplots()
    ax1.errorbar(
        laser_powers,
        taus0,
        yerr=taus0_err,
        marker="o",
        label="During acquisition",
    )
    ax1.errorbar(
        laser_powers, taus, yerr=taus_err, marker="o", label="Reprocessed"
    )
    ax1.legend()
    ax1.set_xlabel("Power in a.u.")
    ax1.set_ylabel(r"$\delta\phi = \tau$ in rad")
    ax1.set_title("Phase shift vs power")
    fig1.savefig(f"{path_dn}/{scan}/taus.svg", dpi=300)
    if plot:
        plt.show()
    plt.close("all")


def compute_phase_dist(scan: str, plot: bool = False):
    """Computes the phase from the interferograms for a distance scan
    dataset

    Args:
        scan (str): Path to the dataset
        plot (bool, optional): Whether to plot the phase. Defaults to False.
    """
    # powers = np.load(f"{path_real}/{scan}/powers.npy")
    # N_times = powers.size
    input_power = np.load(f"{path_real}/{scan}/input_power.npy")
    N_points = input_power.size
    intervortex_dists = np.load(f"{path_real}/{scan}/intervortex_dists.npy")
    field = np.zeros((N_points, Ny // 2, Nx // 2), dtype=np.complex64)
    field_ref = np.zeros((N_points, Ny // 2, Nx // 2), dtype=np.complex64)
    field_vortex = np.zeros((N_points, Ny // 2, Nx // 2), dtype=np.complex64)
    taus = np.zeros(N_points)
    for i in tqdm.tqdm(range(N_points), desc="Computing phase", position=1):
        im_ref = np.array(
            Image.open(f"{path_dn}/{scan}/dn_ref_{i}.tiff"), dtype=np.uint16
        )
        im = np.array(
            Image.open(f"{path_dn}/{scan}/dn_{i}.tiff"), dtype=np.uint16
        )
        im_vortex = np.array(
            Image.open(f"{path_dn}/{scan}/dn_vortex.tiff"), dtype=np.uint16
        )
        field_ref[i, :, :] = contrast.im_osc_fast_t(im_ref)
        field[i, :, :] = contrast.im_osc_fast_t(im)
        field_vortex = contrast.im_osc_fast_t(im_vortex)
        phi_ref = contrast.angle_fast(field_ref[i, :, :])
        rho_ref = field_ref[i, :, :] * np.conj(field_ref[i, :, :])
        rho_ref = rho_ref.real
        threshold = 1e-2
        mask = rho_ref < threshold * np.max(rho_ref)
        phi_ref_masked = np.ma.array(phi_ref, mask=mask)
        phi_ref_unwrapped = restoration.unwrap_phase(
            phi_ref_masked, wrap_around=(True, True)
        )
        taus[i] = np.abs(
            np.nanmax(phi_ref_unwrapped) - np.nanmin(phi_ref_unwrapped)
        )

    print("Saving data ...")
    t0 = time.perf_counter()
    np.save(f"{path_dn}/{scan}/field_ref.npy", field_ref)
    np.save(f"{path_dn}/{scan}/field.npy", field)
    np.save(f"{path_dn}/{scan}/field_vortex.npy", field_vortex)
    np.save(f"{path_dn}/{scan}/taus.npy", taus)
    t = time.perf_counter() - t0
    sz = field_ref.nbytes + field.nbytes + field_vortex.nbytes + taus.nbytes
    rate = sz / t
    print(
        f"Saved {sz*1e-6:.0f} MB of data in {t:.2f} s / {rate*1e-6:.2f} MB/s"
    )
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(np.abs(field[-1, :, :]))
    ax[1].imshow(np.angle(field[-1, :, :]), cmap="twilight_shifted")
    fig.savefig(f"{path_dn}/{scan}/field.svg", dpi=300)
    fig1, ax1 = plt.subplots()
    ax1.plot(intervortex_dists, taus, marker="o")
    ax1.set_xlabel("Initial dist in px")
    ax1.set_ylabel(r"$\delta\phi = \tau$ in rad")
    ax1.set_title("Phase shift vs power")
    fig1.savefig(f"{path_dn}/{scan}/taus.svg", dpi=300)
    if plot:
        plt.show()
    plt.close("all")


def plot_fields(scan: str, plot: bool = False):
    """Plot the fields.

    Args:
        scan (str): Path to the dataset
        plot (bool, optional): Whether to plot the fields. Defaults to False.
    """
    print("Loading data ...")
    t0 = time.perf_counter()
    taus = np.load(f"{path_dn}/{scan}/taus.npy")
    field_ref = np.load(f"{path_dn}/{scan}/field_ref.npy")
    field = np.load(f"{path_dn}/{scan}/field.npy")
    t = time.perf_counter() - t0
    sz = field_ref.nbytes + field.nbytes
    field = field[:, -1, :, :]
    field_ref = field_ref[:, -1, :, :]
    rate = sz / t
    if field_ref.shape[-1] == Nx // 2:
        d_real = 3.76e-6 * 2
    print(
        f"Loaded {sz*1e-6:.0f} MB of data in {t:.2f} s / {rate*1e-6:.2f} MB/s"
    )
    delta_n = taus / (k0 * L)
    xis = 1 / (k0 * np.sqrt(delta_n))
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
    im0 = ax[0].imshow(
        np.ones(field.shape[1:]),
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="none",
    )
    im1 = ax[1].imshow(
        np.ones(field.shape[1:]),
        cmap="twilight_shifted",
        vmin=-np.pi,
        vmax=np.pi,
        interpolation="none",
    )
    for a in ax:
        a.set_xlabel(r"$x/\xi$")
        a.set_ylabel(r"$y/\xi$")
    ax[0].set_title("Density")
    ax[1].set_title("Normalized phase")
    # ax[0].locator_params(axis="both", nbins=5)
    # ax[1].locator_params(axis="both", nbins=5)
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
    fig.colorbar(im0, ax=ax[0], label=r"$\rho$ in a.u.", shrink=0.6)
    cbar = fig.colorbar(
        im1, ax=ax[1], label=r"$\phi/\phi_0$ in rad", shrink=0.6
    )
    cbar.set_ticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        labels=[
            r"$\pi$",
            r"$-\frac{\pi}{2}$",
            r"$0$",
            r"$\frac{\pi}{2}$",
            r"$\pi$",
        ],
    )
    fig.canvas.draw()
    mat = np.array(fig.canvas.renderer._renderer)
    mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
    # create folder if it does not exist
    if not (os.path.exists(f"{path_anim}/{scan}")):
        os.mkdir(f"{path_anim}/{scan}")
    video_writer = cv2.VideoWriter(
        f"{path_anim}/{scan}/field.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        3,
        (mat.shape[1], mat.shape[0]),
    )
    video_writer1 = cv2.VideoWriter(
        f"{path_anim}/{scan}/field_rescaled.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        3,
        (mat.shape[1], mat.shape[0]),
    )
    for i in tqdm.tqdm(
        range(field.shape[0]), desc="Plotting fields", position=1
    ):
        fig.suptitle(
            rf"$\tau$ = {taus[i]:.0f} / $\xi$ = {xis[i]*1e6:.0f} $\mu m$"
        )
        ext_real = [
            -field.shape[-1] * d_real / (2 * xis[i]),
            field.shape[-1] * d_real / (2 * xis[i]),
            -field.shape[-2] * d_real / (2 * xis[i]),
            field.shape[-2] * d_real / (2 * xis[i]),
        ]
        contrast.exp_angle_fast(field[i, :, :], field_ref[i, :, :])
        phi_flat = contrast.angle_fast(field[i, :, :])
        rho = (
            field[i, :, :].real * field[i, :, :].real
            + field[i, :, :].imag * field[i, :, :].imag
        )
        im0.set_data(rho)
        im0.set_clim(np.nanmin(rho), np.nanmax(rho))
        im1.set_data(phi_flat)
        im0.set_extent(ext_real)
        im1.set_extent(ext_real)
        for a in ax:
            a.set_xlim((ext_real[0], ext_real[1]))
            a.set_ylim((ext_real[2], ext_real[3]))
        fig.canvas.draw()
        if plot:
            plt.show()
        mat = np.array(fig.canvas.renderer._renderer)
        mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        video_writer.write(mat)
        fig.savefig(f"{path_dn}/{scan}/fields_{i}.svg", dpi=300)
        window = 100
        for a in ax:
            a.set_xlim((-window, window))
            a.set_ylim((-window, window))
        fig.canvas.draw()
        mat = np.array(fig.canvas.renderer._renderer)
        mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        video_writer1.write(mat)
        fig.savefig(f"{path_dn}/{scan}/fields_rescaled_{i}.svg", dpi=300)
        if plot:
            plt.show()
    print()
    plt.close("all")
    video_writer.release()
    video_writer1.release()


def plot_energy(scan: str, plot: bool = False):
    """Plot the energies.

    Args:
        scan (str): Path to the dataset
        plot (bool, optional): Whether to plot the fields. Defaults to False.
    """
    print("Loading data ...")
    t0 = time.perf_counter()
    taus = np.load(f"{path_dn}/{scan}/taus.npy")
    u_comp = np.load(f"{path_dn}/{scan}/u_comp.npy")
    u_inc = np.load(f"{path_dn}/{scan}/u_inc.npy")
    t = time.perf_counter() - t0
    sz = u_comp.nbytes + u_inc.nbytes
    rate = sz / t
    if u_comp.shape[-1] == Nx // 2:
        d_real = 3.76e-6 * 2
    print(
        f"Loaded {sz*1e-6:.0f} MB of data in {t:.2f} s / {rate*1e-6:.2f} MB/s"
    )
    delta_n = taus / (k0 * L)
    xis = 1 / (k0 * np.sqrt(delta_n))
    fig, ax = plt.subplots(1, 2, figsize=(16, 9), dpi=300)
    im0 = ax[0].imshow(
        np.ones(u_comp.shape[-2:]),
        cmap="nipy_spectral",
        vmin=0,
        vmax=1,
        interpolation="none",
    )
    im1 = ax[1].imshow(
        np.ones(u_inc.shape[-2:]),
        cmap="nipy_spectral",
        vmin=0,
        vmax=1,
        interpolation="none",
    )
    for a in ax:
        a.set_xlabel(r"$x/\xi$")
        a.set_ylabel(r"$y/\xi$")
    ax[0].set_title(r"$|u_{comp}|^2$")
    ax[1].set_title(r"$|u_{inc}|^2$")
    # ax[0].locator_params(axis="both", nbins=5)
    # ax[1].locator_params(axis="both", nbins=5)
    fig.subplots_adjust(
        left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.1
    )
    fig.colorbar(im0, ax=ax[0], label=r"$|u_{comp}|^2 in a.u$", shrink=0.6)
    fig.colorbar(im1, ax=ax[1], label=r"$|u_{inc}|^2 in a.u$", shrink=0.6)
    fig.canvas.draw()
    mat = np.array(fig.canvas.renderer._renderer)
    mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
    # create folder if it does not exist
    if not (os.path.exists(f"{path_anim}/{scan}")):
        os.mkdir(f"{path_anim}/{scan}")
    video_writer = cv2.VideoWriter(
        f"{path_anim}/{scan}/energy.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        3,
        (mat.shape[1], mat.shape[0]),
    )
    video_writer1 = cv2.VideoWriter(
        f"{path_anim}/{scan}/energy_rescaled.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        3,
        (mat.shape[1], mat.shape[0]),
    )
    for i in tqdm.tqdm(
        range(u_inc.shape[0]), desc="Plotting energies", position=1
    ):
        fig.suptitle(
            rf"$\tau$ = {taus[i]:.0f} / $\xi$ = {xis[i]*1e6:.0f} $\mu m$"
        )
        ext_real = [
            -u_comp.shape[-1] * d_real / (2 * xis[i]),
            u_comp.shape[-1] * d_real / (2 * xis[i]),
            -u_comp.shape[-2] * d_real / (2 * xis[i]),
            u_comp.shape[-2] * d_real / (2 * xis[i]),
        ]
        comp = (np.abs(u_comp[i, :, :, :, :]) ** 2).sum(axis=-3).mean(axis=0)
        inc = (np.abs(u_inc[i, :, :, :, :]) ** 2).sum(axis=-3).mean(axis=0)
        im0.set_data(comp)
        im0.set_clim(np.nanmin(comp), np.nanmax(comp))
        im1.set_data(inc)
        im1.set_clim(np.nanmin(inc), np.nanmax(inc))
        im0.set_extent(ext_real)
        im1.set_extent(ext_real)
        for a in ax:
            a.set_xlim((ext_real[0], ext_real[1]))
            a.set_ylim((ext_real[2], ext_real[3]))
        fig.canvas.draw()
        if plot:
            plt.show()
        mat = np.array(fig.canvas.renderer._renderer)
        mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        video_writer.write(mat)
        fig.savefig(f"{path_dn}/{scan}/energies_{i}.svg", dpi=300)
        window = 100
        for a in ax:
            a.set_xlim((-window, window))
            a.set_ylim((-window, window))
        fig.canvas.draw()
        mat = np.array(fig.canvas.renderer._renderer)
        mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        video_writer1.write(mat)
        fig.savefig(f"{path_dn}/{scan}/energies_rescaled_{i}.svg", dpi=300)
        if plot:
            plt.show()
    plt.close("all")
    video_writer.release()
    video_writer1.release()


def energy(scan: str, plot: bool = False):
    """Compute the energy.

    Args:
        scan (str): File path.
        plot (bool, optional): Whether to plot. Defaults to False.
    """
    print("Loading data ...")
    t0 = time.perf_counter()
    field_ref = np.load(f"{path_dn}/{scan}/field_ref.npy")
    N_times = field_ref.shape[0]
    N_avg = field_ref.shape[1]
    field = np.load(f"{path_dn}/{scan}/field.npy")
    field_vortex = np.load(f"{path_dn}/{scan}/field_vortex.npy")
    taus = np.load(f"{path_dn}/{scan}/taus.npy")
    taus_err = np.load(f"{path_dn}/{scan}/taus_err.npy")
    t = time.perf_counter() - t0
    sz = field_ref.nbytes + field.nbytes + field_vortex.nbytes
    rate = sz / t
    print(
        f"Loaded {sz*1e-6:.0f} MB of data in {t:.2f} s / {rate*1e-6:.2f} MB/s"
    )
    filtering_radius = 4
    # field = ndimage.gaussian_filter(field, filtering_radius)
    # field_vortex = ndimage.gaussian_filter(field_vortex, filtering_radius)
    # compute energies
    e_comp = np.zeros(field.shape[:2])
    e_inc = np.zeros(field.shape[:2])
    e_int = np.zeros(field.shape[:2])
    u_inc = np.zeros(
        (*field.shape[:2], 2, field.shape[-2], field.shape[-1]),
        dtype=np.float32,
    )
    u_comp = np.zeros(
        (*field.shape[:2], 2, field.shape[-2], field.shape[-1]),
        dtype=np.float32,
    )
    e_comp_vortex = np.zeros(field.shape[:2])
    e_inc_vortex = np.zeros(field.shape[:2])
    e_int_vortex = np.zeros(field.shape[:2])
    u_inc_vortex = np.zeros(
        (*field.shape[:2], 2, field.shape[-2], field.shape[-1]),
        dtype=np.float32,
    )
    u_comp_vortex = np.zeros(
        (*field.shape[:2], 2, field.shape[-2], field.shape[-1]),
        dtype=np.float32,
    )
    for i in tqdm.tqdm(range(N_times), desc="Computing energies", position=1):
        for j in range(N_avg):
            contrast.exp_angle_fast(field[i, j, :, :], field_ref[i, j, :, :])
            contrast.exp_angle_fast_scalar(
                field[i, j, :, :],
                field[i, j, field.shape[-2] // 2, field.shape[-1] // 2],
            )
            contrast.exp_angle_fast(
                field_vortex[i, j, :, :], field_ref[i, j, :, :]
            )
            contrast.exp_angle_fast_scalar(
                field_vortex[i, j, :, :],
                field_vortex[
                    i,
                    j,
                    field_vortex.shape[-2] // 2,
                    field_vortex.shape[-1] // 2,
                ],
            )
            rho = (
                field[i, j, :, :].real * field[i, j, :, :].real
                + field[i, j, :, :].imag * field[i, j, :, :].imag
            )
            rho_vortex = (
                field_vortex[i, j, :, :].real * field_vortex[i, j, :, :].real
                + field_vortex[i, j, :, :].imag * field_vortex[i, j, :, :].imag
            )
            rho_ref = (
                field_ref[i, j, :, :].real * field_ref[i, j, :, :].real
                + field_ref[i, j, :, :].imag * field_ref[i, j, :, :].imag
            )
            rho_ref = ndimage.gaussian_filter(rho_ref, 10)
            (
                _,
                u_inc[i, j, :, :, :],
                u_comp[i, j, :, :, :],
            ) = velocity.helmholtz_decomp(
                field[i, j, :, :], dx=d_real, plot=False
            )
            u_inc[i, j, :, :, :] = ndimage.gaussian_filter(
                u_inc[i, j, :, :, :], filtering_radius
            )
            u_comp[i, j, :, :, :] = ndimage.gaussian_filter(
                u_comp[i, j, :, :, :], filtering_radius
            )
            e_comp[i, j] = np.sum(np.abs(u_comp[i, j, :, :, :]) ** 2)
            e_inc[i, j] = np.sum(np.abs(u_inc[i, j, :, :, :]) ** 2)
            e_int[i, j] = np.sum(rho * rho)
            (
                _,
                u_inc_vortex[i, j, :, :, :],
                u_comp_vortex[i, j, :, :, :],
            ) = velocity.helmholtz_decomp(
                field_vortex[i, j, :, :], dx=d_real, plot=False
            )
            u_inc_vortex[i, j, :, :, :] = ndimage.gaussian_filter(
                u_inc_vortex[i, j, :, :, :], filtering_radius
            )
            u_comp_vortex[i, j, :, :, :] = ndimage.gaussian_filter(
                u_comp_vortex[i, j, :, :, :], filtering_radius
            )
            e_comp_vortex[i, j] = np.sum(
                np.abs(u_comp_vortex[i, j, :, :, :]) ** 2
            )
            e_inc_vortex[i, j] = np.sum(
                np.abs(u_inc_vortex[i, j, :, :, :]) ** 2
            )
            e_int_vortex[i, j] = np.sum(rho_vortex * rho_vortex)
    print("\nSaving data ...")
    t0 = time.perf_counter()
    np.save(f"{path_dn}/{scan}/e_comp.npy", e_comp)
    np.save(f"{path_dn}/{scan}/e_inc.npy", e_inc)
    np.save(f"{path_dn}/{scan}/e_int.npy", e_int)
    np.save(f"{path_dn}/{scan}/u_inc.npy", u_inc)
    np.save(f"{path_dn}/{scan}/u_comp.npy", u_comp)
    np.save(f"{path_dn}/{scan}/e_comp_vortex.npy", e_comp_vortex)
    np.save(f"{path_dn}/{scan}/e_inc_vortex.npy", e_inc_vortex)
    np.save(f"{path_dn}/{scan}/e_int_vortex.npy", e_int_vortex)
    np.save(f"{path_dn}/{scan}/u_inc_vortex.npy", u_inc_vortex)
    np.save(f"{path_dn}/{scan}/u_comp_vortex.npy", u_comp_vortex)
    t = time.perf_counter() - t0
    sz = (
        e_comp.nbytes
        + e_inc.nbytes
        + u_inc.nbytes
        + u_comp.nbytes
        + e_comp_vortex.nbytes
        + e_inc_vortex.nbytes
        + u_inc_vortex.nbytes
        + u_comp_vortex.nbytes
    )
    rate = sz / t
    print(
        f"Saved {sz*1e-6:.0f} MB of data in {t:.2f} s / {rate*1e-6:.2f} MB/s"
    )
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    ax[0].set_title("Fluid")
    ax[0].errorbar(
        taus,
        np.mean(e_comp, axis=-1),
        xerr=taus_err,
        yerr=np.std(e_comp, axis=-1),
        label="Compressible",
        marker="o",
    )
    ax[0].errorbar(
        taus,
        np.mean(e_inc, axis=-1),
        xerr=taus_err,
        yerr=np.std(e_inc, axis=-1),
        label="Incompressible",
        marker="s",
    )
    ax[0].errorbar(
        taus,
        np.mean(e_int, axis=-1),
        xerr=taus_err,
        yerr=np.std(e_int, axis=-1),
        label="Interaction",
        marker="^",
    )
    ax[1].set_title("Vortex")
    ax[1].errorbar(
        taus,
        np.mean(e_comp_vortex, axis=-1),
        xerr=taus_err,
        yerr=np.std(e_comp_vortex, axis=-1),
        label="Compressible",
        marker="o",
    )
    ax[1].errorbar(
        taus,
        np.mean(e_inc_vortex, axis=-1),
        xerr=taus_err,
        yerr=np.std(e_inc_vortex, axis=-1),
        label="Incompressible",
        marker="s",
    )
    ax[1].errorbar(
        taus,
        np.mean(e_int_vortex, axis=-1),
        xerr=taus_err,
        yerr=np.std(e_int_vortex, axis=-1),
        label="Interaction",
        marker="^",
    )
    ax[2].set_title("Fluid normalized")
    ax[2].errorbar(
        taus,
        np.mean(e_comp, axis=-1) / np.mean(e_comp_vortex, axis=-1),
        xerr=taus_err,
        label="Compressible",
        marker="o",
    )
    ax[2].errorbar(
        taus,
        np.mean(e_inc, axis=-1) / np.mean(e_inc_vortex, axis=-1),
        xerr=taus_err,
        label="Incompressible",
        marker="s",
    )
    ax[2].errorbar(
        taus,
        np.mean(e_int, axis=-1) / np.mean(e_int_vortex, axis=-1),
        xerr=taus_err,
        label="Interaction",
        marker="^",
    )
    for a in ax:
        a.set_xlabel(r"$\tau$ in rad")
        a.set_ylabel(r"$E$ in a.u.")
        a.legend()
    fig.savefig(f"{path_dn}/{scan}/energies.svg", dpi=300)
    if plot:
        plt.show()
    plt.close("all")


def temperature_monitoring(scan: str, plot=False):
    """Plots the temperature monitoring

    Args:
        scan (str): Path to the dataset
        plot (bool, optional): Whether to plot the temperature. Defaults to False.
    """
    N_measurements = np.load(f"{path_temp}/{scan}/N_measurements.npy")
    for i in tqdm.tqdm(range(N_measurements), desc="Plotting spectra"):
        times = np.load(f"{path_temp}/{scan}/times_{i}.npy")
        data = np.load(f"{path_temp}/{scan}/signal_{i}.npy")
        data = (data.T / np.max(data, axis=1)).T
        data = data[:, ::10]
        times = times[:, ::10]
        data = ndimage.uniform_filter1d(data, size=10)
        time_string = np.load(f"{path_temp}/{scan}/datetime_{i}.npy").tobytes()
        time_string = time_string.decode("utf-8")
        if i == 0:
            fig, ax = plt.subplots(layout="constrained", dpi=300)
            (ln0,) = ax.plot(
                times[0, :] * 1e3, data[0, :], label="Transmitted", zorder=10
            )
            (ln1,) = ax.plot(times[0, :] * 1e3, data[1, :], label="SAS")
            (ln2,) = ax.plot(
                times[0, :] * 1e3, data[2, :], label="Normalization"
            )
            (ln3,) = ax.plot(
                times[0, :] * 1e3, data[3, :], label="Fabry-Pérot"
            )
            ax.set_xlabel("Time in ms")
            ax.set_ylabel("Signal in a.u")
            fig.suptitle(f"Temperature monitoring at {i}")
            ax.legend()
            fig.canvas.draw()
            mat = np.array(fig.canvas.renderer._renderer)
            mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
            video_writer = cv2.VideoWriter(
                f"{path_temp}/{scan}/temp.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                5,
                (mat.shape[1], mat.shape[0]),
            )
            video_writer.write(mat)
            if plot:
                plt.show()
        else:
            fig.suptitle(f"Temperature monitoring at {i}")
            ln0.set_ydata(data[0, :])
            ln1.set_ydata(data[1, :])
            ln2.set_ydata(data[2, :])
            ln3.set_ydata(data[3, :])
            fig.canvas.draw()
            mat = np.array(fig.canvas.renderer._renderer)
            mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
            video_writer.write(mat)
    video_writer.release()


if __name__ == "__main__":
    # scan = "02211822_time_opposite"
    # compute_phase_time(scan, plot=True)
    # plot_fields(scan, plot=True)
    # energy(scan, plot=True)
    # plot_energy(scan, plot=False)

    # temperature_monitoring(scan, plot=False)
    scans = next(os.walk(path_dn))[1]
    scans = natural_sort(scans)
    # scans.remove("@eaDir")
    scans = scans[-1:]
    print(scans)
    for s in scans:
        # compute_phase_time(s, plot=False)
        # compute_phase_full_time(s, plot=False)
        # compute_phase_dist(s, plot=False)
        # plot_fields(s, plot=False)
        energy(s, plot=True)
        plot_energy(s, plot=False)
