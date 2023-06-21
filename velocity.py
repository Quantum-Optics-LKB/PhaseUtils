# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 20:45:44 2022

@author: Tangui Aladjidi
"""
from scipy import spatial
from numba import cuda
import numba
import matplotlib.pyplot as plt
import time
import math
import numpy as np
import pyfftw
import pickle
import cupy as cp
import networkx as nx
pyfftw.interfaces.cache.enable()

# simple timing decorator


def timer(func):
    """This function shows the execution time of
    the function object passed"""
    def wrap_func(*args, **kwargs):
        t1 = time.perf_counter()
        result = func(*args, **kwargs)
        t2 = time.perf_counter()
        print(f'Function {func.__name__!r} executed in {(t2-t1)*1e3:.4f} ms')
        return result
    return wrap_func


def timer_repeat(func, *args, N_repeat=1000):
    t = np.zeros(N_repeat, dtype=np.float32)
    for i in range(N_repeat):
        t0 = time.perf_counter()
        func(*args)
        t1 = time.perf_counter()
        t[i] = t1-t0
    print(f"{N_repeat} executions of {func.__name__!r} {np.mean(t)*1e3:.3f} +/- {np.std(t)*1e3:.3f} ms per loop (min : {np.min(t)*1e3:.3f} / max : {np.max(t)*1e3:.3f} ms / med: {np.median(t)*1e3:.3f})")
    return np.mean(t), np.std(t)


def timer_repeat_cp(func, *args, N_repeat=1000):
    t = np.zeros(N_repeat, dtype=np.float32)
    for i in range(N_repeat):
        start_gpu = cp.cuda.Event()
        end_gpu = cp.cuda.Event()
        start_gpu.record()
        func(*args)
        end_gpu.record()
        end_gpu.synchronize()
        t[i] = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    print(f"{N_repeat} executions of {func.__name__!r} {np.mean(t):.3f} +/- {np.std(t):.3f} ms per loop (min : {np.min(t):.3f} / max : {np.max(t):.3f} ms / med: {np.median(t):.3f})")
    return np.mean(t), np.std(t)


@numba.njit(parallel=True, cache=True, fastmath=True)
def az_avg(image: np.ndarray, center: tuple) -> np.ndarray:
    """Calculates the azimuthally averaged radial profile.

    Args:
        image (np.ndarray): The 2D image
        center (tuple): The [x,y] pixel coordinates used as the center. Defaults to None,
        which then uses the center of the image (including fractional pixels).

    Returns:
        np.ndarray: prof the radially averaged profile
    """
    # Calculate the indices from the image
    R = np.empty_like(image)
    max_r = max([np.hypot(center[0], center[1]),
                 np.hypot(center[0]-image.shape[1], center[1]),
                 np.hypot(center[0]-image.shape[1],
                          center[1]-image.shape[0]),
                 np.hypot(center[0], center[1]-image.shape[0])])
    r = np.arange(1, int(max_r)+1, 1)
    prof = np.zeros_like(r, dtype=np.float64)
    prof_counts = np.zeros_like(r, dtype=np.uint64)
    for i in numba.prange(image.shape[0]):
        for j in range(image.shape[1]):
            dist = round(np.hypot(i-center[1], j-center[0]))
            prof[dist] += image[i, j]
            prof_counts[dist] += 1
    prof /= prof_counts
    return prof


@cuda.jit(fastmath=True)
def _az_avg_cp(image: cp.ndarray, prof: cp.ndarray, prof_counts: cp.ndarray,
               center: tuple):
    """Kernel for azimuthal average calculation

    Args:
        image (cp.ndarray): The image from which to calculate the azimuthal average
        prof (cp.ndarray): A vector containing the bins 
        prof_counts (cp.ndarray): A vector of same size as prof to count each bin
    """
    i, j = numba.cuda.grid(2)
    if i < image.shape[0] and j < image.shape[1]:
        dist = round(math.sqrt((i-center[1])**2+(j-center[0])**2))
        prof[dist] += image[i, j]
        prof_counts[dist] += 1


def az_avg_cp(image: cp.ndarray, center: tuple) -> cp.ndarray:
    """Calculates the azimuthally averaged radial profile.

    Args:
        image (cp.ndarray): The 2D image
        center (tuple): The [x,y] pixel coordinates used as the center. Defaults to None,
        which then uses the center of the image (including fractional pixels).

    Returns:
        cp.ndarray: prof the radially averaged profile
    """
    # Calculate the indices from the image
    R = cp.empty_like(image)
    max_r = max([cp.hypot(center[0], center[1]),
                 cp.hypot(center[0]-image.shape[1], center[1]),
                 cp.hypot(center[0]-image.shape[1],
                          center[1]-image.shape[0]),
                 cp.hypot(center[0], center[1]-image.shape[0])])
    r = cp.arange(1, int(max_r)+1, 1)
    prof = cp.zeros_like(r, dtype=np.float32)
    prof_counts = cp.zeros_like(r, dtype=np.float32)
    tpb = 16
    bpgx = math.ceil(image.shape[0]/tpb)
    bpgy = math.ceil(image.shape[1]/tpb)
    _az_avg_cp[(bpgx, bpgy), (tpb, tpb)](image, prof, prof_counts, center)
    prof /= prof_counts
    return prof


@numba.njit(numba.float32[:, :](numba.float32[:, :, :], numba.int64), fastmath=True,
            cache=True, parallel=True)
def phase_sum(velo: np.ndarray, r: int = 1):
    cont = np.zeros((velo.shape[1], velo.shape[2]), dtype=np.float32)
    for i in numba.prange(velo.shape[1]):
        for j in range(velo.shape[2]):
            for k in range(r+1):
                cont[i, j] += velo[0, i, (j+k) % velo.shape[2]]
                cont[i, j] -= velo[0, (i+r) % velo.shape[1],
                                   (j+k) % velo.shape[2]]
                cont[i, j] += velo[1, (i+k) % velo.shape[1],
                                   (j+r) % velo.shape[2]]
                cont[i, j] -= velo[1, (i+k) % velo.shape[1], j]
    return cont


@cuda.jit((numba.float32[:, :, :], numba.float32[:, :], numba.int64), fastmath=True)
def phase_sum_cp(velo, cont, r):
    """Computes the phase gradient winding

    Args:
        grad (cp.ndarray): Phase differences. grad[0, :, :] is d/dy phi (derivative along rows).
        cont (cp.ndarray): output array
    Returns:
        None
    """
    i, j = numba.cuda.grid(2)
    if i < velo.shape[1] and j < velo.shape[2]:
        for k in range(r+1):
            cont[i, j] += velo[0, i, (j+k) % velo.shape[2]]
            cont[i, j] -= velo[0, (i+r) % velo.shape[1],
                               (j+k) % velo.shape[2]]
            cont[i, j] += velo[1, (i+k) % velo.shape[1],
                               (j+r) % velo.shape[2]]
            cont[i, j] -= velo[1, (i+k) % velo.shape[1], j]


def velocity(phase: np.ndarray, dx: float = 1) -> np.ndarray:
    """Returns the velocity from the phase

    Args:
        phase (np.ndarray): The field phase
        dx (float, optional): the pixel size in m. Defaults to 1 (adimensional).

    Returns:
        np.ndarray: The velocity field [vx, vy]
    """
    # 1D unwrap
    phase_unwrap = np.empty(
        (2, phase.shape[0], phase.shape[1]), dtype=np.float32)
    phase_unwrap[0, :, :] = np.unwrap(phase, axis=1)
    phase_unwrap[1, :, :] = np.unwrap(phase, axis=0)
    # gradient reconstruction
    velo = np.empty(
        (2, phase.shape[0], phase.shape[1]), dtype=np.float32)
    velo[0, :, :] = np.gradient(phase_unwrap[0, :, :], dx, axis=1)
    velo[1, :, :] = np.gradient(phase_unwrap[1, :, :], dx, axis=0)
    return velo


def velocity_cp(phase: cp.ndarray, dx: float = 1) -> cp.ndarray:
    """Returns the velocity from the phase

    Args:
        phase (np.ndarray): The field phase
        dx (float, optional): the pixel size in m. Defaults to 1 (adimensional).

    Returns:
        np.ndarray: The velocity field [vx, vy]
    """
    # 1D unwrap
    phase_unwrap = cp.empty(
        (2, phase.shape[0], phase.shape[1]), dtype=np.float32)
    phase_unwrap[0, :, :] = cp.unwrap(phase, axis=1)
    phase_unwrap[1, :, :] = cp.unwrap(phase, axis=0)
    # gradient reconstruction
    velo = cp.empty(
        (2, phase.shape[0], phase.shape[1]), dtype=np.float32)
    velo[0, :, :] = cp.gradient(phase_unwrap[0, :, :], dx, axis=1)
    velo[1, :, :] = cp.gradient(phase_unwrap[1, :, :], dx, axis=0)
    return velo


def velocity_fft(phase: np.ndarray, dx: float = 1) -> np.ndarray:
    """Returns the velocity from the phase using an fft to compute 
    the gradient

    Args:
        phase (np.ndarray): The field phase
        dx (float, optional): the pixel size in m. Defaults to 1 (adimensional).

    Returns:
        np.ndarray: The velocity field [vx, vy]
    """
    # 1D unwrap
    phase_unwrap = np.empty(
        (2, phase.shape[-2], phase.shape[-1]), dtype=np.float32)
    phase_unwrap[0, :, :] = np.unwrap(phase, axis=-1)
    phase_unwrap[1, :, :] = np.unwrap(phase, axis=-2)
    # prepare K matrix
    kx = np.fft.fftfreq(phase.shape[-1], dx)
    ky = np.fft.fftfreq(phase.shape[-2], dx)
    Kx, Ky = np.meshgrid(kx, ky)
    # gradient reconstruction
    velo = np.empty(
        (2, phase.shape[-2], phase.shape[-1]), dtype=np.float32)
    velo[0, :, :] = np.fft.ifft2(Kx*np.fft.fft2(phase_unwrap[0, :, :]))
    velo[1, :, :] = np.fft.ifft2(Ky*np.fft.fft2(phase_unwrap[1, :, :]))
    return velo


def velocity_fft_cp(phase: cp.ndarray, dx: float = 1) -> cp.ndarray:
    """Returns the velocity from the phase using an fft to compute 
    the gradient

    Args:
        phase (cp.ndarray): The field phase
        dx (float, optional): the pixel size in m. Defaults to 1 (adimensional).

    Returns:
        cp.ndarray: The velocity field [vx, vy]
    """
    # 1D unwrap
    phase_unwrap = cp.empty(
        (2, phase.shape[-2], phase.shape[-1]), dtype=np.float32)
    phase_unwrap[0, :, :] = cp.unwrap(phase, axis=-1)
    phase_unwrap[1, :, :] = cp.unwrap(phase, axis=-2)
    # prepare K matrix
    kx = 2*np.pi*cp.fft.fftfreq(phase.shape[-1], dx)
    ky = 2*np.pi*cp.fft.fftfreq(phase.shape[-2], dx)
    K = cp.array(cp.meshgrid(kx, ky))
    # gradient reconstruction
    velo = cp.empty(
        (2, phase.shape[-2], phase.shape[-1]), dtype=np.float32)
    velo = cp.fft.irfft2(1j*K[:, :, 0:K.shape[-1]//2+1]
                         * cp.fft.rfft2(phase_unwrap))
    return velo


def helmholtz_decomp(field: np.ndarray, plot=False, dx: float = 1) -> tuple:
    """Decomposes a phase picture into compressible and incompressible velocities

    Args:
        field (np.ndarray): 2D array of the field
        plot (bool, optional): Final plots. Defaults to True.
        dx (float, optional): Spatial sampling size in m. Defaults to 1.
    Returns:
        tuple: (velo, v_incc, v_comp) a tuple containing the velocity field,
        the incompressible velocity and compressible velocity.
    """
    # try to load previous fftw wisdom
    try:
        with open("fft.wisdom", "rb") as file:
            wisdom = pickle.load(file)
            pyfftw.import_wisdom(wisdom)
    except FileNotFoundError:
        print("No FFT wisdom found, starting over ...")
    sy, sx = field.shape
    # meshgrid in k space
    kx = 2*np.pi*np.fft.fftfreq(sx, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(sy, d=dx)
    K = np.array(np.meshgrid(kx, ky))
    phase = np.angle(field)
    velo = np.abs(field)*velocity(phase, dx)

    v_tot = np.hypot(velo[0], velo[1])
    V_k = pyfftw.interfaces.numpy_fft.fft2(velo)

    # Helmholtz decomposition fot the compressible part
    V_comp = -1j*np.sum(V_k*K, axis=0)/((np.sum(K**2, axis=0))+1e-15)
    v_comp = np.real(pyfftw.interfaces.numpy_fft.ifft2(1j*V_comp*K))

    # Helmholtz decomposition fot the incompressible part
    v_inc = velo - v_comp
    # save FFT wisdom
    with open("fft.wisdom", "wb") as file:
        wisdom = pyfftw.export_wisdom()
        pickle.dump(wisdom, file)
    if plot:
        flow = np.hypot(v_inc[0], v_inc[1])
        YY, XX = np.indices(flow.shape)
        fig, ax = plt.subplots(2, 2, figsize=[12, 9])
        im0 = ax[0, 0].imshow(v_tot)
        ax[0, 0].set_title(r'$|v^{tot}|$')
        ax[0, 0].set_xlabel('x')
        ax[0, 0].set_ylabel('y')
        fig.colorbar(im0, ax=ax[0, 0])

        im1 = ax[0, 1].imshow(flow)
        ax[0, 1].set_title(r'$|v^{inc}|$')
        ax[0, 1].set_xlabel('x')
        ax[0, 1].set_ylabel('y')
        fig.colorbar(im1, ax=ax[0, 1])

        im2 = ax[1, 0].imshow(np.hypot(v_comp[0], v_comp[1]))
        ax[1, 0].set_title(r'$|v^{comp}|$')
        ax[1, 0].set_xlabel('x')
        ax[1, 0].set_ylabel('y')
        fig.colorbar(im2, ax=ax[1, 0])

        # flows are calculated by streamplot
        im3 = ax[1, 1].imshow(flow, cmap='viridis')
        ax[1, 1].streamplot(XX, YY, v_inc[0], v_inc[1],
                            density=5, color='white', linewidth=1)
        ax[1, 1].set_title(r'$v^{inc}$')
        ax[1, 1].set_xlabel('x')
        ax[1, 1].set_ylabel('y')
        fig.colorbar(im3, ax=ax[1, 1], label=r'$|v^{inc}|$')
        plt.show()

    return velo, v_inc, v_comp


def helmholtz_decomp_cp(field: np.ndarray, plot: bool = False, dx: float = 1,
                        regularize: bool = True) -> tuple:
    """Decomposes a phase picture into compressible and incompressible velocities

    Args:
        field (np.ndarray): 2D array of the field
        plot (bool, optional): Final plots. Defaults to True.
        dx (float, optional): Spatial sampling size in m. Defaults to 1.
        regularize (bool, optional): Whether to multiply speed by the amplitude or not.
    Returns:
        tuple: (velo, v_incc, v_comp) a tuple containing the velocity field,
        the incompressible velocity and compressible velocity.
    """
    sy, sx = field.shape
    # meshgrid in k space
    kx = 2*np.pi*cp.fft.fftfreq(sx, d=dx)
    ky = 2*np.pi*cp.fft.fftfreq(sy, d=dx)
    K = cp.array(cp.meshgrid(kx, ky))
    if regularize:
        velo = cp.abs(field)*velocity_fft_cp(cp.angle(field))
    else:
        velo = velocity_fft_cp(cp.angle(field))
    v_tot = cp.hypot(velo[0], velo[1])
    V_k = cp.fft.fft2(velo)
    # Helmohltz decomposition fot the compressible part
    V_comp = -1j*cp.sum(V_k*K, axis=0)/((cp.sum(K**2, axis=0))+1e-15)
    v_comp = cp.real(cp.fft.ifft2(1j*V_comp*K))
    # Helmohltz decomposition fot the incompressible part
    v_inc = velo - v_comp
    if plot:
        flow_inc = cp.hypot(v_inc[0], v_inc[1])
        flow_comp = cp.hypot(v_comp[0], v_comp[1])
        YY, XX = np.indices(flow_comp.shape)
        fig, ax = plt.subplots(2, 2, figsize=[12, 9])
        im0 = ax[0, 0].imshow(v_tot.get())
        ax[0, 0].set_title(r'$|v^{tot}|$')
        ax[0, 0].set_xlabel('x')
        ax[0, 0].set_ylabel('y')
        fig.colorbar(im0, ax=ax[0, 0])

        im1 = ax[0, 1].imshow(flow_inc.get())
        ax[0, 1].set_title(r'$|v^{inc}|$')
        ax[0, 1].set_xlabel('x')
        ax[0, 1].set_ylabel('y')
        fig.colorbar(im1, ax=ax[0, 1])

        im2 = ax[1, 0].imshow(flow_comp.get())
        ax[1, 0].streamplot(XX, YY, v_comp[0].get(), v_comp[1].get(),
                            density=2.5, color='white', linewidth=1)
        ax[1, 0].set_title(r'$|v^{comp}|$')
        ax[1, 0].set_xlabel('x')
        ax[1, 0].set_ylabel('y')
        fig.colorbar(im2, ax=ax[1, 0])

        # flows are calculated by streamplot
        im3 = ax[1, 1].imshow(flow_inc.get(), cmap='viridis')
        ax[1, 1].streamplot(XX, YY, v_inc[0].get(), v_inc[1].get(),
                            density=2.5, color='white', linewidth=1)
        ax[1, 1].set_title(r'$v^{inc}$')
        ax[1, 1].set_xlabel('x')
        ax[1, 1].set_ylabel('y')
        fig.colorbar(im3, ax=ax[1, 1], label=r'$|v^{inc}|$')
        plt.show()
    return velo, v_inc, v_comp


def energy(ucomp: np.ndarray, uinc: np.ndarray) -> tuple:
    """Computes the total energy contained in the given compressible 
    and incompressible velocities

    Args:
        ucomp (np.ndarray): Compressible velocity field
        uinc (np.ndarray): Incompressible velocity field

    Returns:
        (Ucc, Uii): The total compressible and incompressible energies
    """
    # compressible
    Ux_c = np.abs(np.fft.fftshift(np.fft.fft2(ucomp[0])))
    Uy_c = np.abs(np.fft.fftshift(np.fft.fft2(ucomp[1])))
    Uc = Ux_c**2 + Uy_c**2
    Ucc = np.sum(Uc)

    # incompressible
    Ux_i = np.abs(np.fft.fftshift(np.fft.fft2(uinc[0])))
    Uy_i = np.abs(np.fft.fftshift(np.fft.fft2(uinc[1])))
    Ui = Ux_i**2 + Uy_i**2
    Uii = np.sum(Ui)

    return Ucc, Uii


def energy_spectrum(ucomp: np.ndarray, uinc: np.ndarray) -> np.ndarray:
    """Computes the compressible and incompressible energy spectra
    using the Fourier transform of the velocity fields

    Args:
        ucomp (np.ndarray): Compressible velocity field
        uinc (np.ndarray): Incompressible velocity field

    Returns:
        (Ucc, Uii) np.ndarray: The array containing the compressible / incompressible
        energies as a function of the wavevector k
    """

    # compressible
    Ux_c = np.abs(np.fft.fftshift(np.fft.fft2(ucomp[0])))
    Uy_c = np.abs(np.fft.fftshift(np.fft.fft2(ucomp[1])))
    Uc = Ux_c**2 + Uy_c**2
    Ucc = az_avg(Uc, center=(Uc.shape[1]//2, Uc.shape[0]//2))

    # incompressible
    Ux_i = np.abs(np.fft.fftshift(np.fft.fft2(uinc[0])))
    Uy_i = np.abs(np.fft.fftshift(np.fft.fft2(uinc[1])))
    Ui = Ux_i**2 + Uy_i**2
    Uii = az_avg(Ui, center=(Ui.shape[1]//2, Ui.shape[0]//2))

    return Ucc, Uii


def energy_cp(ucomp: cp.ndarray, uinc: cp.ndarray) -> tuple:
    """Computes the total energy contained in the given compressible 
    and incompressible velocities

    Args:
        ucomp (np.ndarray): Compressible velocity field
        uinc (np.ndarray): Incompressible velocity field

    Returns:
        (Ucc, Uii): The total compressible and incompressible energies
    """
    # compressible
    Ux_c = cp.abs(cp.fft.fftshift(cp.fft.fft2(ucomp[0])))
    Uy_c = cp.abs(cp.fft.fftshift(cp.fft.fft2(ucomp[1])))
    Uc = Ux_c**2 + Uy_c**2
    Ucc = cp.sum(Uc)

    # incompressible
    Ux_i = cp.abs(cp.fft.fftshift(cp.fft.fft2(uinc[0])))
    Uy_i = cp.abs(cp.fft.fftshift(cp.fft.fft2(uinc[1])))
    Ui = Ux_i**2 + Uy_i**2
    Uii = cp.sum(Ui)

    return Ucc, Uii


def energy_spectrum_cp(ucomp: cp.ndarray, uinc: cp.ndarray) -> cp.ndarray:
    """Computes the compressible and incompressible energy spectra
    using the Fourier transform of the velocity fields

    Args:
        ucomp (np.ndarray): Compressible velocity field
        uinc (np.ndarray): Incompressible velocity field

    Returns:
        (Ucc, Uii) np.ndarray: The array containing the compressible / incompressible
        energies as a function of the wavevector k
    """
    # compressible
    Ux_c = cp.abs(cp.fft.fftshift(cp.fft.fft2(ucomp[0])))
    Uy_c = cp.abs(cp.fft.fftshift(cp.fft.fft2(ucomp[1])))
    Uc = Ux_c**2 + Uy_c**2
    Ucc = az_avg_cp(Uc, center=(Uc.shape[1]//2, Uc.shape[0]//2))

    # incompressible
    Ux_i = cp.abs(cp.fft.fftshift(cp.fft.fft2(uinc[0])))
    Uy_i = cp.abs(cp.fft.fftshift(cp.fft.fft2(uinc[1])))
    Ui = Ux_i**2 + Uy_i**2
    Uii = az_avg_cp(Ui, center=(Ui.shape[1]//2, Ui.shape[0]//2))


def vortex_detection(phase: np.ndarray, plot: bool = False, r: int = 1) -> np.ndarray:
    """Detects the vortex positions using circulation calculation

    Args:
        phase (np.ndarray): Phase field.
        plot (bool, optional): Whether to plot the result or not. Defaults to True.

    Returns:
        np.ndarray: A list of the vortices position and charge
    """
    velo = velocity(phase)
    windings = phase_sum(velo, r)
    plus_y, plus_x = np.where(windings > 2*np.pi)
    minus_y, minus_x = np.where(windings < -2*np.pi)
    vortices = np.zeros((len(plus_x)+len(minus_x), 3), dtype=np.float32)
    vortices[0:len(plus_x), 0] = plus_x
    vortices[0:len(plus_x), 1] = plus_y
    vortices[0:len(plus_x), 2] = 1
    vortices[len(plus_x):, 0] = minus_x
    vortices[len(plus_x):, 1] = minus_y
    vortices[len(plus_x):, 2] = -1
    if plot:
        velo = velocity(phase)
        fig, ax = plt.subplots(figsize=[12, 9])
        im = plt.imshow(phase, cmap='twilight_shifted')
        ax.scatter(vortices[:, 0], vortices[:, 1],
                   c=vortices[:, 2], cmap='bwr')
        fig.colorbar(im, ax=ax, label="Phase")
        plt.show()
    return vortices


def vortex_detection_cp(phase: cp.ndarray, plot: bool = False,
                        r: int = 1) -> cp.ndarray:
    """Detects the vortex positions using circulation calculation

    Args:
        phase (np.ndarray): Phase field.
        plot (bool, optional): Whether to plot the result or not. Defaults to True.

    Returns:
        np.ndarray: A list of the vortices position and charge
    """
    velo = velocity_cp(phase)
    windings = cp.zeros_like(velo[0])
    tpb = 32
    bpgx = math.ceil(windings.shape[0]/tpb)
    bpgy = math.ceil(windings.shape[1]/tpb)
    phase_sum_cp[(bpgx, bpgy), (tpb, tpb)](velo, windings, r)
    plus_y, plus_x = cp.where(windings > 2*np.pi)
    minus_y, minus_x = cp.where(windings < -2*np.pi)
    vortices = cp.zeros((len(plus_x)+len(minus_x), 3), dtype=np.float32)
    vortices[0:len(plus_x), 0] = plus_x
    vortices[0:len(plus_x), 1] = plus_y
    vortices[0:len(plus_x), 2] = 1
    vortices[len(plus_x):, 0] = minus_x
    vortices[len(plus_x):, 1] = minus_y
    vortices[len(plus_x):, 2] = -1
    if plot:
        plt.figure(1, figsize=[12, 9])
        plt.imshow(phase.get(), cmap='twilight_shifted')
        plt.scatter(vortices[:, 0].get(), vortices[:, 1].get(),
                    c=vortices[:, 2].get(), cmap='bwr')
        plt.colorbar(label="Vorticity")
        plt.show()
    return vortices


@numba.njit(numba.bool_[:](numba.int64[:]), cache=True, fastmath=True)
def mutual_nearest_neighbors(nn) -> np.ndarray:
    """Returns a list of pairs of mutual nearest neighbors and
    the product of their charges

    Args:
        nn (np.ndarray): array of nearest neighbors

    Returns:
        np.ndarray: A list of booleans telling if vortex i is a mutual NN pair without
        double counting.
    """
    mutu = np.zeros(nn.shape[0], dtype=np.bool_)
    for k in range(nn.shape[0]):
        next_closest = nn[k]
        if nn[next_closest] == k and not mutu[next_closest]:
            mutu[k] = True
    return mutu


def build_pairs(vortices: np.ndarray, nn: np.ndarray, mutu: np.ndarray,
                queue: np.ndarray):
    """Builds the dipoles and the pairs of same sign

    Args:
        vortices (np.ndarray): Vortices
        ranking (np.ndarray): Ranking matrix
        queue (np.ndarray): Vortices still under consideration
        mutu (np.ndarray): Mutual nearest neighbors
    Returns:
        dipoles, pairs, queue : np.ndarray dipoles, clusters and updated queue
    """
    closest = nn[mutu]
    ll = vortices[:, 2]*vortices[nn, 2]
    dipoles_ = mutu[ll[mutu] == -1]
    dipoles = np.empty((len(dipoles_), 2), dtype=np.int64)
    dipoles[:, 0] = dipoles_
    dipoles[:, 1] = closest[ll[mutu] == -1]
    # remove them from queue
    queue[dipoles[:, 0]] = -1
    queue[dipoles[:, 1]] = -1
    # check pairs
    pairs_ = mutu[ll[mutu] == 1]
    pairs = np.empty((len(pairs_), 2), dtype=np.int64)
    pairs[:, 0] = pairs_
    pairs[:, 1] = closest[ll[mutu] == 1]
    queue[pairs[:, 0]] = -1
    queue[pairs[:, 1]] = -1
    # update queue
    queue = queue[queue >= 0]
    return dipoles, pairs, queue


@numba.njit(numba.types.UniTuple(numba.int64[:], 2)(
    numba.int64[:, :], numba.float64[:], numba.float64[:, :]),
    cache=True, parallel=True)
def edges_to_connect(neighbors: np.ndarray, dists_opp: np.ndarray,
                     dists: np.ndarray) -> tuple:
    """Generates arrays of edges to add applying rule 2 based on distance to closest opposite
    and vortex to same sign neighbor distance

    Args:
        vort (np.ndarray): Vortices list in which you try to establish connections
        neighbors (np.ndarray): k_th neighbor matrix N_ij is the jth neighbor of ith vortex
        dists_opp (np.ndarray): Distance to closest opposite
        dists (np.ndarray): Distance matrix

    Returns:
        tuple: (q, n) where q are the vortices to connect to n
    """
    # TODO work on the queue that starts on one edge of all same sign pairs + single vortices (more efficient)
    edges_to_add = np.zeros(dists.shape, dtype=np.bool_)
    for i in numba.prange(dists.shape[0]):
        for j in range(1, dists.shape[1]):
            if min(dists_opp[i], dists_opp[neighbors[i, j]]) > dists[i, j]:
                edges_to_add[i, j] = True
    q_to_add, nei_to_add = np.where(edges_to_add)
    return q_to_add, nei_to_add


def grow_clusters(vortices: np.ndarray, plus: np.ndarray, minus: np.ndarray,
                  tree_plus: spatial.KDTree, tree_minus: spatial.KDTree,
                  cluster_graph: nx.Graph):
    """Grows the clusters in the graph by applying rule 2 on the remaining vortices (i.e without dipoles)

    Args:
        vortices (np.ndarray): Array of vortices (x, y, charge)
        plus (np.ndarray): Array of positive charge vortices. Each element is the corresponding index in the vortices array.
        minus (np.ndarray): Same for negative charge vortices
        tree_plus (spatial.KDTree): KDTree representing the plus vortices
        tree_minus (spatial.KDTree): Same for minus vortices
        cluster_graph (nx.Graph): The graph representing all vortices
    """
    # find kth same sign neighbors
    dists_plus, neighbors_plus = tree_plus.query(
        vortices[plus, 0:2], k=len(plus)//2, workers=-1)
    dists_minus, neighbors_minus = tree_minus.query(
        vortices[minus, 0:2], k=len(minus)//2, workers=-1)
    # find closest opposite neighbors
    dists_plus_opp, plus_opp = tree_minus.query(
        vortices[plus, 0:2], k=1, workers=-1)
    dists_minus_opp, minus_opp = tree_plus.query(
        vortices[minus, 0:2], k=1, workers=-1)
    # dist to closest opposite both greater than dist between q and neighbor
    t0 = time.perf_counter()
    plus_to_add_q, plus_to_add_nei = edges_to_connect(
        neighbors_plus, dists_plus_opp, dists_plus)
    minus_to_add_q, minus_to_add_nei = edges_to_connect(
        neighbors_minus, dists_minus_opp, dists_minus)
    t0 = time.perf_counter()
    cluster_graph.add_edges_from(
        zip(plus[plus_to_add_q], plus[neighbors_plus[plus_to_add_q, plus_to_add_nei]]))
    cluster_graph.add_edges_from(
        zip(minus[minus_to_add_q], minus[neighbors_minus[minus_to_add_q, minus_to_add_nei]]))


def cluster_vortices(vortices: np.ndarray) -> list:
    """Clusters the vortices into dipomerging_clusters
        vortices (np.ndarray): Array of vortices [[x, y, l], ...]

    Returns:
        list: dipoles, clusters. Clusters are a Networkx connected_components object (i.e a list of sets).
        It needs to be converted to list of lists for plotting.
    """
    queue = np.arange(0, vortices.shape[0], 1, dtype=np.int64)
    # store vortices in tree
    tree = spatial.KDTree(vortices[:, 0:2])
    # find nearest neighbors
    nn = tree.query(
        vortices[:, 0:2], k=2, workers=-1)[1]
    # nn[i] is vortex i nearest neighbor
    nn = nn[:, 1]
    mutu = mutual_nearest_neighbors(nn)
    mutu = queue[mutu]
    # RULE 1
    dipoles, pairs, queue = build_pairs(
        vortices, nn, mutu, queue)
    assert 2*len(dipoles) + 2*pairs.shape[0] + \
        len(queue) == vortices.shape[0], "PROBLEM count"
    # extract dipole free list
    without_dipoles = np.empty(len(queue) + len(pairs), dtype=np.int64)
    without_dipoles[0:len(queue)] = queue
    without_dipoles[len(queue):len(queue)+len(pairs)] = pairs[:, 0]
    # sort plus and minus
    plus = without_dipoles[vortices[without_dipoles, 2] == 1]
    minus = without_dipoles[vortices[without_dipoles, 2] == -1]
    # build graph to represent clusters
    cluster_graph = nx.Graph()
    cluster_graph.add_nodes_from(pairs[:, 0])
    cluster_graph.add_nodes_from(pairs[:, 1])
    cluster_graph.add_edges_from(pairs.tolist())
    cluster_graph.add_nodes_from(queue)
    tree_plus = spatial.KDTree(vortices[plus, 0:2])
    tree_minus = spatial.KDTree(vortices[minus, 0:2])
    # RULE 2
    grow_clusters(vortices, plus, minus, tree_plus, tree_minus, cluster_graph)
    cluster_graph = nx.minimum_spanning_tree(cluster_graph)
    clusters = nx.connected_components(cluster_graph)
    clusters = np.array([np.array(list(c)) for c in clusters], dtype=object)
    return dipoles, clusters, cluster_graph


def cluster_histogram(clusters, plot: bool = True) -> np.ndarray:
    """Returns a histogram of the number of members in the clusters

    Args:
        clusters (np.ndarray): A set generator comprising of the vortices clustered in connected components
        plot (bool): Wether to plot the histogram
    Returns:
        hist, bin_edges (np.ndarray): Returns an histogram of the size of the clusters
    """
    lengths = np.array([len(c) for c in clusters])
    hist, bin_edges = np.histogram(lengths, bins=np.max(lengths))
    if plot:
        plt.hist(lengths, bins=np.max(lengths))
        plt.yscale('log')
        plt.xlabel("Size of the cluster")
        plt.ylabel("Number of clusters")
        plt.xlim(1, np.max(lengths))
        plt.title("Histogram of cluster size")
        plt.show()
    return hist, bin_edges


def cluster_barycenters(vortices: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Returns an array of barycenters from a list of clusters

    Args:
        vortices (np.ndarray): Vortices array (x, y, l)
        clusters (np.ndarray): Array of vortex indices [[cluster0], [cluster1], ...]

    Returns:
        np.ndarray: The array of barycenters
    """
    barys = np.zeros((len(clusters), 2), dtype=np.float32)
    for k, c in enumerate(clusters):
        length = len(c)
        barys[k, 0] = np.sum(vortices[c, 0])/length
        barys[k, 1] = np.sum(vortices[c, 1])/length
    return barys


def cluster_radii(vortices: np.ndarray, clusters: np.ndarray,
                  barys: np.ndarray) -> np.ndarray:
    """Computes the cluster radius

    Args:
        vortices (np.ndarray): Vortices array (x, y, l)
        clusters (np.ndarray): Array of vortex indices [[cluster0], [cluster1], ...]
        barys (np.ndarray): array of barycenters

    Returns:
        np.ndarray: The array of radii
    """
    radii = np.zeros(barys.shape[0], dtype=np.float32)
    for k, c in enumerate(clusters):
        length = len(c)
        radii[k] = np.sum(np.hypot(vortices[c, 0]-barys[k, 0],
                          vortices[c, 1]-barys[k, 1]))/length
    return radii


@numba.njit(numba.float32(numba.float32[:, :], numba.int64[:, :]),
            parallel=True, cache=True)
def _ck(vortices: np.ndarray, neighbors: np.ndarray) -> float:
    """Correlation kernel

    Args:
        vortices (np.ndarray): Vortices array (x, y, l)
        neighbors (np.ndarray): Array of neighbors up to the kth neighbor

    Returns:
        float: Correlation coefficient C_k
    """
    N = neighbors.shape[0]
    k = neighbors.shape[1]
    c = 0
    for i in numba.prange(N):
        for j in range(k):
            c += abs(vortices[i, 2]*vortices[j, 2])/(2*k*N)
    return c


def ck(vortices: np.ndarray, k: int) -> float:
    """Computes the correlation function C_k of an array of vortices by building a KDTree to
    speed up nearest neighbor search

    Args:
        vortices (np.ndarray): Vortices array (x,y,l)
        k (int): The kth neighbor up until which to compute the correlation

    Returns:
        float: C_k the correlation coefficient
    """
    tree = spatial.KDTree(vortices[:, 0:2])
    neighbors = tree.query(vortices[:, 0:2], k=k+1)[1]
    # remove 0th neighbor
    neighbors = neighbors[:, 1:]
    c = _ck(vortices, neighbors)
    return c


@cuda.jit(cache=True, fastmath=True)
def _distance_matrix(dist: cp.ndarray, x: cp.ndarray, y: cp.ndarray):
    """Compute distance matrix using CUDA

    Args:
        x (cp.ndarray): Nd array of points
        y (cp.ndarray): Nd array of points
    """
    i, j = numba.cuda.grid(2)
    if i < x.shape[0] and j < y.shape[0]:
        if j >= i:
            dist[i, j] += math.sqrt((x[i, 0] - y[j, 0])
                                    ** 2 + (x[i, 1] - y[j, 1])**2)
            dist[j, i] = dist[i, j]


@cuda.jit(cache=True)
def _build_condition(condition: cp.ndarray, dist: cp.ndarray, bins: cp.ndarray):
    """Constructs the array that represents the vortices pair i, j to consider
    in the bin k.

    Args:
        condition (cp.ndarray): Boolean array of shape (k, i, j) where k is an index
        running in the number of bins, i and j in the number of vortices.
        dist (cp.ndarray): Distance matrix where D_ij is the distance between the 
        vortex i and j.
        bins (cp.ndarray): The disk shells of radius r and width d within which we 
        compute the correlations between a vortex and all vortices lying in a bin.
    """
    i, j, k = numba.cuda.grid(3)
    if i < condition.shape[0] and j < condition.shape[1] and k < len(bins):
        condition[k-1, i, j] = dist[i, j] > bins[k-1]
        condition[k-1, i, j] &= dist[i, j] < bins[k]


@cuda.jit(cache=True, fastmath=True)
def _correlate(corr: cp.ndarray, vortices: cp.ndarray, bins: cp.ndarray,
               condition: cp.ndarray):
    """Compute the actual correlation function

    Args:
        corr (cp.ndarray): Output array
        vortices (cp.ndarray): Vortices array where v_i = (x, y, l)
        bins (cp.ndarray): Disk shells in which to consider vortices for the correlation
        calculation
        condition (cp.ndarray): Which vortices to consider
    """
    d = bins[1]-bins[0]
    i, j, k = numba.cuda.grid(3)
    if i < condition.shape[0] and j < condition.shape[1] and k < len(bins):
        if condition[k-1, i, j]:
            r = abs(bins[k]-d/2)
            corr[k-1] += 1/(2*np.pi*r*d*vortices.shape[0]) * \
                vortices[i, 2]*vortices[j, 2]


def pair_correlations_cp(vortices: cp.ndarray, bins: cp.ndarray) -> cp.ndarray:
    """Computes the pair correlation function for a given vortex array.
    See PHYSICAL REVIEW E 95, 052144 (2017) eq.12 

    Args:
        vortices (np.ndarray): Vortices array
        bins (np.ndarray): bins of distance in which to compute the 
        correlation function

    Returns:
        np.ndarray: The correlation function of length len(bins)
    """
    corr = cp.zeros(len(bins)-1)
    # compute distance matrix of vortices
    dist_matrix = cp.zeros(
        (vortices.shape[0], vortices.shape[0]), dtype=np.float32)
    tpb = 32
    bpgx = math.ceil(dist_matrix.shape[0]/tpb)
    bpgy = math.ceil(dist_matrix.shape[1]/tpb)
    _distance_matrix[(bpgx, bpgy), (tpb, tpb)](
        dist_matrix, vortices[:, 0:2],
        vortices[:, 0:2])
    condition = cp.zeros((len(bins),
                          dist_matrix.shape[0],
                          dist_matrix.shape[1]), dtype=np.bool8)
    tpb = 16
    tpbz = 4
    bpgx = math.ceil(dist_matrix.shape[0]/tpb)
    bpgy = math.ceil(dist_matrix.shape[1]/tpb)
    bpgz = math.ceil(len(bins/tpb))
    _build_condition[(bpgx, bpgy, bpgz), (tpb, tpb, tpbz)](
        condition, dist_matrix, bins)
    _correlate[(bpgx, bpgy, bpgz), (tpb, tpb, tpbz)](
        corr, vortices, bins, condition)
    return corr


def drag_force(psi: np.ndarray, U: np.ndarray) -> tuple:
    """Computes the drag force considering an obstacle map U(r)
    and an intensity map I(r)

    Args:
        psi (np.ndarray): Intensity map
        U (np.ndarray): Potential map

    Returns:
        fx, fy (float): The drag force in a.u
    """
    if U.dtype == np.complex64:
        U = np.real(U)
    gradx = np.gradient(U, axis=-1)
    grady = np.gradient(U, axis=-2)
    fx = np.sum(-gradx*psi, axis=(-2, -1))
    fy = np.sum(-grady*psi, axis=(-2, -1))
    if psi.ndim == 3:
        f = np.zeros((psi.shape[0], 2))
        f[:, 0] = fx
        f[:, 1] = fy
        return f
    return (fx, fy)


def drag_force_cp(psi: cp.ndarray, U: cp.ndarray) -> tuple:
    """Computes the drag force considering an obstacle map U(r)
    and an intensity map I(r)

    Args:
        psi (cp.ndarray): Intensity map
        U (cp.ndarray): Potential map

    Returns:
        fx, fy (float): The drag force in a.u
    """
    if U.dtype == np.complex64:
        U = cp.real(U)
    gradx = cp.gradient(U, axis=-1)
    grady = cp.gradient(U, axis=-2)
    fx = cp.sum(-gradx*psi, axis=(-2, -1))
    fy = cp.sum(-grady*psi, axis=(-2, -1))
    if psi.ndim == 3:
        f = np.zeros((psi.shape[0], 2))
        f[:, 0] = fx.get()
        f[:, 1] = fy.get()
        return f
    else:
        return np.array([fx.get(), fy.get()])


def main():
    phase = np.loadtxt("v500_1_phase.txt")
    phase_cp = cp.asarray(phase)
    # Vortex detection step
    vortices_cp = vortex_detection_cp(phase_cp, r=1, plot=True)
    vortices = vortex_detection(phase, plot=True, r=1)
    bins = cp.linspace(0, phase.shape[0], 100)
    corr = pair_correlations_cp(vortices_cp, bins)
    plt.plot(bins[1:].get(), corr.get())
    plt.show()
    timer_repeat(vortex_detection, phase, N_repeat=25)
    timer_repeat(vortex_detection_cp, phase_cp, N_repeat=25)
    # Velocity decomposition in incompressible and compressible
    velo, v_inc, v_comp = helmholtz_decomp_cp(phase_cp, plot=True)
    velo, v_inc, v_comp = helmholtz_decomp(phase, plot=True)
    timer_repeat(helmholtz_decomp, phase, N_repeat=25)
    timer_repeat_cp(helmholtz_decomp_cp, phase_cp, N_repeat=25)
    # Clustering benchmarks
    dipoles, clusters, cluster_graph = cluster_vortices(vortices)
    timer_repeat(cluster_vortices, vortices, N_repeat=100)
    # Plot results
    fig, (ax, ax1) = plt.subplots(1, 2)
    YY, XX = np.indices(v_inc[0].shape)
    im = ax.imshow(np.hypot(v_inc[0], v_inc[1]), cmap='viridis')
    ax.streamplot(XX, YY, v_inc[0], v_inc[1],
                  density=5, color='white', linewidth=1)
    positions = dict(
        zip(cluster_graph.nodes, vortices[cluster_graph.nodes, 0:2]))
    edge_color = []
    for edge in cluster_graph.edges:
        if vortices[edge[0], 2] == 1:
            edge_color.append('r')
        else:
            edge_color.append('b')
    nx.draw(cluster_graph, ax=ax, pos=positions, with_labels=False,
            node_size=45, node_color=vortices[cluster_graph.nodes, 2], cmap='bwr', edge_color=edge_color, width=2)
    for dip in range(dipoles.shape[0]):
        ln_d, = ax.plot(vortices[dipoles[dip, :], 0],
                        vortices[dipoles[dip, :], 1], color='g', marker='o', label='Dipoles')
    ax.set_title(r'Incompressible velocity $|v^{inc}|$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim(0, phase.shape[1])
    ax.set_ylim(0, phase.shape[0])
    ax.legend(handles=[ln_d])
    plt.colorbar(im, ax=ax)
    bins, edges = cluster_histogram(clusters, plot=False)
    ax1.stairs(bins, edges, fill=True)
    ax1.set_xlabel("Size of cluster")
    ax1.set_ylabel("Number of vortices")
    ax1.set_xlim(1, np.max(edges))
    ax1.set_yscale("log")
    ax1.set_title("Histogram of cluster size")
    plt.show()


if __name__ == "__main__":
    main()
