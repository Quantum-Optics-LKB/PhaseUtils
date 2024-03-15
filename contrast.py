# -*-coding:utf-8 -*

import numpy as np
from functools import lru_cache
import pickle
import pyfftw
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage import gaussian_filter
from skimage.restoration import unwrap_phase
from skimage import filters, measure, morphology
from skimage.segmentation import clear_border, flood
from scipy import optimize
from scipy import constants as cst
from scipy import interpolate
import cupy as cp
from numbalsoda import lsoda_sig, lsoda
from simus.nlse import NLSE_1d
import numba
from numba import cuda
import cmath
import math
from typing import Any
import multiprocessing
pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()//2
# try to load previous fftw wisdom
try:
    with open("fft.wisdom", "rb") as file:
        wisdom = pickle.load(file)
        pyfftw.import_wisdom(wisdom)
except FileNotFoundError:
    print("No FFT wisdom found, starting over ...")


def gauss_fit(x, waist, mean):
    """Gaussian BEAM intensity fitting
    Attention !!! Different convention as for a regular gaussian

    Args:
        x (float): Position
        waist (float): Waist
        mean (float): center

    Returns:
        float: Gaussian
    """
    return np.exp(-2*(x-mean)**2/waist**2)


@numba.njit(parallel=True, cache=True, fastmath=True)
def az_avg(image: np.ndarray, center: tuple) -> np.ndarray:
    """Calculates the azimuthally averaged radial profile.

    Args:
        image (np.ndarray): The 2D image
        center (tuple, optional): The [x,y] pixel coordinates used as the center. Defaults to None,
        which then uses the center of the image (including fractional pixels).

    Returns:
        np.ndarray: prof the radially averaged profile
    """
    # Calculate the indices from the image
    max_r = max([np.hypot(center[0], center[1]),
                 np.hypot(center[0]-image.shape[1], center[1]),
                 np.hypot(center[0]-image.shape[1],
                          center[1]-image.shape[0]),
                 np.hypot(center[0], center[1]-image.shape[0])])
    r = np.arange(1, int(max_r)+1, 1)
    prof = np.zeros_like(r, dtype=np.float64)
    prof_counts = np.zeros_like(r)
    for i in numba.prange(image.shape[0]):
        for j in range(image.shape[1]):
            dist = round(np.hypot(i-center[1], j-center[0]))
            prof[dist] += image[i, j]
            prof_counts[dist] += 1
    prof /= prof_counts
    return prof


@cuda.jit(fastmath=True)
def _az_avg_cp(image: cp.ndarray, prof: cp.ndarray, prof_counts: cp.ndarray, center: tuple):
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


@numba.njit(numba.float32[:, :](numba.complex64[:, :]), fastmath=True, cache=True, parallel=True)
def angle_fast(x: np.ndarray) -> np.ndarray:
    """Accelerates a smidge angle by using fastmath

    Args:
        x (np.ndarray): The complex field

    Returns:
        np.ndarray: the argument of the complex field
    """
    out = np.empty_like(x, dtype=np.float32)
    for i in numba.prange(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j] = cmath.phase(x[i, j])
    return out


@cuda.jit((numba.complex64[:, :], numba.float32[:, :]), fastmath=True)
def angle_fast_cp(x: cp.ndarray, out: cp.ndarray):
    """Accelerates a smidge angle by using fastmath

    Args:
        x (np.ndarray): The complex field

    Returns:
        np.ndarray: the argument of the complex field
    """
    i, j = cuda.grid(2)
    if i < x.shape[0]:
        if j < x.shape[1]:
            out[i, j] = cmath.phase(x[i, j])

@numba.njit((numba.complex64[:, :], numba.complex64[:, :]), fastmath=True,
            nogil=True, cache=True, parallel=True)
def exp_angle_fast(x: np.ndarray, y: np.ndarray):
    """Fast multiplication by exp(-1j*x)

    Args:
        x (np.ndarray): The complex field
        y (np.ndarray): the field to multiply
    Returns:
        None
    """
    for i in numba.prange(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] *= cmath.exp(-1j*cmath.phase(y[i, j]))


@numba.njit((numba.complex64[:, :], numba.complex64), fastmath=True,
            nogil=True, cache=True, parallel=True)
def exp_angle_fast_scalar(x: np.ndarray, y: complex):
    """Fast multiplication by exp(-1j*y)

    Args:
        x (np.ndarray): The input array
        y (complex): the scalar to multiply
    Returns:
        None
    """
    for i in numba.prange(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] *= cmath.exp(-1j*cmath.phase(y))

@lru_cache(maxsize=10)
@numba.njit(cache=True, parallel=True, boundscheck=False)
def disk(m: int, n: int, center: tuple, radius: int):
    """Numba compatible mgrid in i,j indexing style

    Args:
        m (int) : size along i axis
        n (int) : size along j axis
    Returns:
        np.ndarray: xx, yy like numpy's meshgrid
    """
    out = np.zeros((m, n), dtype=np.uint8)
    for i in numba.prange(m):
        for j in numba.prange(n):
            r = (i-center[0])*(i-center[0])+(j-center[1])*(j-center[1])
            out[i, j] = r < radius*radius
    return out

def centre(im, truncate: bool = True):
    """Fits the center of the image using gaussian fitting

    Args:
        im (np.ndarray): The image to fit

    Returns:
        Tuple(int): The coordinates of the fitted center.
    """
    out_x = np.sum(im, axis=0)
    out_x = out_x/np.max(out_x)
    out_y = np.sum(im, axis=1)
    out_y = out_y/np.max(out_y)

    absc = np.linspace(0, im.shape[1]-1, im.shape[1])
    ordo = np.linspace(0, im.shape[0]-1, im.shape[0])
    p0x = np.argmax(out_x)
    p0y = np.argmax(out_y)
    ptot, pcov = optimize.curve_fit(gauss_fit, absc, out_x, p0=[
        p0x, len(absc)//2], maxfev=3200)
    centre_x = ptot[1]
    ptot, pcov = optimize.curve_fit(gauss_fit, ordo, out_y, p0=[
        p0y, len(ordo)//2], maxfev=3200)
    centre_y = ptot[1]
    if truncate:
        centre_x = int(centre_x)
        centre_y = int(centre_y)
    return centre_x, centre_y


def waist(im, plot=False):
    """Fits the waist of the image using gaussian fitting

    Args:
        im (np.ndarray): The image to fit

    Returns:
        Tuple(int): The coordinates of the fitted waists.
    """
    out_x = np.sum(im, axis=0)
    out_x = out_x/np.max(out_x)
    out_y = np.sum(im, axis=1)
    out_y = out_y/np.max(out_y)

    absc = np.linspace(0, im.shape[1]-1, im.shape[1])
    ordo = np.linspace(0, im.shape[0]-1, im.shape[0])
    poptx, pcov = optimize.curve_fit(gauss_fit, absc, out_x, p0=[
        100, len(absc)//2], maxfev=3200)
    waist_x = poptx[0]
    perrx = np.sqrt(np.diag(pcov))[0]
    popty, pcov = optimize.curve_fit(gauss_fit, ordo, out_y, p0=[
        100, len(ordo)//2], maxfev=3200)
    waist_y = popty[0]
    perry = np.sqrt(np.diag(pcov))[0]
    if plot:
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(absc, out_x)
        tex = r"$w_x$"
        pm = r'$\pm$'
        lab = f'{tex} = {waist_x:.1f} {pm} {perrx:.1f}'
        ax[0].plot(absc, gauss_fit(absc, *poptx), ls='--', label=lab)
        ax[1].plot(ordo, out_y)
        tex = r"$w_y$"
        lab = f'{tex} = {waist_y:.1f} {pm} {perry:.1f}'
        ax[1].plot(ordo, gauss_fit(ordo, *popty), ls='--', label=lab)
        ax[0].legend()
        ax[1].legend()
        plt.show(block=False)
    return waist_x, waist_y


def cache(radius: int, center: tuple = (1024, 1024), out: bool = True,
          nb_pix: tuple = (2048, 2048)) -> np.ndarray:
    """Defines a circular mask

    Args:
        radius (int): Radius of the mask
        center (tuple, optional): Center of the mask. Defaults to (1024, 1024).
        out (bool, optional): Masks the outside of the disk. Defaults to True.
        nb_pix (tuple, optional): Shape of the mask. Defaults to (2048, 2048).

    Returns:
        np.ndarray: The array of booleans defining the mask
    """
    Y, X = np.ogrid[:nb_pix[0], :nb_pix[1]]
    dist_from_center = np.hypot(X - center[0], Y-center[1])

    if out:
        mask = dist_from_center <= radius
    else:
        mask = dist_from_center > radius

    return mask


def cache_cp(radius: int, center: tuple = (1024, 1024), out: bool = True,
             nb_pix: tuple = (2048, 2048)) -> np.ndarray:
    """Defines a circular mask

    Args:
        radius (int): Radius of the mask
        center (tuple, optional): Center of the mask. Defaults to (1024, 1024).
        out (bool, optional): Masks the outside of the disk. Defaults to True.
        nb_pix (tuple, optional): Shape of the mask. Defaults to (2048, 2048).

    Returns:
        np.ndarray: The array of booleans defining the mask
    """
    Y, X = cp.ogrid[:nb_pix[0], :nb_pix[1]]
    dist_from_center = cp.hypot(X - center[0], Y-center[1])

    if out:
        mask = dist_from_center <= radius
    else:
        mask = dist_from_center > radius

    return mask


def im_osc(im: np.ndarray,  cont: bool = False, plot: bool = False, return_mask: bool = False, big: bool = False) -> tuple:
    """Separates the continuous and oscillating components of an image using
    Fourier filtering.

    :param np.ndarray im: Description of parameter `im`.
    :param bool cont: Returns or not the continuons component
    :param bool plot: Plots a visualization of the analysis result
    :return np.ndarray: The oscillating component of the image, or both
    components

    """
    im = im.astype(np.float32)
    im_fft = pyfftw.interfaces.numpy_fft.rfft2(im)
    im_fft_orig = im_fft.copy()
    im_fft_fringe = pyfftw.zeros_aligned(
        (im.shape[0], im.shape[1]), dtype=np.complex64)
    im_fft_cont = im_fft.copy()
    fft_filt = gaussian_filter(np.abs(im_fft), 1e-3*im_fft.shape[0])
    cont_size = im.shape[0]//4
    mask_cont = cache(cont_size, out=False, center=(0, 0),
                      nb_pix=im_fft_cont.shape)
    mask_cont = np.logical_xor(mask_cont, cache(cont_size, out=False,
                                                center=(
                                                    0, im_fft_cont.shape[0]),
                                                nb_pix=im_fft_cont.shape))
    im_fft_cont[np.logical_not(mask_cont)] = 0
    im_cont = pyfftw.interfaces.numpy_fft.irfft2(im_fft_cont)
    dbl_gradient = np.log(np.abs(np.gradient(fft_filt, axis=0)) +
                          np.abs(np.gradient(fft_filt, axis=1)))
    m_value = np.nanmean(dbl_gradient[dbl_gradient != -np.infty])
    dbl_gradient[mask_cont] = m_value
    dbl_gradient_int = (
        2**16*(dbl_gradient/np.nanmax(dbl_gradient))).astype(np.uint16)
    threshold = filters.threshold_otsu(dbl_gradient_int)
    mask = dbl_gradient_int > threshold
    mask = morphology.remove_small_objects(mask, 1)
    mask = morphology.remove_small_holes(mask, 1)
    mask = clear_border(mask)
    mask = morphology.remove_small_holes(mask, 1, connectivity=2)
    labels = measure.label(mask)
    props = measure.regionprops(labels, dbl_gradient_int)
    # takes the spot with the maximum area
    areas = [prop.area for prop in props]
    maxi_area = np.where(areas == max(areas))[0][0]
    label_osc = props[maxi_area].label
    center_osc = np.round(props[maxi_area].centroid).astype(int)
    contour_osc = measure.find_contours(labels == label_osc, 0.5)[0]
    y, x = contour_osc.T
    y = y.astype(int)
    x = x.astype(int)
    mask_osc = np.zeros(im_fft.shape)
    mask_osc[y, x] = 1
    mask_osc_flood = flood(mask_osc, (y[0]+1, x[0]+1), connectivity=1)
    if big:
        # r_osc = min(center_osc)
        r_osc = 1.9*np.max([[np.hypot(x[i]-x[j], y[i]-y[j])
                             for j in range(len(x))] for i in range(len(x))])
        mask_osc_flood = cache(r_osc, out=False, center=(
            center_osc[1], center_osc[0]), nb_pix=im_fft.shape)
    im_fft[mask_osc_flood] = 0

    # bring osc part to center to remove tilt
    im_fft = np.roll(im_fft,
                     (im_fft.shape[0]//2-center_osc[0],
                      im_fft.shape[1]//2-center_osc[1]),
                     axis=(-2, -1))
    im_fft_fringe[:, im_fft.shape[1] //
                  2:im_fft_fringe.shape[1]//2+im_fft.shape[1]//2+1] = im_fft
    im_fringe = pyfftw.interfaces.numpy_fft.ifft2(
        np.fft.fftshift(im_fft_fringe), s=im.shape, axes=(-1, -2))
    # save FFT wisdom
    with open("fft.wisdom", "wb") as file:
        wisdom = pyfftw.export_wisdom()
        pickle.dump(wisdom, file)
    if plot:
        fig, ax = plt.subplots(1, 4)
        im0 = ax[0].imshow(im, cmap='gray')
        ax[0].set_title("Real space")
        fig.colorbar(im0, ax=ax[0])
        im = ax[1].imshow(np.log10(np.abs(im_fft_orig)+1e-15))
        fig.colorbar(im, ax=ax[1])
        ax[1].plot(x, y, color='r', ls='--')
        if big:
            circle_big = plt.Circle((center_osc[1], center_osc[0]), r_osc, color='r',
                                    fill=False)
            ax[1].add_patch(circle_big)

        ax[1].set_title("Fourier space")
        ax[1].legend(["Oscillating", "Continuous"])
        im = ax[2].imshow(np.log10(np.abs(im_fft)+1e-15))
        fig.colorbar(im, ax=ax[2])
        ax[2].set_title("Filtered Fourier signal")
        im = ax[3].imshow(np.angle(im_fringe), cmap="twilight")
        fig.colorbar(im, ax=ax[3])
        ax[3].set_title("Phase of filtered signal")
        plt.show()
    if cont:
        if return_mask:
            return im_cont, im_fringe, mask_cont_flood, mask_osc_flood, center_osc
        return im_cont, im_fringe
    if return_mask:
        return im_fringe, mask_cont_flood, mask_osc_flood, center_osc
    return im_fringe


def im_osc_center(im: np.ndarray, center: tuple, mask_osc_flood: np.ndarray = None,  cont: bool = False, plot: bool = False, big: bool = False) -> tuple:
    """Separates the continuous and oscillating components of an image using
    Fourier filtering.

    :param np.ndarray im: Description of parameter `im`.
    :param tuple center: i,j position of the 1st order
    :param np.ndarray mask_osc_flood: mask for the 1st order
    :param bool cont: Returns or not the continuons component
    :param bool plot: Plots a visualization of the analysis result
    :return np.ndarray: The oscillating component of the image, or both
    components

    """

    im_fft = np.fft.fftshift(pyfftw.interfaces.numpy_fft.fft2(im))
    im_fft_fringe = im_fft.copy()
    im_fft_cont = im_fft.copy()
    fft_filt = gaussian_filter(np.abs(im_fft), 1e-3*im_fft.shape[0])
    cont_size = 20
    mask_cont_flood = cache(cont_size, out=False, center=(im.shape[0]//2, im.shape[1]//2),
                            nb_pix=im.shape)
    if mask_osc_flood is None:
        dbl_gradient = np.log(np.abs(np.gradient(fft_filt, axis=0)) +
                              np.abs(np.gradient(fft_filt, axis=1)))
        m_value = np.nanmean(dbl_gradient[dbl_gradient != -np.infty])
        dbl_gradient[np.bitwise_not(mask_cont_flood)] = m_value
        dbl_gradient_int = (dbl_gradient*(dbl_gradient > 0.8 *
                                          np.nanmax(dbl_gradient)))
        dbl_gradient_int /= np.nanmax(dbl_gradient_int)
        dbl_gradient_int = (255*dbl_gradient_int).astype(np.uint8)
        threshold = filters.threshold_otsu(dbl_gradient_int)
        mask = dbl_gradient_int > threshold
        mask = morphology.remove_small_objects(mask, 1)
        mask = morphology.remove_small_holes(mask, 1)
        mask = clear_border(mask)
        mask = morphology.remove_small_holes(mask, 1, connectivity=2)
        labels = measure.label(mask)
        props = measure.regionprops(labels, dbl_gradient_int)
        # takes the spot with the maximum area
        areas = [prop.area for prop in props]
        maxi_area = np.where(areas == max(areas))[0][0]
        label_osc = props[maxi_area].label
        center_osc = center
        contour_osc = measure.find_contours(labels == label_osc, 0.5)[0]
        y, x = contour_osc.T
        y = y.astype(int)
        x = x.astype(int)
        mask_osc = np.zeros(im_fft.shape)
        mask_osc[y, x] = 1
        mask_osc_flood = flood(mask_osc, (y[0]+1, x[0]+1), connectivity=1)
        if big:
            r_osc = np.max([[np.hypot(x[i]-x[j], y[i]-y[j])
                             for j in range(len(x))] for i in range(len(x))])
            mask_osc_flood = cache(r_osc, out=False, center=(
                center, center))
            # mask_osc_flood = np.zeros(mask_cont_flood.shape, dtype=bool)
            # mask_osc_flood[0:mask_osc_flood.shape[0] //
            #                2, 0:mask_osc_flood.shape[1]//2] = True
            # mask_osc_flood = np.logical_not(np.logical_and(
            #     mask_osc_flood, mask_cont_flood))
    im_fft_fringe[mask_osc_flood] = 0
    im_fft_cont[mask_cont_flood] = 0
    # bring osc part to center to remove tilt
    im_fft_fringe = np.roll(im_fft_fringe,
                            (im_fft_fringe.shape[0]//2-center[0],
                             im_fft_fringe.shape[1]//2-center[1]),
                            axis=(-2, -1))
    im_fringe = pyfftw.interfaces.numpy_fft.ifft2(
        np.fft.fftshift(im_fft_fringe))
    im_cont = pyfftw.interfaces.numpy_fft.ifft2(np.fft.fftshift(im_fft_cont))
    # save FFT wisdom
    with open("fft.wisdom", "wb") as file:
        wisdom = pyfftw.export_wisdom()
        pickle.dump(wisdom, file)
    if plot:
        circle = plt.Circle((im.shape[1]//2, im.shape[0]//2), cont_size//2, color='b',
                            fill=False)
        fig, ax = plt.subplots(1, 4)
        im0 = ax[0].imshow(im, cmap='gray')
        ax[0].set_title("Real space")
        fig.colorbar(im0, ax=ax[0])
        im = ax[1].imshow(np.abs(im_fft),
                          norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                                 vmin=np.min(np.abs(im_fft)),
                                                 vmax=np.max(np.abs(im_fft)),
                                                 base=10))
        fig.colorbar(im, ax=ax[1])
        if mask_osc_flood is None:
            ax[1].plot(x, y, color='r', ls='--')
        else:
            ax[1].imshow(mask_osc_flood, alpha=0.35, cmap='gray')
        ax[1].add_patch(circle)
        if big and mask_osc_flood is None:
            circle_big = plt.Circle((center[1], center[0]), r_osc, color='r',
                                    fill=False)
            ax[1].add_patch(circle_big)

        ax[1].set_title("Fourier space")
        ax[1].legend(["Oscillating", "Continuous"])
        im = ax[2].imshow(np.abs(im_fft_fringe),
                          norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                                 vmin=np.min(np.abs(im_fft)),
                                                 vmax=np.max(np.abs(im_fft)),
                                                 base=10))
        fig.colorbar(im, ax=ax[2])
        ax[2].set_title("Filtered Fourier signal")
        im = ax[3].imshow(np.angle(im_fringe), cmap="twilight")
        fig.colorbar(im, ax=ax[3])
        ax[3].set_title("Phase of filtered signal")
        plt.show()
    if cont:
        return im_cont, im_fringe
    return im_fringe


def im_osc_mask(im: np.ndarray, masks: tuple,  cont: bool = True, plot: bool = False) -> tuple:
    """Separates the continuous and oscillating components of an image using
    Fourier filtering.

    :param np.ndarray im: Description of parameter `im`.
    :param tuple masks: Continuous and oscillating masks
    :param bool cont: Returns or not the continuons component
    :param bool plot: Plots a visualization of the analysis result
    :return np.ndarray: The oscillating component of the image, or both
    components

    """
    mask_cont_flood, mask_osc_flood, center_osc = masks
    center_osc[0] = int(center_osc[0])
    center_osc[1] = int(center_osc[1])
    im_fft = np.fft.fftshift(pyfftw.interfaces.numpy_fft.fft2(im))
    im_fft_fringe = im_fft.copy()
    im_fft_cont = im_fft.copy()
    im_fft_fringe[mask_osc_flood] = 0
    im_fft_cont[mask_cont_flood] = 0
    # bring osc part to center to remove tilt
    im_fft_fringe = np.roll(im_fft_fringe,
                            (im_fft_fringe.shape[0]//2-center_osc[0],
                             im_fft_fringe.shape[1]//2-center_osc[1]),
                            axis=(-2, -1))
    im_fringe = pyfftw.interfaces.numpy_fft.ifft2(
        np.fft.fftshift(im_fft_fringe))
    im_cont = pyfftw.interfaces.numpy_fft.ifft2(np.fft.fftshift(im_fft_cont))
    # save FFT wisdom
    with open("fft.wisdom", "wb") as file:
        wisdom = pyfftw.export_wisdom()
        pickle.dump(wisdom, file)
    if plot:
        fig, ax = plt.subplots(1, 4)
        im0 = ax[0].imshow(im, cmap='gray')
        ax[0].set_title("Real space")
        fig.colorbar(im0, ax=ax[0])
        im = ax[1].imshow(np.abs(im_fft),
                          norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                                 vmin=np.min(np.abs(im_fft)),
                                                 vmax=np.max(np.abs(im_fft)),
                                                 base=10))
        ax[1].scatter(center_osc[1], center_osc[0], color='red')
        fig.colorbar(im, ax=ax[1])
        ax[1].set_title("Fourier space")
        im = ax[2].imshow(np.abs(im_fft_fringe),
                          norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                                 vmin=np.min(np.abs(im_fft)),
                                                 vmax=np.max(np.abs(im_fft)),
                                                 base=10))
        fig.colorbar(im, ax=ax[2])
        ax[2].set_title("Filtered Fourier signal")
        im = ax[3].imshow(np.angle(im_fringe), cmap="twilight")
        fig.colorbar(im, ax=ax[3])
        ax[3].set_title("Phase of filtered signal")
        plt.show()
    if cont:
        return im_cont, im_fringe
    return im_fringe

 # objective function for Dn vs I fitting


@numba.cfunc(lsoda_sig, cache=True)
def dI_dz(z: float, I: np.ndarray, dI: np.ndarray, p: np.ndarray) -> None:
    """RHS to solve the intensity evolution

    Args:
        z (float): Position in the cell
        I (np.ndarray): Intensity
        dI (np.ndarray): Intensity derivative
        p (np.ndarray): Parameters
    """
    alpha = p[0]
    Isat = p[1]
    dI[0] = -alpha*I[0]/(1+I[0]/Isat)


@numba.cfunc(lsoda_sig, cache=True)
def dphi_dz(z: float, y: np.ndarray, dy: np.ndarray, p: np.ndarray) -> None:
    """RHS to solve the non linear phase evolution

    Args:
        z (float): Position in the cell
        y (np.ndarray): The state vector (I, phi)
        dy (np.ndarray): The derivative of the state vector (dI, dphi)
        p (np.ndarray): The parameters 
    """
    I = y[0]
    k = p[0]
    alpha = p[1]
    n2 = p[2]
    Isat = p[3]
    dy[0] = -alpha*I/(1+I/Isat)
    dy[1] = k*n2*I/(1+I/Isat)


dI_dz_ptr = dI_dz.address
dphi_dz_ptr = dphi_dz.address


@numba.njit(parallel=True, fastmath=True)
def I_z(z: float, I0: np.ndarray, alpha: float, Isat: float) -> np.ndarray:
    """Returns the intensity after a propagation of z in the cell

    Args:
        z (float): Position in the cell
        I0 (np.ndarray): Initial intensity
        alpha (float): Linear losses coeff
        Isat (float): Saturation intensity in W/m^2

    Returns:
        np.ndarray: The final intensity
    """
    Iz = np.empty_like(I0)
    t_eval = np.array([0, z], dtype=np.float64)
    p = np.array([alpha, Isat], dtype=np.float64)
    for i in numba.prange(I0.shape[0]):
        usol, success = lsoda(
            dI_dz_ptr, np.array([I0[i]], dtype=np.float64), t_eval, rtol=1e-6, atol=1e-6, data=p)
        Iz[i] = usol[-1, 0]
    return Iz


@numba.njit(parallel=True, fastmath=True)
def phi_z(z: float, I0: np.ndarray, k: float, alpha: float, n2: float, Isat: float) -> np.ndarray:
    """Returns the nonlinear dephasing after the a length z

    Args:
        z (float): The length in m
        I0 (np.ndarray): Initial intensity profile in W/m^2
        k (float): Wavenumber in m^-1
        alpha (float): Linear losses coeff in m^-1
        n2 (float): Non linear coeff in m^2/W
        Isat (float): Saturation intensity in W/m^2

    Returns:
        np.ndarray: Final non-linear phase profile
    """
    phi = np.empty_like(I0)
    t_eval = np.array([0, z], dtype=np.float64)
    p = np.array([k, alpha, n2, Isat], dtype=np.float64)
    for i in numba.prange(I0.shape[0]):
        usol, success = lsoda(
            dphi_dz_ptr, np.array([I0[i], 0], dtype=np.float64), t_eval, rtol=1e-6, atol=1e-6, data=p)
        phi[i] = usol[-1, 1]
    return phi


def delta_n(im0: np.ndarray, I0: float, Pf: float, w0: float, d: float,
            k: float, L: float, alpha: float = 50,  plot: bool = False, err: bool = False):
    """Computes the total dephasing of an interferogram and fits the linear 
    loss coefficient alpha, the nonlinear coefficient n2 and the saturation intensity
    from a single interferogram.

    :param np.ndarray im0: Image to extract Dn
    :param float I0: Initial intensity
    :param float Pf: Final power
    :param w0: initial waist
    :param float d: Pixel pitch of the image
    :param float k: wavenumber
    :param float L: length of the cell in m
    :param bool plot: Plots a visualization of the analysis result
    :param bool err: Returns the error
    :return tuple: phi_tot, (n2, Isat, alpha) with the errors if err is True.
    """
    # im = im/np.max(im)
    im_fringe = im_osc_fast_t(im0, cont=False)
    # ATTENTION : Because im_osc_fast_t truncates the image,
    # d needs to become 2d in the rest of the function
    im_cont = np.abs(im_fringe)**2
    im_cont /= np.max(im_cont)
    wx, wy = waist(im_cont, plot=False)
    wx *= 2*d
    wy *= 2*d
    # ratio of camera sensor surface over whole beam if waist is bigger than the whole camera
    If = Pf/(np.sum(im_cont)*(2*d)**2) * \
        (np.pi*(wx**2+wy**2))/(np.prod(im_cont.shape)*(2*d)**2)
    # fit Isat

    def fit_function_Isat(I, alpha, Isat):
        return I_z(L, I, alpha, Isat)
    phase_raw = angle_fast(im_fringe)
    im_cont *= If
    centre_x, centre_y = centre(im_cont)
    cont_avg = az_avg(im_cont, center=(centre_x, centre_y))
    phase = unwrap_phase(phase_raw, wrap_around=False)
    phi_avg = az_avg(phase, center=(centre_x, centre_y))
    phi_avg = gaussian_filter(phi_avg, 25)
    cont_avg = gaussian_filter(cont_avg, 25)
    phi_avg -= np.max(phi_avg)
    # fit input intensity using waist
    x = np.linspace(0, len(cont_avg)-1, len(cont_avg))*2*d
    selec = x < max(im_cont.shape)*d
    cont_fit = cont_avg[selec]
    phi_fit = phi_avg[selec]
    phi_fit -= np.max(phi_fit)
    x = x[selec]
    dphi = abs(np.max(phi_fit)-np.min(phi_fit))
    dn_guess = dphi/(k*L)
    n2_guess = dn_guess/I0
    I_in = I0*np.exp(-2*x**2/w0**2)
    (alpha, Isat), cov = optimize.curve_fit(fit_function_Isat, I_in,
                                            cont_fit, p0=(alpha, 1e4), bounds=[(0, 1e2), (-np.log(1e-9)/L, 5e6)],
                                            maxfev=3200)
    alpha_err, Isat_err = np.sqrt(np.diag(cov))

    def fit_phi_vs_I(I: np.ndarray, n2: float):
        return phi_z(L, I, k, alpha, n2, Isat)-phi_z(L, np.array([np.min(I)]), k, alpha, n2, Isat)
    (n2,), pcov = optimize.curve_fit(fit_phi_vs_I, I_in,
                                     phi_fit,
                                     bounds=[(-1e-6,),
                                             (0,)],
                                     p0=(-n2_guess),
                                     maxfev=3200)
    # gets fitting covariance/error for each parameter
    n2_err = np.sqrt(np.diag(pcov))[0]
    phase_tot = np.abs(phi_z(L, np.array([np.max(I_in)]), k, alpha,
                       n2, Isat)-phi_z(L, np.array([1e-10]), k, alpha, n2, Isat))
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(2, 3, 1)
        i00 = ax.imshow(im_cont*1e-4)
        ax.set_title("Dc intensity")
        fig.colorbar(i00, ax=ax, label=r'Intensity ($W/cm^2$)')
        ax = fig.add_subplot(2, 3, 3)
        i01 = ax.imshow(phase_raw, cmap='twilight_shifted')
        ax.set_title("Phase")
        fig.colorbar(i01, ax=ax, label=r'$\phi$ (rad)')
        ax = fig.add_subplot(2, 3, 4)
        i10 = ax.imshow(phase, cmap="viridis")
        ax.set_title(r"Unwrapped $\phi$")
        fig.colorbar(i10, ax=ax, label='Phase (rad)')
        ax = fig.add_subplot(2, 3, 5)
        lab = f"n2 = {n2:.2e} +/- {n2_err:.2e}\n"
        lab += f"alpha = {alpha:.2f} +/- {alpha_err:.2f}\n"
        lab += f"Isat = {Isat*1e-4:.2f} +/- {Isat_err*1e-4:.2f} W/cm²\n"
        ax.plot(x*1e3, fit_phi_vs_I(I_in, n2),
                label=lab)
        ax.plot(x*1e3, phi_fit, label=r"$\phi_{fit}$")
        ax.set_title("Azimuthal average")
        ax.set_xlabel(r"Position in mm")
        ax.set_ylabel(r"$\phi$ in rad")
        ax.legend()
        ax = fig.add_subplot(2, 3, 6)
        ax.plot(I_in*1e-4, cont_fit*1e-4, label=r'$I_{out}$')
        lab = f"alpha = {alpha:.2f} +/- {alpha_err:.2f}\n"
        lab += f"Isat = {Isat*1e-4:.2f} +/- {Isat_err*1e-4:.2f} W/cm²\n"
        ax.plot(I_in*1e-4, fit_function_Isat(I_in, alpha, Isat)*1e-4, label=lab)
        ax.set_title(r"$I_{sat}$ and $\alpha$ fit")
        ax.set_xlabel("Fitted input intensity in $W/cm^2$")
        ax.set_ylabel("Output intensity in $W/cm^2$")
        ax.legend()
        plt.show()
        # plt.tight_layout()
        plt.show()
    if err:
        return phase_tot, (n2, Isat, alpha), (n2_err, Isat_err, alpha_err)
    else:
        return phase_tot, (n2, Isat, alpha)


def delta_n_sim(im0: np.ndarray, Pi: float, Pf: float, w0: float, d: float,
                k: float, L: float, alpha: float = 50,  plot: bool = False):
    """Computes the total dephasing of an interferogram and fits the linear 
    loss coefficient alpha, the nonlinear coefficient n2 and the saturation intensity
    from a single interferogram.

    :param np.ndarray im0: Image to extract Dn
    :param float I0: Initial intensity
    :param float Pf: Final power
    :param float d: Pixel pitch of the image
    :param float k: wavenumber
    :param bool plot: Plots a visualization of the analysis result
    :param bool err: Returns the error
    :return tuple: phi_tot, (n2, Isat, alpha) with the errors if err is True.
    """
    I0 = Pi/(np.pi*w0**2)
    # im = im/np.max(im)
    im_fringe = im_osc_fast_t(im0, cont=False)
    # ATTENTION : Because im_osc_fast_t truncates the image,
    # d needs to become 2d in the rest of the function
    im_cont = np.abs(im_fringe)**2
    im_cont /= np.max(im_cont)
    wx, wy = waist(im_cont, plot=False)
    wx *= 2*d
    wy *= 2*d
    # ratio of camera sensor surface over whole beam if waist is bigger than the whole camera
    If = Pf/(np.sum(im_cont)*(2*d)**2) * \
        (np.pi*(wx**2+wy**2))/(np.prod(im_cont.shape)*(2*d)**2)
    phase_raw = angle_fast(im_fringe)
    im_cont *= If
    centre_x, centre_y = centre(im_cont)
    cont_avg = az_avg(im_cont, center=(centre_x, centre_y))
    phase = unwrap_phase(phase_raw, wrap_around=False)
    phi_avg = az_avg(phase, center=(centre_x, centre_y))
    phi_avg = gaussian_filter(phi_avg, 25)
    cont_avg = gaussian_filter(cont_avg, 25)
    phi_avg -= np.max(phi_avg)
    x = np.linspace(0, len(cont_avg)-1, len(cont_avg))*2*d
    selec = x < max(im_cont.shape)*d
    cont_fit = np.zeros(2*np.sum(selec))
    phi_fit = np.zeros(2*np.sum(selec))
    x = np.linspace(-len(cont_fit)//2, len(cont_fit)//2, len(cont_fit))*2*d
    window = 2*d*len(cont_fit)
    cont_fit[:len(cont_fit)//2] = np.flip(cont_avg[selec])
    phi_fit[:len(cont_fit)//2] = np.flip(phi_avg[selec])
    cont_fit[len(cont_fit)//2:] = cont_avg[selec]
    phi_fit[len(cont_fit)//2:] = phi_avg[selec]
    phi_fit -= np.max(phi_fit)
    dphi = abs(np.max(phi_fit)-np.min(phi_fit))
    dn_guess = dphi/(k*L)
    n2_guess = dn_guess/I0
    Isat_guess = 1e5
    alpha_guess = -np.log(Pf/Pi)/L
    # define simulation for fitting
    N = 512
    simu = NLSE_1d(alpha_guess, Pi, w0, window, n2_guess, None,
                   L, NX=N)
    simu.I_sat = Isat_guess
    simu.delta_z = L/500
    E_in_0 = cp.ones((simu.NX), dtype=cp.complex64) * \
        cp.exp(-2*(cp.asarray(simu.X)**2)/simu.waist**2)
    cont_interp_func = interpolate.interp1d(x, cont_fit)
    phi_interp_func = interpolate.interp1d(x, phi_fit)
    cont_interp = cont_interp_func(simu.X)
    phi_interp = phi_interp_func(simu.X)
    # define fitting functions

    def fit_function(x, n2, Isat, alpha):
        simu.n2 = n2
        simu.I_sat = Isat
        simu.alpha = alpha
        simu.delta_z = L/500
        out = simu.out_field(E_in_0, L, plot=False, verbose=False)
        out_a = (cp.abs(out)**2*cst.c*cst.epsilon_0/2).get()
        out_p = cp.unwrap(cp.angle(out)).get()
        return np.hstack([out_a, out_p])

    def error_function(params: tuple, x: np.ndarray, y: np.ndarray) -> float:
        return np.sum((y-fit_function(x, *params))**2)

    bds_ = ((-2e-8, -1e-10), (1e3, 2e5), (1, 50))
    init_params = optimize.brute(error_function, bds_, args=(
        simu.X, np.hstack([cont_interp, phi_interp])), Ns=20, disp=True)
    A_fit = fit_function(simu.X, *init_params)
    A_fit = A_fit[0:A_fit.shape[0]//2]*np.exp(1j*A_fit[A_fit.shape[0]//2:])
    n2, Isat, alpha_fit = init_params
    print(init_params)
    if plot:
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(x, phi_fit, label=r'$\phi$')
        ax[0].plot(simu.X, phi_interp, label=r'$\phi$ interp')
        ax[0].plot(simu.X, np.unwrap(np.angle(A_fit)), label='Fit')
        ax[1].plot(x, cont_fit, label='Amp')
        ax[1].plot(simu.X, cont_interp, label='Interpolated')
        ax[1].plot(simu.X, np.abs(A_fit), label='Fit')
        ax[0].legend()
        ax[1].legend()
        plt.show()
        # fig = plt.figure()
        # ax = fig.add_subplot(2, 3, 1)
        # i00 = ax.imshow(im_cont*1e-4)
        # ax.set_title("Dc intensity")
        # fig.colorbar(i00, ax=ax, label=r'Intensity ($W/cm^2$)')
        # ax = fig.add_subplot(2, 3, 3)
        # i01 = ax.imshow(phase_raw, cmap='twilight_shifted')
        # ax.set_title("Phase")
        # fig.colorbar(i01, ax=ax, label=r'$\phi$ (rad)')
        # ax = fig.add_subplot(2, 3, 4)
        # i10 = ax.imshow(phase, cmap="viridis")
        # ax.set_title(r"Unwrapped $\phi$")
        # fig.colorbar(i10, ax=ax, label='Phase (rad)')
        # ax = fig.add_subplot(2, 3, 5)
        # lab = f"n2 = {n2:.2e} +/- {n2_err:.2e}\n"
        # lab += f"alpha = {alpha:.2f} +/- {alpha_err:.2f}\n"
        # lab += f"Isat = {Isat*1e-4:.2f} +/- {Isat_err*1e-4:.2f} W/cm²\n"
        # ax.plot(simu.X*1e3, np.unwrap(np.angle(A_fit)),
        #         label=lab)
        # ax.plot(x*1e3, phi_fit, label=r"$\phi_{fit}$")
        # ax.set_title("Azimuthal average")
        # ax.set_xlabel(r"Position in mm")
        # ax.set_ylabel(r"$\phi$ in rad")
        # ax.legend()
        # ax = fig.add_subplot(2, 3, 6)
        # ax.plot(x*1e3, cont_fit*1e-4, label=r'$I_{out}$')
        # lab = f"alpha = {alpha:.2f} +/- {alpha_err:.2f}\n"
        # lab += f"Isat = {Isat*1e-4:.2f} +/- {Isat_err*1e-4:.2f} W/cm²\n"
        # ax.plot(simu.X*1e3, np.abs(A_fit)**2 * 1e-4, label=lab)
        # ax.set_title(r"$I_{sat}$ and $\alpha$ fit")
        # ax.set_xlabel(r"Position in mm")
        # ax.set_ylabel("Output intensity in $W/cm^2$")
        # ax.legend()
        # plt.show()
        # # plt.tight_layout()
        # plt.show()
    else:
        return (n2, Isat, alpha)


def delta_n_cp(im0: cp.ndarray, I0: float, Pf: float, w0: float, d: float,
               k: float, L: float, alpha: float = 50,  plot: bool = False, err: bool = False):
    """Computes the total dephasing of an interferogram and fits the linear 
    loss coefficient alpha, the nonlinear coefficient n2 and the saturation intensity
    from a single interferogram.

    :param np.ndarray im0: Image to extract Dn
    :param float I0: Initial intensity
    :param float Pf: Final power
    :param float d: Pixel pitch of the image
    :param float k: wavenumber
    :param bool plot: Plots a visualization of the analysis result
    :param bool err: Returns the error
    :return tuple: phi_tot, (n2, Isat, alpha) with the errors if err is True.
    """
    im = cp.copy(im0)
    im = im/cp.max(im)
    im_fringe = im_osc_fast_cp(im, cont=False)
    im_cont = cp.abs(im_fringe)
    im_cont /= cp.max(im_cont)
    # ratio of camera sensor surface over whole beam if waist is bigger than the whole camera
    If = Pf/(cp.sum(im_cont)*d**2) * \
        (np.pi*w0**2)/(np.prod(im0.shape)*d**2)
    # fit Isat

    def fit_function_Isat(I, alpha, Isat):
        return I_z(L, I, alpha, Isat)
    phase = cp.empty_like(im_fringe, dtype=cp.float32)
    tpb = 16
    bpg = math.ceil(phase.shape[0]/tpb)
    angle_fast_cp[(bpg, bpg), (tpb, tpb)](im_fringe, phase)
    im_cont *= If
    im_cont = im_cont.get()
    phase = phase.get()
    centre_x, centre_y = centre(im_cont)
    cont_avg = az_avg(im_cont, center=(centre_x, centre_y))
    phase = unwrap_phase(phase, wrap_around=True)
    phi_avg = az_avg(phase, center=(centre_x, centre_y))
    phi_avg = gaussian_filter(phi_avg, 50)
    phi_avg = -phi_avg
    phi_avg -= np.max(phi_avg)
    cont_fit = cont_avg[np.linspace(
        0, len(cont_avg)-1, 100, dtype=int)]
    phi_fit = phi_avg[np.linspace(
        0, len(phi_avg)-1, 100, dtype=int)]
    dphi = abs(np.max(phi_fit)-np.min(phi_fit))
    dn_guess = dphi/(k*L)
    n2_guess = dn_guess/I0
    # fit input intensity using waist
    x = np.linspace(0, len(cont_avg)-1, len(cont_avg))*d
    x = x[np.linspace(0, len(x)-1, 100, dtype=int)]
    I_in = I0*np.exp(-x**2/w0**2)
    (alpha, Isat), cov = optimize.curve_fit(fit_function_Isat, I_in,
                                            cont_fit, p0=(alpha, 1e4), bounds=[(0, 1e2), (-np.log(1e-9)/L, 1e7)])
    alpha_err, Isat_err = np.sqrt(np.diag(cov))

    def fit_phi_vs_I(I: np.ndarray, n2):
        return phi_z(L, I, k, alpha, n2, Isat)
    (n2,), pcov = optimize.curve_fit(fit_phi_vs_I, I_in,
                                     phi_fit,
                                     bounds=[(-1e-6,),
                                             (0,)],
                                     p0=(-n2_guess),
                                     maxfev=3200)
    # gets fitting covariance/error for each parameter
    n2_err = np.sqrt(np.diag(pcov))[0]
    phase_tot = np.abs(phi_z(L, np.array([np.max(I_in)]), k, alpha,
                       n2, Isat)-phi_z(L, np.array([1e-10]), k, alpha, n2, Isat))
    if plot:
        fig, ax = plt.subplots(1, 3)
        i0 = ax[0].imshow(np.abs(im_cont))
        ax[0].set_title("Dc intensity")
        fig.colorbar(i0, ax=ax[0])
        i1 = ax[1].imshow(phase, cmap='viridis')
        ax[1].set_title("Unwrapped phase")
        fig.colorbar(i1, ax=ax[1])
        ax[2].plot(cont_avg*1e-4, phi_avg, label="Unwrapped phase")
        lab = f"n2 = {n2:.2e} +/- {n2_err:.2e}\n"
        lab += f"alpha = {alpha:.2f} +/- {alpha_err:.2f}\n"
        lab += f"Isat = {Isat*1e-4:.2f} +/- {Isat_err*1e-4:.2f} W/cm²\n"
        ax[2].plot(I_z(L, I_in, alpha, Isat)*1e-4, fit_phi_vs_I(I_in, n2),
                   label=lab)
        ax[2].plot(cont_avg*1e-4, phi_avg,
                   label="Unwrapped phase filtered")
        ax[2].set_title("Azimuthal average")
        ax[2].set_xlabel(r"Az avg output intensity $W/cm^2$")
        ax[2].set_ylabel("Phase in rad")
        ax[2].legend()
        # plt.tight_layout()
        plt.show()
    if err:
        return phase_tot, (n2, Isat, alpha), (n2_err, Isat_err, alpha_err)
    else:
        return phase_tot, (n2, Isat, alpha)


def contr(im: np.ndarray) -> np.ndarray:
    """Computes the contrast of an interferogram

    Args:
        im (np.ndarray): The interferogram

    Returns:
        np.ndarray: The contrast map
    """
    im_cont, im_fringe = im_osc(im, cont=True)
    analytic = np.abs(im_fringe)
    cont = np.abs(im_cont)
    return 2*analytic/cont


def phase(im: np.ndarray, plot: bool = False, masks: tuple = None, big: bool = False) -> np.ndarray:
    """Returns the phase from an interfogram

    Args:
        im (np.ndarray): The interferogram
        plot (bool) : whether to plot something

    Returns:
        np.ndarray: The unwrapped phase
    """
    if masks is not None:
        im_fringe = im_osc_mask(im, masks, cont=False, plot=plot, big=big)
    else:
        im_fringe = im_osc(im, cont=False, plot=plot, big=big)
    im_phase = unwrap_phase(np.angle(im_fringe))

    return im_phase


def phase_center(im: np.ndarray, center: tuple, mask_osc_flood: np.ndarray = None, plot: bool = False, masks: tuple = None, big: bool = False, unwrap=True) -> np.ndarray:
    """Returns the phase from an interfogram

    Args:
        im (np.ndarray): The interferogram
        plot (bool) : whether to plot something

    Returns:
        np.ndarray: The unwrapped phase
    """
    im_fringe = im_osc_center(
        im, center, mask_osc_flood=mask_osc_flood, cont=False, plot=plot, big=big)
    if unwrap:
        return unwrap_phase(np.angle(im_fringe))
    return np.angle(im_fringe)


def im_osc_fast(
    im: np.ndarray, radius: int = 0, cont: bool = False, plans: Any = None
) -> np.ndarray:
    """Return the field.

    Fast field recovery assuming ideal reference angle i.e minimum fringe
    size of sqrt(2) pixels.

    Args:
        im (cp.ndarray): Interferogram
        radius (int, optional): Radius of filter in px. Defaults to 512.
        return_cont (bool, optionnal): Returns the continuous part of the
        field.
        Defaults to False.
        plans (FFTW plan list, optionnal): [plan_fft, plan_ifft] for optional
        plan caching

    Returns:
        np.ndarray: Recovered field
    """
    if plans is not None:
        plan_fft, plan_ifft = plans
    if radius == 0:
        radius = min(im.shape[-2:]) // 4
    center = (im.shape[-2] // 4, im.shape[-1] // 4)
    im_ifft = np.empty(im.shape, dtype=np.complex64)
    if plans is None:
        im_fft = pyfftw.interfaces.numpy_fft.rfft2(im)
    else:
        im_fft = plan_fft(im)
    cont_size = int((np.sqrt(2) - 1) * radius)
    if cont:
        im_ifft_cont = pyfftw.empty_aligned(
            (im.shape[-2], im.shape[-1] // 2 + 1), dtype=np.complex64
        )
        mask_cont = cache(
            cont_size, out=False, center=(0, 0), nb_pix=im_ifft_cont.shape
        )
        mask_cont = np.logical_xor(
            mask_cont,
            cache(
                cont_size,
                out=False,
                center=(0, im_ifft_cont.shape[0]),
                nb_pix=im_ifft_cont.shape,
            ),
        )
        im_ifft_cont[0 : im_ifft_cont.shape[0] // 2, :] = im_fft[
            0 : im_ifft_cont.shape[0] // 2, 0 : im_ifft_cont.shape[1]
        ]
        im_ifft_cont[im_ifft_cont.shape[0] // 2 :, :] = im_fft[
            im_fft.shape[0] - im_ifft_cont.shape[0] // 2 : im_fft.shape[0],
            0 : im_ifft_cont.shape[1],
        ]
        im_ifft_cont[np.logical_not(mask_cont)] = 0
        im_cont = pyfftw.interfaces.numpy_fft.irfft2(im_ifft_cont)
    if center is not None:
        offset = (
            -center[0] + im_fft.shape[-2] // 2,
            -center[1] + im_fft.shape[-1] // 2,
        )
        im_fft = np.roll(im_fft, offset, axis=(-2, -1))
    mask = disk(*im_fft.shape[-2:], center=center, radius=radius)
    im_fft *= mask
    # upper left quadran
    im_ifft[
        ...,
        :radius,
        :radius,
    ] = im_fft[
        ...,
        center[0] : center[0] + radius,
        center[1] : center[1] + radius,
    ]
    # bottom left quadran
    im_ifft[..., -radius:, :radius] = im_fft[
        ...,
        center[0] - radius : center[0],
        center[1] : center[1] + radius,
    ]
    # upper right quadran
    im_ifft[..., :radius, -radius:] = im_fft[
        ..., center[0] : center[0] + radius, center[1] - radius : center[1]
    ]
    # bottom right quadran
    im_ifft[..., -radius:, -radius:] = im_fft[
        ..., center[0] - radius : center[0], center[1] - radius : center[1]
    ]
    # set the rest to 0 bc np.empty does not instantiate an actual empty array
    im_ifft[..., radius:-radius, radius:-radius] = 0
    im_ifft[..., radius:-radius, :radius] = 0
    im_ifft[..., radius:-radius, -radius:] = 0
    im_ifft[..., -radius:, radius:-radius] = 0
    im_ifft[..., :radius, radius:-radius] = 0
    if plans is None:
        im_ifft = pyfftw.interfaces.numpy_fft.ifft2(im_ifft)
    else:
        im_ifft = plan_ifft(im_ifft).copy()
    if im.ndim == 2:
        exp_angle_fast_scalar(
            im_ifft, im_ifft[im_ifft.shape[0] // 2, im_ifft.shape[1] // 2]
        )
    if cont:
        return im_cont, im_ifft
    return im_ifft


def im_osc_fast_t(
    im: np.ndarray,
    radius: int = 0,
    center: Any = None,
    cont: bool = False,
    plans: Any = None,
) -> np.ndarray:
    """Return the field.

    Fast field recovery assuming ideal reference angle i.e minimum fringe
    size of sqrt(2) pixels.

    Truncated for optimal speed: returns an array with size (Ny//2, Nx//2)
    since the recovery process has a resolution of 2px.

    Args:
        im (cp.ndarray): Interferogram
        center (tuple, optional): Center of the field. Defaults to (Ny//4, Nx//4).
        radius (int, optional): Radius of filter in px. Defaults to 512.
        cont (bool, optionnal): Returns the continuous part of the field.
        Defaults to False.
        plans (FFTW plan list, optionnal): [plan_fft, plan_ifft] for optional
        plan caching in streaming applications (like for a viewer).
        Must provide a list of plans for
        both the rfft and ifft.

    Returns:
        np.ndarray: Recovered field
    """
    if plans is not None:
        plan_fft, plan_ifft = plans
    if plans is None:
        im_fft = pyfftw.interfaces.numpy_fft.rfft2(im)
    else:
        im_fft = plan_fft(im)
    if radius == 0:
        radius = min(im_fft.shape[-2:]) // 2
    if center is None:
        center = (im_fft.shape[-2] // 4, im_fft.shape[-1] // 2)
    if cont:
        cont_size = int((np.sqrt(2) - 1) * radius)
        im_ifft_cont = pyfftw.empty_aligned(
            (im.shape[0] // 2, im.shape[1] // 2), dtype=np.complex64
        )
        mask_cont = cache(
            cont_size, out=False, center=(0, 0), nb_pix=im_ifft_cont.shape
        )
        mask_cont = np.logical_xor(
            mask_cont,
            cache(
                cont_size,
                out=False,
                center=(0, im_ifft_cont.shape[0]),
                nb_pix=im_ifft_cont.shape,
            ),
        )
        im_ifft_cont[0 : im_ifft_cont.shape[0] // 2, :] = im_fft[
            0 : im_ifft_cont.shape[0] // 2, 0 : im_ifft_cont.shape[1]
        ]
        im_ifft_cont[im_ifft_cont.shape[0] // 2 :, :] = im_fft[
            im_fft.shape[0] - im_ifft_cont.shape[0] // 2 : im_fft.shape[0],
            0 : im_ifft_cont.shape[1],
        ]
        im_ifft_cont[np.logical_not(mask_cont)] = 0
        im_cont = pyfftw.interfaces.numpy_fft.ifft2(im_ifft_cont)
    im_fft = im_fft[
        ...,
        center[0] - im_fft.shape[-2] // 4 : center[0] + im_fft.shape[-2] // 4,
        : im_fft.shape[-1] - 1,
    ]
    if center is not None:
        offset = (
            -center[0] + im.shape[-2] // 2,
            -center[1] + im_fft.shape[-1] // 2,
        )
        im_fft = np.roll(im_fft, offset, axis=(-2, -1))
    mask = disk(
        *im_fft.shape[-2:],
        center=(im_fft.shape[-2] // 2, im_fft.shape[-1] // 2),
        radius=radius,
    )
    im_fft *= mask
    im_ifft = np.fft.fftshift(im_fft, axes=(-2, -1))
    if plans is None:
        im_ifft = pyfftw.interfaces.numpy_fft.ifft2(im_ifft)
    else:
        im_ifft = plan_ifft(im_ifft).copy()
    if im.ndim == 2:
        exp_angle_fast_scalar(
            im_ifft, im_ifft[im_ifft.shape[0] // 2, im_ifft.shape[1] // 2]
        )
    if cont:
        return im_cont, im_ifft
    return im_ifft

                      
def im_osc_fast_t_cp(im: cp.ndarray, radius: int = None, cont: bool = False, quadran: str = 'upper') -> cp.ndarray:
    """Fast field recovery assuming ideal reference angle i.e minimum fringe size of sqrt(2) pixels
    Truncated for optimal speed
    Args:
        im (cp.ndarray): Interferogram
        radius (int, optional): Radius of filter in px. Defaults to 512.
        return_cont (bool, optionnal): Returns the continuous part of the field. Defaults to False.

    Returns:
        cp.ndarray: Recovered field
    """
    if radius is None:
        radius = max(im.shape)//4
    # center of first quadran
    if quadran == 'upper':
        center = (im.shape[0]//4, im.shape[1]//4)
    elif quadran == 'lower':
        center = (im.shape[0]//4+im.shape[0]//2, im.shape[1]//4)
    assert len(im.shape) == 2, "Can only work with 2D images !"
    # center of first quadran
    im_ifft = cp.zeros(
        (im.shape[0]//2, im.shape[1]//2), dtype=np.complex64)
    im_fft = cp.fft.rfft2(im)
    Y, X = cp.ogrid[:im_fft.shape[0], :im_fft.shape[1]]
    dist_from_center = cp.hypot(X - center[1], Y-center[0])
    mask = dist_from_center > radius
    if cont:
        cont_size = int((np.sqrt(2)-1)*radius)
        im_ifft_cont = cp.empty(
            (im.shape[0]//2, im.shape[1]//2), dtype=np.complex64)
        mask_cont = cache_cp(cont_size, out=False, center=(0, 0),
                             nb_pix=im_ifft_cont.shape)
        mask_cont = cp.logical_xor(mask_cont, cache_cp(cont_size, out=False, center=(0, im_ifft_cont.shape[0]),
                                                       nb_pix=im_ifft_cont.shape))
        im_ifft_cont[0:im_ifft_cont.shape[0]//2,
                     :] = im_fft[0:im_ifft_cont.shape[0]//2, 0:im_ifft_cont.shape[1]]
        im_ifft_cont[im_ifft_cont.shape[0]//2:,
                     :] = im_fft[im_fft.shape[0]-im_ifft_cont.shape[0]//2:im_fft.shape[0], 0:im_ifft_cont.shape[1]]
        im_ifft_cont[cp.logical_not(mask_cont)] = 0
        im_cont = cp.fft.ifft2(im_ifft_cont)
    im_fft[mask] = 0
    if quadran == 'upper':
        im_ifft[:, :] = im_fft[:im_fft.shape[0]//2, :im_fft.shape[1]-1]
    elif quadran=='lower':
        im_ifft[:, :] = im_fft[im_fft.shape[0]//2:, :im_fft.shape[1]-1]
    im_ifft = cp.fft.fftshift(im_ifft)
    im_ifft = cp.fft.ifft2(im_ifft)
    im_ifft *= cp.exp(-1j *
                      cp.angle(im_ifft[im_ifft.shape[0]//2, im_ifft.shape[1]//2]))
    if cont:
        return im_cont, im_ifft
    return im_ifft


def im_osc_fast_cp(im: cp.ndarray, radius: int = 0, cont: bool = False) -> cp.ndarray:
    """Fast field recovery assuming ideal reference angle

    Args:
        im (cp.ndarray): Interferogram
        radius (int, optional): Radius of filter in px. Defaults to 512.
        return_cont (bool, optionnal): Returns the continuous part of the field. Defaults to False.

    Returns:
        np.ndarray: Recovered field
    """
    if radius == 0:
        radius = min(im.shape)//4
    center = (im.shape[0]//4, im.shape[1]//4)
    assert len(im.shape) == 2, "Can only work with 2D images !"
    # center of first quadran
    im_ifft = cp.zeros((im.shape[0], im.shape[1]), dtype=np.complex64)
    im_fft = cp.fft.rfft2(im)
    Y, X = cp.ogrid[:im_fft.shape[0], :im_fft.shape[1]]
    dist_from_center = cp.hypot(X - center[1], Y-center[0])
    mask = dist_from_center > radius
    if cont:
        cont_size = int((np.sqrt(2)-1)*radius)
        im_ifft_cont = im_fft.copy()
        mask_cont = cache_cp(cont_size, out=False, center=(0, 0),
                             nb_pix=im_ifft_cont.shape)
        mask_cont = cp.logical_xor(mask_cont, cache_cp(cont_size, out=False, center=(0, im_ifft_cont.shape[0]),
                                                       nb_pix=im_ifft_cont.shape))
        im_ifft_cont[cp.logical_not(mask_cont)] = 0
        im_cont = cp.fft.irfft2(im_ifft_cont)
    im_fft[mask] = 0
    im_ifft[im_ifft.shape[0]//2-radius:im_ifft.shape[0]//2+radius,
            im_ifft.shape[1]//2-radius:im_ifft.shape[1]//2+radius] = im_fft[center[0]-radius:center[0]+radius,
                                                                            center[1]-radius:center[1]+radius]
    im_ifft = cp.fft.fftshift(im_ifft)
    im_ifft = cp.fft.ifft2(im_ifft)
    im_ifft *= np.exp(-1j *
                      cp.angle(im_ifft[im_ifft.shape[0]//2, im_ifft.shape[1]//2]))
    if cont:
        return im_cont, im_ifft
    return im_ifft


def phase_fast(im: np.ndarray, radius: int = 0, cont: bool = False) -> np.ndarray:
    """Fast phase recovery assuming ideal reference angle

    Args:
        im (np.ndarray): Interferogram
        radius (int, optional): Radius of filter in px. Defaults to a quarter the size of the image.
        return_cont (bool, optionnal): Returns the continuous part of the field. Defaults to false.

    Returns:
        np.ndarray: Recovered phase
    """
    if cont:
        im_ifft, im_cont = im_osc_fast_t(im, radius=radius, cont=True)
        phase = angle_fast(im_ifft)
        return phase, im_cont
    im_ifft = im_osc_fast(im, radius=radius, cont=False)
    phase = angle_fast(im_ifft)
    return phase


def phase_fast_cp(im: cp.ndarray, radius: int = 0, cont: bool = False) -> cp.ndarray:
    """Fast phase recovery assuming ideal reference angle

    Args:
        im (cp.ndarray): Interferogram
        radius (int, optional): Radius of filter in px. Defaults to 512.
        return_cont (bool, optionnal): Returns the continuous part of the field. Defaults to False.

    Returns:
        np.ndarray: Recovered phase
    """
    angle = cp.empty((im.shape[0]//2, im.shape[1]//2), dtype=np.float32)
    tpb = 16
    bpgx = math.ceil(angle.shape[0]/tpb)
    bpgy = math.ceil(angle.shape[1]/tpb)
    if cont:
        im_ifft, im_cont = im_osc_fast_t_cp(im, radius=radius, cont=True)
        angle_fast_cp[(bpgx, bpgy), (tpb, tpb)](im_ifft, angle)
        return angle, im_cont
    im_ifft = im_osc_fast_t_cp(im, radius=radius, cont=False)
    angle_fast_cp[(bpgx, bpgy), (tpb, tpb)](im_ifft, angle)
    return angle


def contr_fast(im: np.ndarray) -> np.ndarray:
    """Computes the contrast of an interferogram assuming proper alignment
    i.e minimum fringe size of sqrt(2) pixels

    Args:
        im (np.ndarray): The interferogram

    Returns:
        np.ndarray: The contrast map
    """
    im_cont, im_fringe = im_osc_fast(im, cont=True)
    analytic = np.abs(im_fringe)
    cont = np.abs(im_cont)
    return 2*analytic/cont


def contr_fast_cp(im: cp.ndarray) -> cp.ndarray:
    """Computes the contrast of an interferogram assuming proper alignment
    i.e minimum fringe size of sqrt(2) pixels

    Args:
        im (np.ndarray): The interferogram

    Returns:
        np.ndarray: The contrast map
    """
    im_cont, im_fringe = im_osc_fast_cp(im, cont=True)
    analytic = cp.abs(im_fringe)
    cont = cp.abs(im_cont)
    return 2*analytic/cont
