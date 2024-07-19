# -*- coding: utf-8 -*-
"""
Created by Tangui Aladjidi at 19/03/2021
"""

import cv2
import numpy as np
import numba
import math
from PhaseUtils.fast_interp import interp1d
from scipy.interpolate import make_interp_spline
import screeninfo
from functools import lru_cache
from typing import Any

x = np.linspace(-np.pi, 1e-15, 100)
y = np.sin(x) / x
y[-1] = 1
# carry out the interpolation
interpolator = make_interp_spline(y, x, k=1)
N_interp = 2000
f = interpolator(np.linspace(0, 1, N_interp))
f[-1] = 0
inv_sinc = interp1d(0, 1, 1 / N_interp, f, e=3, k=1)
DIFFUSION_MAPS = {}
DIFFUSION_MAPS["floyd-steinberg"] = np.array(
    [
        [1, 0, 7, 16],
        [-1, 1, 3, 16],
        [0, 1, 5, 16],
        [1, 1, 1, 16],
    ]
)
DIFFUSION_MAPS["atkinson"] = np.array(
    [
        [1, 0, 1, 8],
        [2, 0, 1, 8],
        [-1, 1, 1, 8],
        [0, 1, 1, 8],
        [1, 1, 1, 8],
        [0, 2, 1, 8],
    ]
)
DIFFUSION_MAPS["jarvis-judice-ninke"] = np.array(
    [
        [1, 0, 7, 48],
        [2, 0, 5, 48],
        [-2, 1, 3, 48],
        [-1, 1, 5, 48],
        [0, 1, 7, 48],
        [1, 1, 5, 48],
        [2, 1, 3, 48],
        [-2, 2, 1, 48],
        [-1, 2, 3, 48],
        [0, 2, 5, 48],
        [1, 2, 3, 48],
        [2, 2, 1, 48],
    ]
)
DIFFUSION_MAPS["stucki"] = np.array(
    [
        [1, 0, 8, 42],
        [2, 0, 4, 42],
        [-2, 1, 2, 42],
        [-1, 1, 4, 42],
        [0, 1, 8, 42],
        [1, 1, 4, 42],
        [2, 1, 2, 42],
        [-2, 2, 1, 42],
        [-1, 2, 2, 42],
        [0, 2, 4, 42],
        [1, 2, 2, 42],
        [2, 2, 1, 42],
    ]
)
DIFFUSION_MAPS["burkes"] = np.array(
    [
        [1, 0, 8, 32],
        [2, 0, 4, 32],
        [-2, 1, 2, 32],
        [-1, 1, 4, 32],
        [0, 1, 8, 32],
        [1, 1, 4, 32],
        [2, 1, 2, 32],
    ]
)
DIFFUSION_MAPS["sierra3"] = np.array(
    [
        [1, 0, 5, 32],
        [2, 0, 3, 32],
        [-2, 1, 2, 32],
        [-1, 1, 4, 32],
        [0, 1, 5, 32],
        [1, 1, 4, 32],
        [2, 1, 2, 32],
        [-1, 2, 2, 32],
        [0, 2, 3, 32],
        [1, 2, 2, 32],
    ]
)
DIFFUSION_MAPS["sierra2"] = np.array(
    [
        [1, 0, 4, 16],
        [2, 0, 3, 16],
        [-2, 1, 1, 16],
        [-1, 1, 2, 16],
        [0, 1, 3, 16],
        [1, 1, 2, 16],
        [2, 1, 1, 16],
    ]
)
DIFFUSION_MAPS["sierra-2-4a"] = np.array(
    [
        [1, 0, 2, 4],
        [-1, 1, 1, 4],
        [0, 1, 1, 4],
    ]
)


class SLMscreen:
    def __init__(self, position: int, name: str = "SLM"):
        """Initializes the SLM screen

        Args:
            position (int): Position of the SLM screen
            name (str, optional): Name of the SLM window. Defaults to "SLM".
        """
        self.name = name
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        shift = None
        for m in screeninfo.get_monitors():
            print(m)
            if str(position) in m.name:
                shift = m.x
                self.resX = m.width
                self.resY = m.height
        if shift is None:
            print("ERROR : Could not find SLM !")
        cv2.moveWindow(name, shift, 0)
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.update(np.ones((self.resY, self.resX), dtype=np.uint8))

    def update(self, A: np.ndarray, delay: int = 1):
        """Updates the pattern on the SLM

        Args:
            A (np.ndarray): Array
            delay (int, optional): Delay in ms. Defaults to 1.
        """
        assert A.dtype == np.uint8, "Only 8 bits patterns are supported by the SLM !"
        cv2.imshow(self.name, A)
        cv2.waitKey(1 + delay)

    def close(self):
        cv2.destroyWindow(self.name)


@numba.njit(fastmath=True, cache=True, parallel=True, boundscheck=False)
def _phase_amplitude(amp: np.ndarray, phase: np.ndarray, grat: np.ndarray) -> None:
    for i in numba.prange(amp.shape[0]):
        for j in numba.prange(amp.shape[1]):
            grat[i, j] *= 2 * np.pi
            amp[i, j] /= np.pi
            amp[i, j] += 1
            phase[i, j] -= np.pi * amp[i, j]
            # generate modulation field according to eq.9
            phase[i, j] += grat[i, j]
            phase[i, j] %= 2 * np.pi
            phase[i, j] *= amp[i, j]
            phase[i, j] /= 2.0 * np.pi


def phase_amplitude(
    amp: np.ndarray,
    phase: np.ndarray,
    grat: np.ndarray = None,
    theta: int = 45,
    pitch: int = 8,
    cal_value: int = 256,
) -> np.ndarray:
    """Function to generate SLM pattern with given intensity and phase
    Based on Bolduc et al. Exact solution to simultaneous intensity and
    phase encryption with a single phase-only hologram
    https://opg.optica.org/ol/fulltext.cfm?uri=ol-38-18-3546&id=260924

    Args:
        intensity (np.ndarray): Intensity mask
        phase (np.ndarray): Phase mask
        theta (int, optional): Angle of the grating in degrees. Defaults to 45.
        pitch (int, optional): Pixel pitch of the grating in px
        cal_value (int, optional): Pixel value corresponding to a 2pi dephasing on the SLM.

    Returns:
        np.ndarray: The phase mask to display on the SLM
    """
    # check that shapes match
    assert amp.shape == phase.shape, "Shape mismatch between phase and intensity !"
    m, n = amp.shape
    # normalize input to less than 1 for the inv_sinc function
    amp /= np.nanmax(amp) + 1e-3
    # generate grating with given angle
    if grat is None:
        grat = grating(m, n, theta, pitch)
    # eq 4
    amp = inv_sinc(amp)
    _phase_amplitude(amp, phase, grat)
    phase *= cal_value
    phase = np.round(phase)
    return phase.astype(np.uint8)


def phase_only(
    intensity: np.ndarray,
    phase: np.ndarray,
    theta: int = 45,
    pitch: int = 8,
    cal_value: int = 255,
) -> np.ndarray:
    """Function to generate SLM pattern with only phase

    Args:
        intensity (np.ndarray): Intensity mask.
        phase (np.ndarray): Phase mask
        theta (int, optional): Angle of the grating in degrees. Defaults to 45.
        pitch (int, optional): Pixel pitch of the grating in px
        cal_value (int, optional): Pixel value corresponding to a 2pi dephasing on the SLM.

    Returns:
        np.ndarray: The phase mask to display on the SLM
    """
    # check that shapes match
    assert (
        intensity.shape == phase.shape
    ), "Shape mismatch between phase and intensity !"
    m, n = intensity.shape
    # normalize input
    intensity = intensity / np.nanmax(intensity)
    # generate grating with given angle
    grat = grating(m, n, theta, pitch)
    grat *= 2 * np.pi
    # generate modulation field according to eq.9
    field = np.exp(1j * grat + 1j * phase)
    phase_map = np.angle(field) + 2.0 * np.pi
    phase_map = phase_map % (2.0 * np.pi) / (2.0 * np.pi) * cal_value
    phase_map = phase_map // 1

    return phase_map.astype(np.uint8)


@lru_cache(maxsize=10)
@numba.njit(cache=True, parallel=True, boundscheck=False)
def mgrid(m: int, n: int):
    """Numba compatible mgrid in i,j indexing style

    Args:
        m (int) : size along i axis
        n (int) : size along j axis
    Returns:
        np.ndarray: xx, yy like numpy's meshgrid
    """
    xx = np.empty((m, n), dtype=np.uint64)
    yy = np.empty((m, n), dtype=np.uint64)
    for i in numba.prange(m):
        for j in numba.prange(n):
            xx[i, j] = j
            yy[i, j] = i
    return yy, xx


@lru_cache(maxsize=10)
@numba.njit(fastmath=True, cache=True, parallel=True, boundscheck=False)
def grating(m: int, n: int, theta: float = 45, pitch: int = 8) -> np.ndarray:
    """Generates a grating of size (m, n)

    Args:
        m (int): Size along i axis
        n (int): Size along j axis
        theta (float, optional): Angle of the grating in degrees. Defaults to 45.
        pitch (int, optional): Pixel pitch. Defaults to 8.

    Returns:
        np.ndarray: Array representing the grating
    """
    grating = np.zeros((m, n), dtype=np.float32)
    c = math.cos(np.pi / 180 * theta)
    s = math.sin(np.pi / 180 * theta)
    for i in numba.prange(m):
        for j in numba.prange(n):
            grating[i, j] = c * i + s * j
            grating[i, j] %= pitch
            grating[i, j] /= pitch
    return grating


@numba.njit(fastmath=True, cache=True, boundscheck=False)
def error_diffusion_dithering(ni, diff_map=DIFFUSION_MAPS["floyd-steinberg"]):
    """Perform image dithering by error diffusion method.

    Reference:
        http://bisqwit.iki.fi/jutut/kuvat/ordered_dither/error_diffusion.txt

    Call the diffusion maps as SLM.DIFFUSION_MAPS["map_name"].

    Quantization error of *current* pixel is added to the pixels
    on the right and below according to the formulas below.
    This works nicely for most static pictures, but causes
    an avalanche of jittering artifacts if used in animation.

    Floyd-Steinberg:

              *  7
           3  5  1      / 16

    Jarvis-Judice-Ninke:

              *  7  5
        3  5  7  5  3
        1  3  5  3  1   / 48

    Stucki:

              *  8  4
        2  4  8  4  2
        1  2  4  2  1   / 42

    Burkes:

              *  8  4
        2  4  8  4  2   / 32


    Sierra3:

              *  5  3
        2  4  5  4  2
           2  3  2      / 32

    Sierra2:

              *  4  3
        1  2  3  2  1   / 16

    Sierra-2-4A:

              *  2
           1  1         / 4

    Stevenson-Arce:

                      *   .  32
        12   .   26   .  30   .  16
        .   12    .  26   .  12   .
        5    .   12   .  12   .   5    / 200

    Atkinson:

              *   1   1    / 8
          1   1   1
              1

    Args:
        image (np.ndarray): The image to apply error
          diffusion dithering to.
        diff_map (np.ndarray): The error diffusion map to use.
    Returns:
        ni (np.ndarray): The error diffusion dithered array.
    """
    for y in range(ni.shape[0]):
        for x in range(ni.shape[1]):
            old_pixel = ni[y, x]
            new_pixel = np.round(old_pixel)
            quantization_error = old_pixel - new_pixel
            ni[y, x] = new_pixel
            for dx, dy, diffusion_coeff0, diffusion_coeff1 in diff_map:
                xn, yn = x + dx, y + dy
                if (0 <= xn < ni.shape[1]) and (0 <= yn < ni.shape[0]):
                    ni[yn, xn] += (
                        quantization_error * diffusion_coeff0 / diffusion_coeff1
                    )
    return ni


@numba.njit(fastmath=True, cache=True, boundscheck=False)
def floyd_steinberg(image: np.ndarray):
    """Apply the Floyd-Steinberg dithering algorithm to an image.

    Args:
        image (np.ndarray): Float image between 0 and 1.
    """
    h, w = image.shape
    for y in range(h):
        for x in range(w):
            old = image[y, x]
            new = np.round(old)
            image[y, x] = new
            error = old - new
            # precomputing the constants helps
            if x + 1 < w:
                image[y, x + 1] += error * 0.4375  # right, 7 / 16
            if (y + 1 < h) and (x + 1 < w):
                image[y + 1, x + 1] += error * 0.0625  # right, down, 1 / 16
            if y + 1 < h:
                image[y + 1, x] += error * 0.3125  # down, 5 / 16
            if (x - 1 >= 0) and (y + 1 < h):
                image[y + 1, x - 1] += error * 0.1875  # left, down, 3 / 16


def circle(m: int, n: int, R: int, width: int = 20, value: int = 255) -> np.ndarray:
    """Draws a circle

    Args:
        m (int): Size in i
        n (int): Size in j
        R (int): Radius in px
        width (int, optional): Width of the circle in px. Defaults to 20.
        value (int, optional): Value inside the circle. Defaults to 255.

    Returns:
        np.ndarray: _description_
    """
    Y, X = np.mgrid[0:m, 0:n]
    X -= n // 2
    Y -= m // 2
    out = np.zeros_like(X, dtype="uint8")
    Radii = np.sqrt(X**2 + Y**2)
    cond = Radii > (R - width / 2)
    cond &= Radii < (R + width / 2)
    out[cond] = value
    return out


def disk(m: int, n: int, R: int, value: int = 255) -> np.ndarray:
    """Draws a disk

    Args:
        m (int): Size in i
        n (int): Size in j
        R (int): Radius in px
        value (int, optional): Value inside the circle. Defaults to 255.

    Returns:
        np.ndarray: _description_
    """
    ii, jj = np.mgrid[0:m, 0:n]
    jj = jj - n / 2
    ii = ii - m / 2
    out = np.zeros_like(ii, dtype="uint8")
    Radii = np.sqrt(ii**2 + jj**2)
    cond = Radii < R
    out[cond] = value
    return out


def hyper_gaussian(m: int, n: int, R: int, k: float, value: int = 255) -> np.ndarray:
    """Draw a hypergaussian.

    Args:
        m (int): Size in i
        n (int): Size in j
        R (int): Radius in px
        value (int, optional): Value inside the circle. Defaults to 255.

    Returns:
        np.ndarray: The beam profile.
    """
    ii, jj = np.mgrid[0:m, 0:n]
    jj = jj - n / 2
    ii = ii - m / 2
    out = np.zeros_like(ii, dtype="uint8")
    Radii = np.sqrt(ii**2 + jj**2)
    out = np.exp(-(Radii**k) / R**k)
    return out


def cross(m: int, n: int, x0: int = 0, y0: int = 0, width: int = 2) -> np.ndarray:
    """Defines a cross pattern for alignment

    Args:
        m (int): Size in i
        n (int): Size in j
        x0 (int, optional): Center position in x. Defaults to 0.
        y0 (int, optional): Center position in y. Defaults to 0.
        width (int, optional): Width of the cross in px. Defaults to 2.

    Returns:
        np.ndarray: _description_
    """
    out = np.zeros((m, n), dtype="uint8")
    out[x0 - width // 2 : x0 + width // 2, :] = 1
    out[:, y0 - width // 2 : y0 + width // 2] = 1
    return out


def snake_instability(m: int, n: int) -> np.ndarray:
    """Defines a phase slip to observe the snake instability phenomena

    Args:
        m (int): Size in i
        n (int): Size in j

    Returns:
        np.ndarray: The phase mask to display on the SLM
    """
    out = np.ones((m, n), dtype=float)
    # out[x0-width//2:x0+width//2, :] = 1
    out[int(m / 2) :, :] = -1
    return np.angle(out)


def checkerboard(m: int, n: int, gridsize: int = 20) -> np.ndarray:
    """Defines a square checkerboard pattern for camera alignment

    Args:
        m (int): Size of the pattern in i
        n (int): Size of the pattern in j
        gridsize (int, optional): Size of the board squares. Defaults to 20.

    Returns:
        np.ndarray: The phase mask to display on the SLM
    """
    x = np.zeros((1080, 1920), dtype=bool)
    X, Y = np.mgrid[0 : x.shape[0], 0 : x.shape[1]]
    condx = X % (2 * gridsize) < gridsize
    condy = Y % (2 * gridsize) < gridsize
    x[np.logical_xor(condx, condy)] = True
    return x


def vortex(m: int, n: int, i: int, j: int, ll: int) -> np.ndarray:
    """Defines a vortex phase pattern

    Args:
        m (int): Size of the pattern in i
        n (int): Size of the pattern in j
        i (int): i position of vortex
        j (int): j position of vortex
        l (int): Charge of vortex

    Returns:
        np.ndarray: The phase mask to display on the SLM
    """
    x = np.zeros((m, n), dtype=bool)
    ii, jj = np.mgrid[0 : x.shape[0], 0 : x.shape[1]]
    return np.angle((ii - i + 1j * (jj - j)) ** ll)


def jrs(m: int, n: int, y0: int, x0: int, Mach: float, xi: float) -> np.ndarray:
    """Define a JRS phase pattern

    Args:
        m (int): Size of the pattern in i
        n (int): Size of the pattern in j
        i (int): i position of center
        j (int): j position of center
        l (int): size of jrs
        velo (float): velocity of the jrs

    Returns:
        np.ndarray: The phase mask to display on the SLM
    """
    y, x = np.mgrid[0:m, 0:n]
    x = x - x0
    y = y - y0
    eps = np.sqrt(1 - Mach**2)
    amp = 1 - 4 * eps**2 * (
        (3 / 2 + eps**4 * y**2 / xi**2 - eps**2 * x**2 / xi**2)
        / (3 / 2 + eps**4 * y**2 / xi**2 + eps**2 * x**2 / xi**2) ** 2
    )
    amp = np.sqrt(amp)
    phi = (
        -2
        * np.sqrt(2)
        * eps
        * (eps * x / xi)
        / (3 / 2 + eps**4 * y**2 / xi**2 + eps**2 * x**2 / xi**2)
    )
    return amp, phi


def black_hole_phase_profile(
    m: int,
    n: int,
    x_center: float,
    y_center: float,
    ll: int = 1,
    r_velocity: float = 0,
    SLM_pitch: float = 8e-6,
) -> np.ndarray:
    """Defines a 2D Kerr rotating black hole profile

    Args:
        m (int): Size of the pattern in i
        n (int): Size of the pattern in j
        x_center (float): x center of the vortex
        y_center (float): y center of the vortex
        l (int): Charge of the vortex
        r_velocity (float): Radial velocity

    Returns:
        np.ndarray: The phase mask to display on the SLM
    """
    XX, YY = np.meshgrid(
        np.linspace(-n / 2, n / 2, n, dtype=float) * SLM_pitch,
        np.linspace(-m / 2, m / 2, m, dtype=float) * SLM_pitch,
    )
    XX = XX - x_center * SLM_pitch
    YY = YY - y_center * SLM_pitch
    vortex_phase = np.arctan2(YY, XX)
    r = np.sqrt(XX**2.0 + YY**2.0)
    radial_velocity = -2.0 * np.pi * np.sqrt(r / r_velocity)
    phase_temp = np.exp(1j * vortex_phase * ll + 1j * radial_velocity)
    return np.angle(phase_temp)


@numba.njit(fastmath=True, cache=True, parallel=True, boundscheck=False)
def bragg_density_profile(
    m: int,
    n: int,
    kp: float,
    alpha: float = 0.1,
    SLM_pitch: float = 8.0e-6,
    width: int = 250,
):
    """Generates a density modulation in cos(kp*xx)

    Args:
        m (int): Number of rows
        n (int): Number of columns
        kp (float): Wavevector in m^-1
        alpha (float, optional): Modulation depth. Defaults to 0.1.
        SLM_pitch (float, optional): SLM pixel pitch in m. Defaults to 8.0e-6
        width (int, optional): Width of the strip pattern on the SLM in pixels
    """
    inten = np.ones((m, n))
    for i in numba.prange(m):
        for j in numba.prange(n):
            inten[i, j] -= alpha * (1 + np.cos(2 * np.pi * kp * j * SLM_pitch)) / 2
    inten[0 : m // 2 - width // 2, :] = 0
    inten[m // 2 + width // 2 :, :] = 0
    return inten


def codify_two_values(
    m: int,
    n: int,
    f1: float,
    f2: float,
    size: int = 16,
    max_value: float = 2.0 * np.pi,
    x_shift: int = 0,
    y_shift: int = 0,
) -> np.ndarray:
    """Defines a two value codification for the extreme learning machines

    Args:
        m (int): Size of the pattern in i
        n (int): Size of the pattern in j
        f1 (float): first value to encode on the beam
        f2 (float): second value to encode on the beam
        size (int): Half width of the codification square zone. Default is 16
        max_value (float): Max phase value availabe for the codification. Default is 2pi
        x_shift (int): Shift to apply along the x coordinate to the phase mask.
                       Default is 0
        y_shift (int): Shift to apply along the y coordinate to the phase mask.
                       Default is 0
        alpha (float): modulation effect

    Returns:
        np.ndarray: The phase mask to display on the SLM
    """

    unit_codification = np.array([[f1, f2], [f1, f2]])
    expanded_codification = np.kron(unit_codification, np.ones((size, size)))
    mask = np.zeros((m, n))
    mask[
        int(m / 2 - size - y_shift) : int(m / 2 + size - y_shift),
        int(n / 2 - size - x_shift) : int(n / 2 + size - x_shift),
    ] = expanded_codification

    phase_temp = np.exp(1j * mask * max_value)
    return np.angle(phase_temp)


def hyberbolic_tangent(
    m: int, n: int, width: float, contrast: float = 1, angle: float = 0
) -> np.ndarray:
    """Generate a hyberbolic tangent profile.
    This generates a smoothed jump from 1-contrast to contrast
    using a tanh function.

    Args:
        m (int): Height of the pattern
        n (int): Width of the pattern
        width (float): Width of the transition zone
        contrast (float, optional): Height difference between the two zones.
        Defaults to 1.
        angle (float, optional): Angle of the transition zone.
        Defaults to 0 (vertical).
    """
    angle = np.pi * angle / 180
    y, x = mgrid(m, n)
    x = x - n / 2.0
    y = y - m / 2.0
    jump = np.tanh(np.cos(angle) * x / width + np.sin(angle) * y / width)
    jump = (jump + 1) / 2
    jump = 1 - contrast + contrast * jump
    return jump


def hyberbolic_tangent_2d(
    m: int,
    n: int,
    contrast: float = 1,
    angle: float = 90,
    pos: tuple = (0, 0),
    xi: float = 1,
) -> np.ndarray:
    """Generate a 2d hyberbolic tangent profile.
    This generates a smoothed jump from 1-contrast to contrast
    using a tanh function.

    Args:
        m (int): Height of the pattern
        n (int): Width of the pattern
        width (float): Width of the transition zone
        contrast (float, optional): Height difference between the two zones.
        Defaults to 1.
        angle (float, optional): Angle of the transition zone.
        Defaults to 0 (vertical).
    """
    angle = np.pi * angle / 180
    y, x = mgrid(m, n)
    x = x - n / 2.0 + pos[0]
    y = y - m / 2.0 + pos[1]
    jump = np.tanh(
        np.sqrt(
            (np.cos(angle) * x + np.sin(angle) * y) ** 2
            + (np.cos(angle) * x - np.sin(angle) * y) ** 2
        )
        / (np.sqrt(2) * xi)
    )
    # jump = (jump + 1) / 2
    jump = np.ones((m, n)) - contrast + contrast * jump
    return jump


def lens(m: int, n: int, f: float, d: float = 8e-6, wvl: float = 780e-9) -> np.ndarray:
    """Lens phase profile.

    Args:
        m (int): Rows
        n (int): Cols
        f (float): Focal length in m
        d (float): Pixel pitch of the slm in m
        wvl (float): Wavelength in m

    Returns:
        np.ndarray: The lens phase profile
    """
    x = np.linspace(-n / 2, n / 2, n) * d
    y = np.linspace(-m / 2, m / 2, m) * d
    X, Y = np.meshgrid(x, y)
    return np.angle(np.exp(1j * 2 * np.pi * (X**2 + Y**2) / (2 * f * wvl)))


def smooth_ring(m, n, ell, p, r0, dr) -> tuple[Any, Any]:
    y, x = mgrid(m, n)
    x = x - n / 2
    y = y - m / 2
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    t_field = np.exp(1j * (ell * theta + p * r))
    t_field = t_field * np.exp(-((r - r0) ** 2) / (dr / 2) ** 2)
    Amp = np.abs(t_field)
    Phase = np.angle(t_field)
    return Amp, Phase
