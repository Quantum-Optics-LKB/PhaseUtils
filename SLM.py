# -*- coding: utf-8 -*-
"""
Created by Tangui Aladjidi at 19/03/2021
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


class SLMscreen:
    def __init__(self, resX: int, resY: int, name: str = "SLM"):
        """Initializes the SLM screen

        Args:
            resX (int): Resolution along x axis
            resY (int): Resolution along y axis
            name (str, optional): Name of the SLM window. Defaults to "SLM".
        """
        self.name = name
        cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(name, resX, 0)
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
        self.update(np.ones((resY, resX), dtype=np.uint8))

    def update(self, A: np.ndarray, delay: int = 1):
        """Updates the pattern on the SLM

        Args:
            A (np.ndarray): Array
            delay (int, optional): Delay in ms. Defaults to 1.
        """
        assert A.dtype == np.uint8, "Only 8 bits patterns are supported by the SLM !"
        cv2.imshow(self.name, A)
        cv2.waitKey(1+delay)

    def close(self):
        cv2.destroyWindow(self.name)


def phase_intensity(intensity: np.ndarray, phase: np.ndarray,
                    theta: int = 45, pitch: int = 8, cal_value: int = 255) -> np.ndarray:
    """Function to generate SLM pattern with given intensity and phase
    Based on Davis et al. Encoding amplitude information onto phase-only filters
    https://opg.optica.org/ao/fulltext.cfm?uri=ao-38-23-5004&id=60239 

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
    assert intensity.shape == phase.shape, "Shape mismatch between phase and intensity !"
    m, n = intensity.shape
    # normalize input
    intensity = intensity/np.nanmax(intensity)
    # this corrects the aberrations described in Section 6. eq. 26
    # assumes we look at the first order
    intensity /= np.sqrt(np.sinc(1-intensity))
    # normalize again since we divided by values smaller than 1
    intensity /= np.nanmax(intensity)
    # generate grating with given angle
    grat = grating(m, n, theta, pitch)
    grat *= 2*np.pi
    # generate modulation field according to eq.9
    field = np.exp(1j*grat + 1j*phase)
    phase_map = intensity*(np.angle(field) + np.pi)
    phase_map -= np.pi*intensity
    phase_map = phase_map % (2.0*np.pi) / (2.0*np.pi) * cal_value
    phase_map = phase_map // 1
    return phase_map


def phase_only(intensity: np.ndarray, phase: np.ndarray,
               theta: int = 45, pitch: int = 8, cal_value: int = 255) -> np.ndarray:
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
    assert intensity.shape == phase.shape, "Shape mismatch between phase and intensity !"
    m, n = intensity.shape
    # normalize input
    intensity = intensity/np.nanmax(intensity)
    # this corrects the aberrations described in Section 6. eq. 26
    # assumes we look at the first order
    intensity /= np.sinc(1-intensity)
    # normalize again since we divided by values smaller than 1
    intensity /= np.nanmax(intensity)
    # generate grating with given angle
    grat = grating(m, n, theta, pitch)
    grat *= 2*np.pi
    # generate modulation field according to eq.9
    field = np.exp(1j*grat + 1j*phase)
    phase_map = intensity*(np.angle(field) + np.pi)
    phase_map -= np.pi*intensity
    phase_map = phase_map % (2.0*np.pi) / (2.0*np.pi) * cal_value
    phase_map = phase_map // 1

    return phase_map.astype(np.uint8)


def grating(m: int, n: int, theta: float = 45, pitch: int = 8) -> np.ndarray:
    """Generates a grating of size (m, n) 

    Args:
        m (int): Size along i axis
        n (int): Size along j axis
        theta (float, optional): Angle of the grating in degrees wrt horizontal axis. 
        Defaults to 45.
        pitch (int, optional): Pixel pitch. Defaults to 8.

    Returns:
        np.ndarray: Array representing the grating
    """
    YY, XX = np.mgrid[0:m, 0:n]
    grating = np.cos(np.pi/180*theta)*YY + np.sin(np.pi/180*theta)*XX
    grating %= pitch
    grating /= pitch
    return grating


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
    X -= n//2
    Y -= m//2
    out = np.zeros_like(X, dtype='uint8')
    Radii = np.sqrt(X**2 + Y**2)
    cond = Radii > (R-width/2)
    cond &= Radii < (R+width/2)
    out[cond] = value
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
    out = np.zeros((m, n), dtype='uint8')
    out[x0-width//2:x0+width//2, :] = 1
    out[:, y0-width//2:y0+width//2] = 1
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
    out[int(m/2):, :] = -1
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
    X, Y = np.mgrid[0:x.shape[0], 0:x.shape[1]]
    condx = X % (2*gridsize) < gridsize
    condy = Y % (2*gridsize) < gridsize
    x[np.logical_xor(condx, condy)] = True
    return x


def vortex(m: int, n: int, i: int, j: int, l: int) -> np.ndarray:
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
    ii, jj = np.mgrid[0:x.shape[0], 0:x.shape[1]]
    return np.angle((ii-i+1j*(jj-j))**l)


def black_hole_phase_profile(m: int, n: int, x_center: float, y_center: float,
                             l: int = 1,
                             r_velocity: float = 0,
                             SLM_pitch: float = 8e-6) -> np.ndarray:
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
    XX, YY = np.meshgrid(np.linspace(-n/2, n/2, n, dtype=float)*SLM_pitch,
                         np.linspace(-m/2, m/2, m, dtype=float)*SLM_pitch)

    XX = XX - x_center*SLM_pitch
    YY = YY - y_center*SLM_pitch
    vortex_phase = np.arctan2(YY, XX)
    r = np.sqrt(XX**2.0 + YY**2.0)
    radial_velocity = -2.0*np.pi*np.sqrt(r/r_velocity)
    phase_temp = np.exp(1j*vortex_phase*l + 1j*radial_velocity)
    return np.angle(phase_temp)


def bragg_phase_profile(m: int, n: int, kp: float, alpha: float = 0.1,
                        beta: float = 0,
                        SLM_pitch: float = 8e-6) -> np.ndarray:
    """Defines a bragg spectroscopy phase profile

    Args:
        m (int): Size of the pattern in i
        n (int): Size of the pattern in j
        kp (float): wavenumber to proble
        alpha (float): modulation effect

    Returns:
        np.ndarray: The phase mask to display on the SLM
    """
    XX, YY = np.meshgrid(np.linspace(-n/2, n/2, n, dtype=float)*SLM_pitch,
                         np.linspace(-m/2, m/2, m, dtype=float)*SLM_pitch)
    phase_temp = np.exp(1j*alpha*np.cos(kp*XX) + 1j*beta)
    return np.angle(phase_temp)


def codify_two_values(m: int, n: int, f1: float, f2: float,
                      size: int = 16, max_value: float = 2.0*np.pi,
                      x_shift: int = 0, y_shift: int = 0) -> np.ndarray:
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
    mask[int(m/2 - size - y_shift):int(m/2 + size - y_shift),
         int(n/2 - size - x_shift):int(n/2 + size - x_shift)] = expanded_codification
    phase_temp = np.exp(1j*mask*max_value)
    return np.angle(phase_temp)


def main():
    import sys
    import time
    from PIL import Image
    resX, resY = 1920, 1080
    slm = SLMscreen(resX, resY)
    T = np.zeros(20)
    for i in range(20):
        sys.stdout.flush()
        one = np.ones((resY, resX), dtype=np.uint8)
        slm_pic = (i % 2)*one[:, 0:resX//2] + \
            ((i+1) % 2)*255*one[:, resX//2:]
        slm_pic = np.random.choice(
            [0, 255], size=(resY, resX)).astype(np.uint8)
        t0 = time.time()
        slm.update(slm_pic, delay=1)
        t = time.time()-t0
        T[i] = t
        sys.stdout.write(f"\r{i+1} : time displayed = {t} s")
    slm.close()

    print(f"\nAverage display time = {np.mean(T)} ({np.std(T)}) s")


if __name__ == "__main__":
    main()
