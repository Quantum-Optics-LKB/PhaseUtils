import numpy as np
import pyfftw
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider
import multiprocessing
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import pickle
import PhaseUtils.contrast as contrast
import matplotlib.style as mplstyle
from typing import Any

mplstyle.use("fast")
plt.switch_backend("Qt5Agg")
pyfftw.interfaces.cache.enable()
# try to load previous fftw wisdom
try:
    with open("fft.wisdom", "rb") as file:
        wisdom = pickle.load(file)
        pyfftw.import_wisdom(wisdom)
except FileNotFoundError:
    print("No FFT wisdom found, starting over ...")

# Assumes a camera that has OpenCV-like commands to get a frame
# like cam.read()

# For the interferometry setup, assumes the highest possible angle
# between reference and signal beams as described in the demo_contrast
# notebook


def monitor_fourier_fringes(cam: Any) -> None:
    """Displays a pop up window to monitor in live the position of satellite peaks
    in the Fourier space

    Args:
        cam (any): the camera object

    Returns:
        None
    """
    # grab first frame for reference
    ret, frame = cam.read()
    # allocate array to store current picture
    frames = pyfftw.empty_aligned(frame.shape, dtype=np.float32)
    frames[:, :] = frame
    # instantiate frame handler for asynchronous acquisition
    frames_fft = pyfftw.empty_aligned(
        (frame.shape[0], frame.shape[1] // 2 + 1), dtype=np.complex64
    )
    plan_rfft = pyfftw.builders.rfft2(
        frames, planner_effort="FFTW_PATIENT", threads=multiprocessing.cpu_count()
    )
    with open("fft.wisdom", "wb") as file:
        wisdom = pyfftw.export_wisdom()
        pickle.dump(wisdom, file)
    plan_rfft(frames, frames_fft)
    fig, ax = plt.subplots(1, 2)
    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    divider1 = make_axes_locatable(ax[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plan_rfft(frames, frames_fft)
    # initialize plot
    im0 = ax[0].imshow(np.log10(np.abs(frames_fft[0 : frames_fft.shape[0] // 2, :])))
    im1 = ax[1].imshow(np.log10(np.abs(frames_fft[frames_fft.shape[0] // 2 :, :])))
    scat0 = ax[0].scatter(
        frames_fft.shape[1] // 2, frames_fft.shape[0] // 4, color="red", marker="+"
    )
    scat1 = ax[1].scatter(
        frames_fft.shape[1] // 2, frames_fft.shape[0] // 4, color="red", marker="+"
    )
    fig.colorbar(im0, cax=cax0)
    fig.colorbar(im1, cax=cax1)
    ax[0].set_title("Positive frequencies")
    ax[1].set_title("Negative ferquencies")

    def animate(i):
        """Animation function for the window

        Args:
            i (int): Counter from the animate function
        """
        ret, frames[:, :] = cam.read()
        plan_rfft(frames, frames_fft)
        im0.set_data(np.log10(np.abs(frames_fft[0 : frames_fft.shape[0] // 2, :])))
        im1.set_data(np.log10(np.abs(frames_fft[frames_fft.shape[0] // 2 :, :])))
        return im0, im1, scat0, scat1

    # need to assign the animation to a variable or else it gets immediately
    # garbage collected
    anim = animation.FuncAnimation(
        fig, animate, interval=16, blit=True, cache_frame_data=False
    )
    plt.show(block=True)


def monitor_phase(cam: Any):
    """Displays a pop up window to monitor in live the phase"""
    frame = cam.capture().astype(np.float32)
    plan_rfft = pyfftw.builders.rfft2(frame)
    plan_ifft = pyfftw.builders.ifft2(
        frame[0 : frame.shape[0] // 2, 0 : frame.shape[1] // 2]
    )
    plans = (plan_rfft, plan_ifft)
    im_fringe = contrast.im_osc_fast_t(frame, cont=False, plans=plans)
    phase = contrast.angle_fast(im_fringe)
    cont = im_fringe.real * im_fringe.real + im_fringe.imag * im_fringe.imag
    fig, ax = plt.subplots(1, 2)
    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    divider1 = make_axes_locatable(ax[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    im0 = ax[0].imshow(phase, cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi)
    im1 = ax[1].imshow(cont, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im0, cax=cax0)
    fig.colorbar(im1, cax=cax1)
    ax[0].set_title("Phase")
    ax[1].set_title("amplitude")

    def animate(i):
        """Animation function for the window

        Args:
            i (int): Counter from the animate function
        """
        frame = cam.capture_video_frame()
        im_fringe[:, :] = contrast.im_osc_fast_t(frame, cont=False, plans=plans)
        phase[:, :] = contrast.angle_fast(im_fringe)
        cont[:, :] = im_fringe.real * im_fringe.real + im_fringe.imag * im_fringe.imag
        im0.set_data(phase)
        im1.set_data(cont)
        im1.set_clim(np.nanmin(cont), np.nanmax(cont))
        return im0, im1

    anim = animation.FuncAnimation(
        fig, animate, interval=0, blit=True, cache_frame_data=False
    )
    plt.show(block=True)
    cam.stop_video_capture()


def monitor_fourier_space(cam: any) -> None:
    """Displays a pop up window to monitor in live the Fourier space

    Args:
        cam (any): the camera object

    Returns:
        None
    """
    # grab first frame for reference
    ret, frame = cam.read()
    # allocate array to store current picture
    frames = pyfftw.empty_aligned(frame.shape, dtype=np.float32)
    frames[:, :] = frame
    # instantiate frame handler for asynchronous acquisition
    im_fringe = contrast.im_osc_fast_t(frames, cont=False)
    im_fft = np.abs(np.fft.fftshift(pyfftw.interfaces.numpy_fft.fft2(im_fringe)))
    fig, ax = plt.subplots(1, 2)
    divider0 = make_axes_locatable(ax[0])
    divider1 = make_axes_locatable(ax[1])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    im0 = ax[0].imshow(np.log10(im_fft), cmap="jet", vmax=4.8)
    im1 = ax[1].imshow(frames, cmap="viridis")
    fig.colorbar(im0, cax=cax0)
    fig.colorbar(im1, cax=cax1)
    ax[0].set_title("Fourier space")
    ax[1].set_title("Raw interferogram")

    def animate(i):
        """Animation function for the window

        Args:
            i (int): Counter from the animate function
        """
        ret, frames[:, :] = cam.read()
        im_fringe[:, :] = contrast.im_osc_fast_t(frames, cont=False)
        im_fft = np.abs(np.fft.fftshift(pyfftw.interfaces.numpy_fft.fft2(im_fringe)))
        im0.set_data(np.log10(im_fft))
        im1.set_data(frames)
        return im0, im1

    anim = animation.FuncAnimation(
        fig, animate, interval=16, blit=True, cache_frame_data=False
    )
    plt.show(block=True)


def monitor_abs_speed(cam: Any, y_range: int, r_calib=0.08 * 2) -> tuple:
    """Displays a pop up window to monitor in live the intensity of the field, a cut of the intensity and of the gradient of the phase
    the figure also displays a rectangle on top of the intensity image to show the zone where the cut and averaging are done.
    Some sliders are also available to change the position and size of the rectangle.

    Args:
        cam (any): the camera object
        y_range (int): the range of the cut and averaging
        r_calib (float): the calibration factor for the imaging system times 2 because of lose of spatial resolution in the fft
    Returns:
        frame (np.array): the last frame captured by the camera
        cont (np.array): the last intensity image compurted from frame
        phase (np.array): the last phase image computed from frame
    """
    # grab first frame for reference

    ret, frame = cam.read()
    im_fringe = contrast.im_osc_fast_t(frame, cont=False)

    phase = contrast.angle_fast(im_fringe)
    cont = np.abs(im_fringe)

    cut_x = phase.shape[1] // 2
    mean_range = 100
    cut_y = phase.shape[0] // 2

    unwr_phase = np.unwrap(phase, axis=0)
    mean_uwr_phase = np.mean(
        unwr_phase[
            cut_y - y_range // 2 : cut_y + y_range // 2,
            cut_x - mean_range // 2 : cut_x + mean_range // 2,
        ],
        axis=1,
    )
    mean_dens = np.mean(
        cont[
            cut_y - y_range // 2 : cut_y + y_range // 2,
            cut_x - mean_range // 2 : cut_x + mean_range // 2,
        ],
        axis=1,
    )

    speed = np.gradient(mean_uwr_phase) / r_calib

    # Creation of the figure
    fig = plt.figure(figsize=(10, 5))
    ax0 = fig.add_subplot(221)  # Subplot top left
    ax1 = fig.add_subplot(223)  # Subplot bottom right
    ax2 = fig.add_subplot(122)  # Subplot right
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)

    x_axis = np.arange(-y_range // 2, y_range // 2) * r_calib
    (im0,) = ax0.plot(x_axis, mean_dens)
    (im1,) = ax1.plot(x_axis, speed)
    im2 = ax2.imshow(cont, cmap="inferno")

    fig.colorbar(im2, cax=cax2)
    ax0.set_title("Mean density along y")
    ax1.set_title("Speed of the fluid along y")

    # Creates a rectangular patch highlighting the ROI on which the density and velocity are averaged
    x1, y1 = cut_x - mean_range // 2, cut_y - y_range // 2
    rectangle = plt.Rectangle(
        (x1, y1),
        mean_range,
        y_range,
        edgecolor="black",
        linestyle="--",
        linewidth=2,
        fill=False,
    )
    # Add the patch on top of the plot
    ax2.add_patch(rectangle)

    # Sliders to control rectangle location and size
    rect_x_slider_ax = fig.add_axes([0.1, 0.03, 0.65, 0.03])
    rect_y_slider_ax = fig.add_axes([0.1, 0.01, 0.65, 0.03])
    rect_width_slider_ax = fig.add_axes([0.1, 0.05, 0.65, 0.03])

    rect_x_slider = Slider(rect_x_slider_ax, "pos X", 0, phase.shape[1], valinit=x1)
    rect_y_slider = Slider(rect_y_slider_ax, "pos Y", 0, phase.shape[0], valinit=y1)
    rect_width_slider = Slider(
        rect_width_slider_ax, "Width", 0, phase.shape[1], valinit=mean_range
    )

    plt.subplots_adjust(top=0.95)

    def update(val):
        # Update location and size of the rectangle
        rectangle.set_xy((rect_x_slider.val, rect_y_slider.val))
        rectangle.set_width(rect_width_slider.val)
        fig.canvas.draw_idle()

    # Connect update function to the sliders
    rect_x_slider.on_changed(update)
    rect_y_slider.on_changed(update)
    rect_width_slider.on_changed(update)

    def animate(i: int) -> tuple:
        """Animation function for the window

        Args:
            i (int): Counter from the animate function

        Returns:
            im0, im1, im2, rectangle: the tuple containing the updated artists
        """
        ret, frame = cam.read()
        im_fringe[:, :] = contrast.im_osc_fast_t(frame, cont=False)
        x1, y1 = int(np.round(rect_x_slider.val)), int(np.round(rect_y_slider.val))
        mean_range = int(rect_width_slider.val)
        phase[:, :] = contrast.angle_fast(im_fringe)
        cont[:, :] = np.abs(im_fringe)
        unwr_phase = np.unwrap(phase, axis=0)
        mean_uwr_phase = np.mean(
            unwr_phase[y1 : y1 + y_range, x1 : x1 + mean_range], axis=1
        )
        mean_dens = np.mean(cont[y1 : y1 + y_range, x1 : x1 + mean_range], axis=1)
        speed = np.gradient(mean_uwr_phase) / r_calib
        im0.set_ydata(mean_dens)
        im1.set_ydata(speed)
        im2.set_data(cont)
        # Update rectangle's size and location
        rectangle.set_xy((rect_x_slider.val, rect_y_slider.val))
        rectangle.set_width(rect_width_slider.val)
        return im0, im1, im2, rectangle

    anim = animation.FuncAnimation(fig, animate, interval=33, blit=True)
    plt.show(block=True)

    return phase, cont, frame
