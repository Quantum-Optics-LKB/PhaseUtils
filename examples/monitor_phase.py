import numpy as np
import pyfftw
import matplotlib.pyplot as plt
from matplotlib import animation
import multiprocessing
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import pickle
import contrast
import matplotlib.style as mplstyle
from typing import Any

mplstyle.use('fast')
plt.switch_backend('Qt5Agg')
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
        (frame.shape[0], frame.shape[1]//2+1), dtype=np.complex64)
    plan_rfft = pyfftw.builders.rfft2(frames,
                                        planner_effort="FFTW_PATIENT",
                                        threads=multiprocessing.cpu_count())
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
    im0 = ax[0].imshow(
        np.log10(np.abs(frames_fft[0:frames_fft.shape[0]//2, :])))
    im1 = ax[1].imshow(
        np.log10(np.abs(frames_fft[frames_fft.shape[0]//2:, :])))
    scat0 = ax[0].scatter(frames_fft.shape[1]//2, frames_fft.shape[0] //
                            4, color='red', marker='+')
    scat1 = ax[1].scatter(frames_fft.shape[1]//2, frames_fft.shape[0] //
                            4, color='red', marker='+')
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
        im0.set_data(
            np.log10(np.abs(frames_fft[0:frames_fft.shape[0]//2, :])))
        im1.set_data(
            np.log10(np.abs(frames_fft[frames_fft.shape[0]//2:, :])))
        return im0, im1, scat0, scat1
    # need to assign the animation to a variable or else it gets immediately
    # garbage collected
    anim = animation.FuncAnimation(fig, animate,
                                    interval=50, blit=True)
    plt.show(block=True)

def monitor_phase(cam: Any):
        """Displays a pop up window to monitor in live the phase
        """
        frame = cam.capture().astype(np.float32)
        plan_rfft = pyfftw.builders.rfft2(frame)
        plan_ifft = pyfftw.builders.ifft2(
            frame[0:frame.shape[0]//2, 0:frame.shape[1]//2])
        plans = (plan_rfft, plan_ifft)
        im_fringe = contrast.im_osc_fast_t(frame, cont=False, plans=plans)
        phase = contrast.angle_fast(im_fringe)
        cont = im_fringe.real*im_fringe.real + im_fringe.imag*im_fringe.imag
        fig, ax = plt.subplots(1, 2)
        divider0 = make_axes_locatable(ax[0])
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        divider1 = make_axes_locatable(ax[1])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        im0 = ax[0].imshow(phase, cmap='twilight_shifted',
                           vmin=-np.pi, vmax=np.pi)
        im1 = ax[1].imshow(cont,
                           cmap='viridis', vmin=0, vmax=1)
        fig.colorbar(im0, cax=cax0)
        fig.colorbar(im1, cax=cax1)
        ax[0].set_title("Phase")
        ax[1].set_title("amplitude")

        def animate(i):
            """Animation function for the window

            Args:
                i (int): Counter from the animate function
            """
            frame = self.cam_real.capture_video_frame(buffer_=frame_buf)
            im_fringe[:, :] = contrast.im_osc_fast_t(
                frame, cont=False, plans=plans)
            phase[:, :] = contrast.angle_fast(im_fringe)
            cont[:, :] = im_fringe.real*im_fringe.real + im_fringe.imag*im_fringe.imag
            im0.set_data(phase)
            im1.set_data(cont)
            im1.set_clim(np.nanmin(cont), np.nanmax(cont))
            return im0, im1
        anim = animation.FuncAnimation(fig, animate,
                                       interval=0, blit=True,
                                       cache_frame_data=False)
        plt.show(block=True)
        self.cam_real.stop_video_capture()
        self.toggle_ref()

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
    im0 = ax[0].imshow(np.log10(im_fft), cmap='jet', vmax=4.8)
    im1 = ax[1].imshow(frames, cmap='viridis')
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
    anim = animation.FuncAnimation(fig, animate,
                                    interval=33, blit=True)
    plt.show(block=True)
