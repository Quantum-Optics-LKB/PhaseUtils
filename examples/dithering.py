import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PhaseUtils import SLM

# Small example of dithering and low pass filtering in the
# context of DMD display.
# A DMD is a binary display, so we need to dither the image in order to
# get the "illusion" of several gray levels (as for posters in the metro)

im0 = np.array(Image.open("harambe.png"))[:, :, 0]
im0 = im0 / 255.0
im = SLM.error_diffusion_dithering(
    im0.copy(), SLM.DIFFUSION_MAPS["jarvis-judice-ninke"]
)
ky = np.fft.fftfreq(im.shape[0])
kx = np.fft.fftfreq(im.shape[1])
kxx, kyy = np.meshgrid(kx, ky)
k = np.hypot(kxx, kyy)
im_lp = np.fft.fft2(im, norm="ortho")
im_lp[k > 0.2] = 0
im_lp = np.fft.ifft2(im_lp, norm="ortho")
fig, ax = plt.subplots(1, 3)
ax[0].imshow(im0, cmap="gray", interpolation="none")
ax[0].set_title("Original Image")
ax[1].imshow(im, cmap="gray", interpolation="none")
ax[1].set_title("Dithered Image")
ax[2].imshow(np.abs(im_lp), cmap="gray", interpolation="none")
ax[2].set_title("Low Pass Filtered Image")
plt.show()
