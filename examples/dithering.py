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
im = SLM.error_diffusion_dithering(im0.copy(), SLM.DIFFUSION_MAPS["floyd-steinberg"])
ky = np.linspace(-im.shape[0] / 2, im.shape[0] / 2, im.shape[0]) * 1 / max(im.shape)
kx = np.linspace(-im.shape[1] / 2, im.shape[1] / 2, im.shape[1]) * 1 / max(im.shape)
kx = np.fft.fftshift(kx)
ky = np.fft.fftshift(ky)
kxx, kyy = np.meshgrid(kx, ky)
k = np.hypot(kxx, kyy)
im_lp = np.fft.fft2(im, norm="ortho")
# Hand tuned for best mse
im_lp[k > 0.13] = 0
im_lp = np.abs(np.fft.ifft2(im_lp, norm="ortho"))
err = im_lp - im0
mse = np.mean(err**2)
fig, ax = plt.subplots(2, 2, figsize=(10, 10), layout="constrained")
ims = []
ims.append(ax[0, 0].imshow(im0, cmap="gray", interpolation="none"))
ax[0, 0].set_title("Original Image")
ims.append(ax[0, 1].imshow(im, cmap="gray", interpolation="none"))
ax[0, 1].set_title("Dithered Image")
ims.append(ax[1, 0].imshow(im_lp, cmap="gray", interpolation="none"))
ax[1, 0].set_title("Low Pass Filtered Image")
ims.append(ax[1, 1].imshow(err, cmap="seismic", interpolation="none"))
ax[1, 1].set_title(
    f"Error : MSE = {mse*1e2:.3f} % / max = {np.abs(err).max()*1e2:.2f} %"
)
for i, a in enumerate(ax.flat):
    fig.colorbar(ims[i], ax=a, shrink=0.6)
plt.show()
