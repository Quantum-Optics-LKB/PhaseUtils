import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from PhaseUtils.SLM import phase_amplitude, grating
from PIL import Image
from tqdm import tqdm

harambe = np.array(Image.open("harambe.png"))[:, :, 0].astype(np.float64)
kitty = np.array(Image.open("kitty.tif")).astype(np.float64)
kitty = kitty / np.nanmax(kitty)
Y, X = np.mgrid[0 : harambe.shape[0], 0 : harambe.shape[1]]
X -= harambe.shape[1] // 2
Y -= harambe.shape[0] // 2
# phi_target = 0.2*X
phi_target = np.ones(harambe.shape)
i_target = harambe / np.nanmax(harambe)
a_target = np.sqrt(harambe)
theta = 90
slm_pitch = 8
grating = grating(harambe.shape[0], harambe.shape[1], theta=theta, pitch=slm_pitch)
phi_slm = a_target * (2 * np.pi * grating - phi_target) % (2 * np.pi)
I_slm = np.ones(harambe.shape)
angle_x = np.sin(theta * np.pi / 180) * grating.shape[1] / slm_pitch
angle_y = np.cos(theta * np.pi / 180) * grating.shape[0] / slm_pitch
Y, X = np.mgrid[0 : grating.shape[0], 0 : grating.shape[1]]
# circle
filt = (
    np.hypot(
        X - (grating.shape[1] // 2 + angle_x), Y - (grating.shape[0] // 2 + angle_y)
    )
    < np.hypot(angle_x, angle_y) / 2
)

# diffraction efficiency
eps = 2e-2


def field_after_4f_filtering(I_slm: np.ndarray, phi_slm: np.ndarray, filt: np.ndarray):
    im_fft = np.fft.fftshift(
        np.fft.fft2((1 - eps) * I_slm * np.exp(1j * phi_slm) + eps * I_slm)
    )
    im_fft[im_fft.shape[0] // 2, im_fft.shape[1] // 2] = 0
    # filter
    im_fft *= filt
    # roll to remove off-axis component
    im_fft = np.roll(
        im_fft, (-int(np.round(angle_y)), -int(np.round(angle_x))), axis=(0, 1)
    )
    # back transform
    im_ifft = np.fft.ifft2(np.fft.ifftshift(im_fft))
    return im_ifft


field = field_after_4f_filtering(I_slm, phi_slm, filt)
rho = np.abs(field) ** 2
rho /= np.nanmax(rho)
phi = np.angle(field)
fig, ax = plt.subplots(2, 2)
ax[0, 0].set_title("Intensity")
im00 = ax[0, 0].imshow(rho, cmap="gray")
fig.colorbar(im00, ax=ax[0, 0], shrink=0.6, label="Normalized Intensity")
ax[0, 1].set_title("Phase")
im01 = ax[0, 1].imshow(phi, cmap="twilight_shifted")
fig.colorbar(im01, ax=ax[0, 1], shrink=0.6, label="Phase (rad)")
ax[1, 0].set_title("Error Intensity")
im10 = ax[1, 0].imshow(
    i_target - rho, cmap="coolwarm", norm=colors.CenteredNorm(vcenter=0)
)
fig.colorbar(im10, ax=ax[1, 0], shrink=0.6)
ax[1, 1].set_title("Error Phase")
im11 = ax[1, 1].imshow(
    phi_target - phi, cmap="coolwarm", norm=colors.CenteredNorm(vcenter=0)
)
fig.colorbar(im11, ax=ax[1, 1], shrink=0.6)
plt.show()
# try to optimize the result
N_tries = 100
mix = 0.5
noise = 0
error = np.zeros(N_tries)
for i in tqdm(range(N_tries), desc="Optimizing"):
    field = field_after_4f_filtering(I_slm, phi_slm, filt)
    rho = np.abs(field) ** 2
    rho /= np.nanmax(rho)
    phi = np.angle(field)
    err_phi = phi_target - phi
    err_rho = i_target - rho
    error[i] = (1 - np.sum(np.sqrt(rho * i_target) * np.cos(phi - phi_target))) ** 2
    phi = phi_target + mix * err_phi + noise * np.random.random(phi.shape)
    rho = i_target + mix * err_rho + noise * np.random.random(rho.shape)
    rho[rho < 0] = 0
    a = np.sqrt(rho)
    phi_slm = a * (2 * np.pi * grating - phi) % (2 * np.pi)
field = field_after_4f_filtering(I_slm, phi_slm, filt)
rho = np.abs(field) ** 2
phi = np.angle(field)
fig, ax = plt.subplots(1, 3)
ax[0].set_title("Intensity")
ax[0].imshow(rho, cmap="gray")
ax[1].set_title("Phase")
ax[1].imshow(phi, cmap="twilight_shifted")
ax[2].set_title("Error")
ax[2].plot(error, label="Error", marker="o")
ax[2].legend()
ax[2].set_xlabel("Iteration")
ax[2].set_ylabel("Error")
ax[2].set_yscale("log")
ax[2].set_xscale("log")
plt.show()
