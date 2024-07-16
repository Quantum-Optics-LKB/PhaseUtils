from PhaseUtils import contrast
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# cupy available logic
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def main():
    from PhaseUtils import velocity
    from cupyx.scipy import ndimage

    im = np.array(Image.open("../examples/dn_ref.tiff"))
    if CUPY_AVAILABLE:
        im_cp = cp.asarray(im)
        field = contrast.im_osc_fast_cp(im_cp)
        field_t = contrast.im_osc_fast_t_cp(im_cp)
        plt.imshow(cp.abs(field_t).get())
        plt.show()
    field = contrast.im_osc_fast(im)
    field_t = contrast.im_osc_fast_t(im)
    velo = velocity.velocity_cp(cp.asarray(np.angle(field_t)))
    velo_field = velocity.velocity_fft_cp(cp.asarray(field_t))
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(velo[0, :, :].get(), vmin=-0.5, vmax=0.5)
    ax[0, 1].imshow(velo[1, :, :].get(), vmin=-0.5, vmax=0.5)
    ax[0, 0].set_title("Vx")
    ax[0, 1].set_title("Vy")
    ax[1, 0].imshow(velo_field[0, :, :].get(), vmin=-0.5, vmax=0.5)
    ax[1, 1].imshow(velo_field[1, :, :].get(), vmin=-0.5, vmax=0.5)
    ax[1, 0].set_title("Vx field")
    ax[1, 1].set_title("Vy field")
    plt.show()
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow((velo[0] - velo_field[0]).get(), cmap="coolwarm", vmin=-0.5, vmax=0.5)
    ax[1].imshow((velo[1] - velo_field[1]).get(), cmap="coolwarm", vmin=-0.5, vmax=0.5)
    ax[0].set_title("Vx diff")
    ax[1].set_title("Vy diff")
    plt.show()
    velo, u_inc, u_comp = velocity.helmholtz_decomp_cp(cp.asarray(field), plot=True)
    ucc, uii = velocity.energy_spectrum_cp(u_comp, u_inc)
    fig, ax = plt.subplots()
    ax.plot(ucc.get(), label="Compressible")
    ax.plot(uii.get(), label="Incompressible")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel("E(k)")
    ax.legend()
    plt.show()
    field_t = contrast.im_osc_fast_t(im)
    contrast.im_osc(im, plot=True)
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(np.abs(field) ** 2)
    ax[0, 1].imshow(np.abs(field_t) ** 2)
    ax[0, 0].set_title("Density full")
    ax[0, 1].set_title("Density truncated")
    ax[1, 0].imshow(np.angle(field), cmap="twilight_shifted")
    ax[1, 1].imshow(np.angle(field_t), cmap="twilight_shifted")
    ax[1, 0].set_title("Phase full")
    ax[1, 1].set_title("Phase truncated")
    plt.show()
    rho = np.abs(field_t) ** 2
    x, y = contrast.centre(rho)
    wx, wy = contrast.waist(rho)
    print(f"Fitted waist: {wx:.2f}, {wy:.2f}")
    rho_avg = contrast.az_avg(rho, (x, y))
    fig, ax = plt.subplots(1, 2)
    fig.suptitle("Azimuthal average")
    ax[0].imshow(rho)
    ax[0].scatter(x, y, c="r", label="Fitted center")
    ax[0].legend()
    ax[1].plot(rho_avg)
    plt.show()
    if CUPY_AVAILABLE:
        im = cp.asarray(im)
        field = contrast.im_osc_fast_cp(im)
        field_t = contrast.im_osc_fast_t_cp(im)
        fig, ax = plt.subplots(2, 2)
        rho = cp.abs(field) ** 2
        rho_t = cp.abs(field_t) ** 2
        phi = cp.angle(field)
        phi_t = cp.angle(field_t)
        ax[0, 0].imshow(rho.get())
        ax[0, 1].imshow(rho_t.get())
        ax[0, 0].set_title("Density full")
        ax[0, 1].set_title("Density truncated")
        ax[1, 0].imshow(phi.get(), cmap="twilight_shifted")
        ax[1, 1].imshow(phi_t.get(), cmap="twilight_shifted")
        ax[1, 0].set_title("Phase full")
        ax[1, 1].set_title("Phase truncated")
        plt.show()
        rho_avg = contrast.az_avg_cp(rho, (x, y))
        fig, ax = plt.subplots(1, 2)
        fig.suptitle("Azimuthal average")
        ax[0].imshow(rho.get())
        ax[0].scatter(x, y, c="r", label="Fitted center")
        ax[0].legend()
        ax[1].plot(rho_avg.get())
        plt.show()


if __name__ == "__main__":
    main()
