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
    im = np.array(Image.open("../examples/dn_ref.tiff"))
    field = contrast.im_osc_fast(im)
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
