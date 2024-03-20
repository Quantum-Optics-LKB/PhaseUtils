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


if __name__ == "__main__":
    main()
