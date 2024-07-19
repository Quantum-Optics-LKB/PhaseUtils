import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PhaseUtils import SLM

im = np.array(Image.open("harambe.png"))[:, :, 0]
im = im / 255.0
im = SLM.error_diffusion_dithering(im, SLM.DIFFUSION_MAPS["jarvis-judice-ninke"])
plt.imshow(im, cmap="gray")
plt.show()
