from PhaseUtils.SLM import (
    grating,
    phase_amplitude,
    mgrid,
    bragg_density_profile,
    SLMscreen,
)


def main():
    # import sys
    import time
    import numpy as np
    import sys

    # from PIL import Image
    resX, resY = 1920, 1080
    slm = SLMscreen(0)
    T = np.zeros(20)
    for i in range(20):
        sys.stdout.flush()
        one = np.ones((resY, resX), dtype=np.uint8)
        slm_pic = (i % 2) * one[:, 0 : resX // 2] + ((i + 1) % 2) * 255 * one[
            :, resX // 2 :
        ]
        slm_pic = np.random.choice([0, 255], size=(resY, resX)).astype(np.uint8)
        t0 = time.time()
        slm.update(slm_pic, delay=1)
        t = time.time() - t0
        T[i] = t
        sys.stdout.write(f"\r{i+1} : time displayed = {t} s")
    slm.close()

    print(f"\nAverage display time = {np.mean(T)} ({np.std(T)}) s")
    grat = grating(resY, resX)
    x = np.random.random((resY, resX)).astype(np.float32)
    y = np.random.random((resY, resX)).astype(np.float32)
    phase_amplitude(x, y, grat)
    mgrid(resY, resX)
    grating(resY, resX)
    bragg_density_profile(resY, resX, 1e4, alpha=0.1, width=200)


if __name__ == "__main__":
    main()
