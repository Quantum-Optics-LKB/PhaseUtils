from PhaseUtils import velocity
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import time
from typing import Callable

# cupy available logic
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
# simple timing decorator


def timer(func: Callable):
    """This function shows the execution time of
    the function object passed"""

    def wrap_func(*args, **kwargs):
        t1 = time.perf_counter()
        result = func(*args, **kwargs)
        t2 = time.perf_counter()
        print(f"Function {func.__name__!r} executed in {(t2-t1)*1e3:.4f} ms")
        return result

    return wrap_func


def timer_repeat(func: Callable, *args, N_repeat=1000):
    t = np.zeros(N_repeat, dtype=np.float32)
    for i in range(N_repeat):
        t0 = time.perf_counter()
        func(*args)
        t1 = time.perf_counter()
        t[i] = t1 - t0
    print(
        f"{N_repeat} executions of {func.__name__!r} "
        f"{np.mean(t)*1e3:.3f} +/- {np.std(t)*1e3:.3f} ms per loop (min : {np.min(t)*1e3:.3f} / max : {np.max(t)*1e3:.3f} ms / med: {np.median(t)*1e3:.3f})"
    )
    return np.mean(t), np.std(t)


if CUPY_AVAILABLE:

    def timer_repeat_cp(func: Callable, *args, N_repeat=1000):
        t = np.zeros(N_repeat, dtype=np.float32)
        for i in range(N_repeat):
            start_gpu = cp.cuda.Event()
            end_gpu = cp.cuda.Event()
            start_gpu.record()
            func(*args)
            end_gpu.record()
            end_gpu.synchronize()
            t[i] = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
        print(
            f"{N_repeat} executions of {func.__name__!r} "
            f"{np.mean(t):.3f} +/- {np.std(t):.3f} ms per loop (min : {np.min(t):.3f} / max : {np.max(t):.3f} ms / med: {np.median(t):.3f})"
        )
        return np.mean(t), np.std(t)


def main() -> None:
    N_repeat = 5
    phase = np.loadtxt("../examples/v500_1_phase.txt")
    vortices = velocity.vortex_detection(phase, plot=True, r=1)
    timer_repeat(velocity.vortex_detection, phase, N_repeat=N_repeat)
    # Velocity decomposition in incompressible and compressible
    velo, v_inc, v_comp = velocity.helmholtz_decomp(phase, plot=False)
    timer_repeat(velocity.helmholtz_decomp, phase, N_repeat=N_repeat)
    # Clustering benchmarks
    dipoles, clusters, cluster_graph = velocity.cluster_vortices(vortices)
    timer_repeat(velocity.cluster_vortices, vortices, N_repeat=N_repeat)
    # Plot results
    fig, (ax, ax1) = plt.subplots(1, 2)
    YY, XX = np.indices(v_inc[0].shape)
    im = ax.imshow(np.hypot(v_inc[0], v_inc[1]), cmap="viridis")
    ax.streamplot(XX, YY, v_inc[0], v_inc[1], density=5, color="white", linewidth=1)
    positions = dict(zip(cluster_graph.nodes, vortices[cluster_graph.nodes, 0:2]))
    edge_color = []
    for edge in cluster_graph.edges:
        if vortices[edge[0], 2] == 1:
            edge_color.append("r")
        else:
            edge_color.append("b")
    nx.draw(
        cluster_graph,
        ax=ax,
        pos=positions,
        with_labels=False,
        node_size=45,
        node_color=vortices[cluster_graph.nodes, 2],
        cmap="bwr",
        edge_color=edge_color,
        width=2,
    )
    for dip in range(dipoles.shape[0]):
        (ln_d,) = ax.plot(
            vortices[dipoles[dip, :], 0],
            vortices[dipoles[dip, :], 1],
            color="g",
            marker="o",
            label="Dipoles",
        )
    ax.set_title(r"Incompressible velocity $|v^{inc}|$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(0, phase.shape[1])
    ax.set_ylim(0, phase.shape[0])
    ax.legend(handles=[ln_d])
    plt.colorbar(im, ax=ax)
    bins, edges = velocity.cluster_histogram(clusters, plot=False)
    ax1.stairs(bins, edges, fill=True)
    ax1.set_xlabel("Size of cluster")
    ax1.set_ylabel("Number of vortices")
    ax1.set_xlim(1, np.max(edges))
    ax1.set_yscale("log")
    ax1.set_title("Histogram of cluster size")
    plt.show()
    if CUPY_AVAILABLE:
        bins = cp.linspace(0, phase.shape[0], 100)
        phase_cp = cp.asarray(phase)
        # Vortex detection step
        vortices_cp = velocity.vortex_detection_cp(phase_cp, r=1, plot=True)
        corr = velocity.pair_correlations_cp(vortices_cp, bins)
        plt.plot(bins[1:].get(), corr.get())
        plt.show()
        velo, v_inc, v_comp = velocity.helmholtz_decomp_cp(phase_cp, plot=False)
        timer_repeat_cp(velocity.vortex_detection_cp, phase_cp, N_repeat=N_repeat)
        timer_repeat_cp(velocity.helmholtz_decomp_cp, phase_cp, N_repeat=N_repeat)


if __name__ == "__main__":
    main()
