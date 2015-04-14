#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
from PIL import Image

from pcp import pcp


def bitmap_to_mat(bitmap_seq):
    """from blog.shriphani.com"""
    matrix = []
    shape = None
    for bitmap_file in bitmap_seq:
        img = Image.open(bitmap_file).convert("L")
        if shape is None:
            shape = img.size
        assert img.size == shape
        img = np.array(img.getdata())
        matrix.append(img)
    print(matrix)
    return np.array(matrix), shape[::-1]


def do_plot(ax, img, shape):
    ax.imshow(img.reshape(shape), cmap="gray", interpolation="nearest")


if __name__ == "__main__":
    import sys
    import glob
    import matplotlib.pyplot as pl

    if "--test" in sys.argv:
        M = (10*np.ones((10, 10))) + (-5 * np.eye(10))
        L, S, svd = pcp(M, verbose=True, l=10)
        assert np.allclose(M, L + S), "Failed"
        print("passed")
        sys.exit(0)

    M, shape = bitmap_to_mat(glob.glob("test_data/Escalator/*.bmp")[:30])

    fig, axes = pl.subplots(1, 3, sharex=True, sharey=True)
    for i in range(len(M)):
        do_plot(axes[0], M[i], shape)
        fig.savefig("results/{0:05d}.png".format(i))

    # fig = do_plot(M[0], shape)
    # fig.savefig("plot-initial.png")

    # L, S, (u, s, v) = pcp(M, maxiter=1, verbose=True, delta=1e-4)
    # fig = do_plot(L[0], shape)
    # fig.savefig("plot-L.png")

    # fig = do_plot(S[0], shape)
    # fig.savefig("plot-S.png")

    # fig = do_plot(v[0], shape)
    # fig.savefig("plot-eigen.png")
