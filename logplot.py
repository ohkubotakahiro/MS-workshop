#!/usr/bin/env python
# Scripts for workshop on MD simulation of molten salts ans ionic liquid.
# Author: Takahiro Ohkubo <ohkubo.takahiro@faculty.chiba-u.jp>
# Version: 2019/09
# http://amorphous.tf.chiba-u.jp/MS%E8%AC%9B%E7%BF%92%E4%BC%9A/molten.html
# Copyright (C) 2019 Takahiro Ohkubo (Chiba university)

import ms_molten as lm
import numpy as np
import matplotlib.pyplot as plt
import re
import argparse

description = """plotter for lammps log file"""
par = argparse.ArgumentParser(description="test")
par.add_argument('infile')
par.add_argument('-k', '--keywords', nargs='+', default=["TotEng"])
par.add_argument('-w', '--window', nargs=2, default=None, type=int)
par.add_argument('-o', '--outfile', default=None)
args = par.parse_args()


def GetData(infile):
    body = open(infile).read()
    data = re.findall("\n(Step.+?)\n([0-9 \.\-e\n]+)", body)
    keywords = np.array(data[0][0].split())
    dset = np.empty((0, len(keywords)))
    for i, d in enumerate(data):
        lines = d[1].split("\n")
        dd = " ".join(lines).split()
        amari = len(dd) % len(keywords)
        dd = dd[0:len(dd) - amari]
        dd = np.array(dd, dtype=float)
        dd = dd.reshape((-1, len(keywords)))
        dset = np.vstack((dset, dd))
    print("Total run: {}".format(len(data)))
    return keywords, dset


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def make_figure():
    fig, host = lm.makefig(minor=False, grid=False)
    ax = [host]
    for i in range(len(args.keywords) - 1):
        ax.append(host.twinx())
    if len(args.keywords) > 2:
        for i in range(2, len(args.keywords)):
            ax[i].spines["right"].set_position(("axes", 1 + (i-1)*0.20))
        make_patch_spines_invisible(ax[i])
        ax[i].spines["right"].set_visible(True)
        rmargin = 1.0 - (0.15 * (len(args.keywords) - 2))
        fig.subplots_adjust(right=rmargin)
    if len(args.keywords) == 1:
        ax[0].grid()
    return fig, ax


color = iter(['r', 'g', 'b', 'c', 'k', 'y'])
fig, ax = make_figure()
keywords, data = GetData(args.infile)
print("keywords: [{}]".format(", ".join(keywords)))
data[:, 0] = np.arange(data.shape[0])
for i, k in enumerate(args.keywords):
    p = (keywords == k)
    if np.sum(p) == 0:
        print("\"{}\" is not in keywords in {}".format(k, args.infile))
        exit()
    if args.window is not None:
        idx = lm.near(args.window, data[:, 0])
        ave = np.average(data[idx, p])
        std = np.std(data[idx, p])
        print("{} = {:g} +/- {:g}".format(k, ave, std))
    p, = ax[i].plot(data[:, 0], data[:, p], label=k, c=next(color))
    ax[i].set_ylabel("${{\\rm {}}}$".format(k))
    if len(args.keywords) > 1:
        ax[i].yaxis.label.set_color(p.get_color())
        tkw = dict(size=4, width=1.5)
        ax[i].tick_params(axis='y', colors=p.get_color(), **tkw)
if args.window is not None:
    s, e = idx[0], idx[-1]
    ax[0].axvspan(data[s, 0], data[e, 0], color='green', alpha=0.1)
ax[0].set_xlabel("${\\rm Step}$")

if args.outfile is not None:
    fig.savefig(args.outfile)
    print(args.outfile, "was created.")
plt.show()
