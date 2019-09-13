#!/usr/bin/env python
# Scripts for workshop on MD simulation of molten salts ans ionic liquid.
# Author: Takahiro Ohkubo <ohkubo.takahiro@faculty.chiba-u.jp>
# Version: 2019/09
# http://amorphous.tf.chiba-u.jp/MS%E8%AC%9B%E7%BF%92%E4%BC%9A/molten.html
# Copyright (C) 2019 Takahiro Ohkubo (Chiba university)

import ms_molten as lm
import numpy as np
import matplotlib.pyplot as plt
import argparse

par = argparse.ArgumentParser(description="test")
par.add_argument('files', nargs='+')
par.add_argument('-w', '--window', nargs=2, default=None, type=float)
par.add_argument('-n', '--norder', default=1, type=float)
par.add_argument('-m', '--mode', choices=['x', 'y', 'z', 'r'], default='r')
par.add_argument('--order', default=1, type=int)
par.add_argument('--dt', default=None, type=float)
par.add_argument('-o', '--outfile', default=None)
args = par.parse_args()

if args.order == 0:
    sfmt = "r2={:.3g} Ang2"
if args.order == 1:
    sfmt = "D={:.3g} m2/s"

# column, ylabel, coeff
dic = {}
dic['x'] = [1, '\overline{x^2}', 2]
dic['y'] = [2, '\overline{y^2}', 2]
dic['z'] = [3, '\overline{z^2}', 2]
dic['r'] = [4, '\overline{r^2}', 6]
col = dic[args.mode]
fig, ax = lm.makefig()
for f in args.files:
    lines = open(f).readlines()
    param = lm.getparam(lines[0])
    if ('dt' in param.keys()) is False and args.dt is None:
        print("dt is not set. (femto-second unit")
        print("plase use --dt option")
        exit()
    if args.dt is None:
        dt = param['dt']
    else:
        param['dt'] = args.dt
    d = np.array(" ".join(lines[3:]).split(), dtype=float)
    d = d.reshape((-1, 5))
    time, msd = d[:, 0] - d[0, 0], d[:, col[0]]
    time = time * param['dt'] * 1e-3  # fs -> ps
    p = ax.plot(time, msd, ".", label=f)
    if args.window is None:
        continue
    idx = lm.near(args.window, time)
    out = np.polyfit(time[idx], msd[idx], args.order)
    fit = np.polyval(out, time[idx])
    if args.order == 1:
        D = out[0] * 1e-8 / col[2]
        s = sfmt.format(out[0] * 1e-8 / col[2])
    if args.order == 0:
        s = sfmt.format(out[0])
    print("{}: {}".format(f, s))
    ax.plot(time[idx], fit, "-", color=p[0].get_color(), label=s)

ax.legend(loc="best")
ax.set_xlabel("${\\rm Time/ps}$")
ax.set_ylabel("${}/{{\\rm \AA^2}}$".format(col[1]))
if args.outfile is not None:
    fig.savefig(args.outfile)
    print(args.outfile, "was created.")
plt.show()
