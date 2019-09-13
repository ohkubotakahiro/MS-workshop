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
par.add_argument('infile')
par.add_argument('-s', '--shift', metavar=1, default=1, type=int,
                 help="time shift for correlation function")
par.add_argument('-w', '--window', metavar=0.4, default=0.4, type=float,
                 help="correlation time window in ps")
par.add_argument('-o', '--outfile', default=None,
                 help="output image file")
par.add_argument('--dt', metavar=None, default=None, type=float,
                 help="simulation time step in fs")
args = par.parse_args()


def GetData(infile):
    lines = open(infile).readlines()
    param = lm.getparam(lines[0])
    if ('dt' in param.keys()) is False and args.dt is None:
        print("timestep (dt) is not set. (femto-second unit")
        print("plase use --dt option")
    data = " ".join(open(args.infile).readlines()[1:]).split()
    data = np.array(data, dtype=float).reshape((-1, 4))
    timestep = data[1, 0] - data[0, 0]
    data[:, 0] = data[:, 0] * param['dt'] * 1e-3  # ps
    param['dt'] = param['dt'] * timestep
    param['wsize'] = int(args.window/param['dt'] * 1e3)
    return param, data


def PlotData(param, corr):
    fig, ax = lm.makefig()
    bx = ax.twinx()
    t = np.arange(param['wsize']) * param['dt'] * 1e-3  # ps
    ax.plot(t, corr[:, 0], label="$x$")
    ax.plot(t, corr[:, 1], label="$y$")
    ax.plot(t, corr[:, 2], label="$z$")
    ax.plot(t, np.mean(corr, axis=1), "k-", label="$r$")
    ax.legend(loc="best")
    ax.set_xlabel('${\\rm Time/ps}$')
    fmt = '$J(t)J(t+\\!\Delta t)\ {{\\rm {}}}$'
    ax.set_ylabel(fmt.format(param['unit1']))
    fmt = "$\{}={{{:.3g}}}\ {{\\rm {}}}$"
    val = fmt.format(param['symbol'], param['val'], param['unit2'])
    ax.text(0.5, 0.5, val, size=16, transform=ax.transAxes)
    val = np.cumsum(corr, axis=0) * param['dt'] * param['scale']
    bx.plot(t, val[:, 0], "--")
    bx.plot(t, val[:, 1], "--")
    bx.plot(t, val[:, 2], "--")
    bx.plot(t, np.mean(val, axis=1), "k--")
    fmt = '$\{}\ {{\\rm {}}}$'
    bx.set_ylabel(fmt.format(param['symbol'], param['unit2']))
    bx.grid()
    if args.outfile is not None:
        fig.savefig(args.outfile)
    plt.show()


param, data = GetData(args.infile)
corr = np.empty((param['wsize'], 3))
idx = np.arange(data.shape[0], dtype=int)
for i in range(corr.shape[0]):
    idx_t = idx + i
    idx_t = idx_t[idx_t < data.shape[0]]
    idx_0 = idx[:idx_t.shape[0]]
    idx_0, idx_t = idx_0[::args.shift], idx_t[::args.shift]
    corr[i, 0:3] = np.mean(data[idx_0, 1:4] * data[idx_t, 1:4], axis=0)

if param['mode'] == 'Kappa_calc.':
    param['convert'] = lm.kCal2J**2/lm.fs2s/lm.A2m  # kcal/mol/Ang/K -> W/m/K
    param['scale'] = param['convert']/lm.kB/param['T']**2*param['V']
    param['symbol'] = 'kappa'
    param['unit1'] = 'kcal/mol/\AA^2/fs'
    param['unit2'] = 'W/m/K'
if param['mode'] == 'Eta_calc.':
    param['convert'] = lm.atm2Pa**2 * lm.fs2s * lm.A2m**3
    param['scale'] = param['convert']/lm.kB/param['T']*param['V']
    param['symbol'] = 'eta'
    param['unit1'] = 'Pa'
    param['unit2'] = 'Pa\cdot s'


print("total simulation period: {} ps".format(data[-1, 0]))
print("correlation time window: {} ps".format(args.window))
print("correlation time window: {} steps".format(param['wsize']))
    
val = np.empty(3)
val[0] = np.trapz(corr[:, 0], dx=param['dt']) * param['scale']
val[1] = np.trapz(corr[:, 1], dx=param['dt']) * param['scale']
val[2] = np.trapz(corr[:, 2], dx=param['dt']) * param['scale']
print("{}    = {:g} {}".format(param['symbol'], np.mean(val), param['unit2']))
print("{}_xx = {:g} {}".format(param['symbol'], val[0], param['unit2']))
print("{}_yy = {:g} {}".format(param['symbol'], val[1], param['unit2']))
print("{}_zz = {:g} {}".format(param['symbol'], val[2], param['unit2']))
param['val'] = np.mean(val)
PlotData(param, corr)
