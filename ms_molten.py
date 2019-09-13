#!/usr/bin/env python
# Scripts for workshop on MD simulation of molten salts ans ionic liquid.
# Author: Takahiro Ohkubo <ohkubo.takahiro@faculty.chiba-u.jp>
# Version: 2019/09
# http://amorphous.tf.chiba-u.jp/MS%E8%AC%9B%E7%BF%92%E4%BC%9A/molten.html
# Copyright (C) 2019 Takahiro Ohkubo (Chiba university)

import matplotlib.pyplot as plt
import numpy as np
import re

kB = 1.3806503e-23
NA = 6.02214129e23
kCal2J = 4186.0/NA
A2m = 1.0e-10
fs2s = 1.0e-15
atm2Pa = 101325.0


def makefig(minor=True, grid=True):
    # 図の体裁
    plt.style.use('classic')
    plt_dic = {}
    plt_dic['legend.fancybox'] = True
    plt_dic['legend.labelspacing'] = 0.3
    plt_dic['legend.numpoints'] = 1
    plt_dic['figure.figsize'] = [8, 6]
    plt_dic['font.size'] = 12
    plt_dic['legend.fontsize'] = 12
    plt_dic['axes.labelsize'] = 16
    plt_dic['xtick.major.size'] = 5
    plt_dic['xtick.minor.size'] = 2
    plt_dic['ytick.major.size'] = 5
    plt_dic['ytick.minor.size'] = 2
    plt_dic['xtick.direction'] = 'in'
    plt_dic['savefig.bbox'] = 'tight'
    plt_dic['savefig.dpi'] = 150
    plt_dic['savefig.transparent'] = False
    plt_dic['axes.grid'] = grid
    plt.rcParams.update(plt_dic)
    fig, ax = plt.subplots()
    if grid is True:
        plt_dic['axes.grid'] = True
    if minor is True:
        from matplotlib.ticker import AutoMinorLocator
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    return fig, ax


def near(windows, vlist):
    w = np.empty(2)
    w[0] = np.argmin(np.abs(vlist - windows[0]))
    w[1] = np.argmin(np.abs(vlist - windows[1]))
    w = np.sort(w)
    idx = np.arange(w[0], w[1]+1, 1, dtype=int)
    if idx.shape[0] == 0:
        print("No window data, change value of window option (-w).")
        exit(0)
    return idx


def getparam(line):
    param = {}
    param['PAIRS'] = None
    param['ELEMENTS'] = None
    val = re.findall("([A-Za-z]+)=([0-9\.\-e\+]+)+?", line)
    obj = re.search("([A-Za-z]+)=([A-Za-z\-\,]+)", line)
    if obj is not None:
        k, v = obj.group(1), obj.group(2).split(',')
        param[k] = v
    for k in val:
        param[k[0]] = float(k[1])
    obj = re.search("\[(.+?)\]", line)
    if obj is not  None:
        param['mode'] = obj.group(1)
    return param


if __name__ == "__main__":
    pass
