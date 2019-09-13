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
par.add_argument('-o', '--outfile', default=None)
args = par.parse_args()


def getdata(f):
    lines = open(f).readlines()
    if len(lines) == 3:
        print("No data in {}".format(f))
        return [], []
    param = lm.getparam(lines[0])
    nrows = int(lines[3].split()[-1])
    ncols = len(lines[4].split())
    body = np.array(lines[3:]).reshape((-1, nrows+1))
    head = np.array(" ".join(body[:, 0]).split())
    head = head.reshape((body.shape[0], -1)).astype(int)
    data = np.array(" ".join(body[:, 1:].reshape(-1)).split())
    data = data.reshape((body.shape[0], nrows, ncols)).astype(float)
    data = np.average(data, axis=0)
    pairs = int((data.shape[1] - 2)*0.5)
    elems = int((-1 + np.sqrt(1 + 8 * pairs)) * 0.5)
    if param['PAIRS'] is None or len(param['PAIRS']) != pairs:
        param['PAIRS'] = []
        for i in range(elems):
            for j in range(i, elems):
                param['PAIRS'].append("{}-{}".format(i+1, j+1))
    labels = param['PAIRS']
    print("file: {}".format(f))
    print("dataset: {}".format(head.shape[0]))
    return labels, data[:, 1:]


fig, ax = lm.makefig(minor=False)
bx = ax.twinx()
for f in args.files:
    labels, data = getdata(f)
    for i, l in enumerate(labels):
        print("{} pair: {}".format(f, l))
        label = "{}({})".format(f, l)
        ax.plot(data[:, 0], data[:, 2*i + 1], label=label)
        bx.plot(data[:, 0], data[:, 2*i + 2])

ax.legend(loc="best")
ax.set_xlabel("${\\rm Distance/\AA}$")
ax.set_ylabel("$G(r)$")
bx.grid(False)
bx.set_ylabel("$CN(r)$")
bx.set_ylim(0, 20)
if args.outfile is not None:
    fig.savefig(args.outfile)
    print(args.outfile, "was created.")
plt.show()
