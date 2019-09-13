#!/usr/bin/env python
# Scripts for workshop on MD simulation of molten salts ans ionic liquid.
# Author: Takahiro Ohkubo <ohkubo.takahiro@faculty.chiba-u.jp>
# Version: 2019/09
# http://amorphous.tf.chiba-u.jp/MS%E8%AC%9B%E7%BF%92%E4%BC%9A/molten.html
# Copyright (C) 2019 Takahiro Ohkubo (Chiba university)

import numpy as np
import argparse
import re
import ms_template as template

ATOMS = """
element  oxide  charge    mass
O        None   -0.945    15.9994
Si       SiO2    1.89     28.0855
B        B2O3    1.4175   10.811
Ca       CaO     0.945    40.078
Na       Na2O    0.4725   22.989768
Ti       TiO2    1.89     47.88
Al       Al2O3   1.4175   26.981539
Fe3+     Fe2O3   1.4175   55.847
Fe2+     FeO     0.945    55.847
Mg       MgO     0.945    24.305
K        K2O     0.4725   39.0983
"""

PAIR_FF = """
Bond       Aij (eV)   rhoij(Ang.)   Cij (eV Ang6)
O-O        9022.79    0.265	    85.0921
Si-O      50306.10    0.161	    46.2978
B-O      206941.81    0.124	    35.0018
B-B         484.40    0.35	     0.0
Si-B        337.70    0.29	     0.0
Na-O     120303.80    0.17	     0.0
Ca-O     155667.70    0.178	    42.2597
Ti-O      50126.64    0.178	    46.2978
Al-O      28538.42    0.172	    34.5778
Fe3+-O     8020.27    0.19	     0.0
Fe2+-O    13032.93    0.19	     0.0
Mg-O      32652.64    0.178	    27.2810
K-O        2284.77    0.29           0.0
"""

PAIRS = np.array(ATOMS.strip().split()).reshape((-1, 4))[2:, 1]
par = argparse.ArgumentParser(description="test")
for p in PAIRS:
    lopt = "--{}".format(p)
    shelp = "number of {}".format(p)
    par.add_argument(lopt, metavar='N', default=0, type=int, help=shelp)
par.add_argument('--density', metavar=2.0, default=2.0, type=float,
                 help="target density g/cm3")
par.add_argument('--deform', metavar=1, default=1, type=float,
                 help="expand scale for deformation of initial cell length")
par.add_argument('-o', '--outfile', required=True,
                 help="basename for output files")
args = par.parse_args()


class LammpsData():
    def __init__(self):
        self.systemname = ""
        self.L = 0.0
        self.target_density = args.density

    def LoadAtomsFromArg(self):
        # 0:elem_a 1:elem_b 2:molname 3:num
        atoms = np.empty((0, 4), dtype=object)
        atoms = {}
        atoms['O'] = 0
        systemname = []
        for k, value in par.parse_args()._get_kwargs():
            if (k in PAIRS) is False or (value == 0):
                continue
            systemname += ["{}{}".format(value, k)]
            cation_n, anion_n = 1, 1
            obj = re.search("([A-Z][a-z]*)(\d*)([A-Z][a-z]*)(\d*)", k)
            cation = obj.group(1)
            anion = obj.group(3)
            if obj.group(2) != "":
                cation_n = int(obj.group(2))
            if obj.group(4) != "":
                anion_n = int(obj.group(4))
            if cation == "Fe" and cation_n == 2:
                cation = "Fe3+"
            if cation == "Fe" and cation_n == 1:
                cation = "Fe2+"
            atoms[cation] = cation_n * value
            atoms[anion] += anion_n * value
        self.systemname = "-".join(systemname)
        elem_order = ["Si", "B", "Ti", "Al", "Fe3+", "Fe2+"]
        elem_order += ["Ca", "Mg", "K", "Na"]
        elem_order += ["O"]  # Oは必ず最後にもってくるようにする。
        atoms_ = {}
        for elem in elem_order:
            if elem in atoms.keys():
                atoms_[elem] = atoms[elem]
        self.atomdata = atoms_

    def SetAtoms(self, lmpdata):
        weight = 0
        elems = []
        for k, v in self.atomdata.items():
            weight += FF_ATOMS[k][1] * v
            elems += [k] * v
        nm = (len(elems), lmpdata.atoms.shape[1])
        lmpdata.atoms = np.empty(nm, dtype=object)
        lmpdata.atoms[:, 0] = np.arange(1, lmpdata.atoms.shape[0] + 1)
        lmpdata.atoms[:, 1] = np.arange(1, lmpdata.atoms.shape[0] + 1)
        lmpdata.atoms[:, 7] = elems
        lmpdata.atoms[:, 8] = elems
        lmpdata.atoms[:, 9] = elems
        elems_u = self.atomdata.keys()
        nm = (len(elems_u), lmpdata.atom_coeffs.shape[1])
        lmpdata.atom_coeffs = np.empty(nm, dtype=object)
        for i, e in enumerate(elems_u):
            flag = lmpdata.atoms[:, 7] == e
            lmpdata.atoms[flag, 2] = i + 1  # type
            lmpdata.atoms[flag, 3] = FF_ATOMS[e][0]  # charge
            lmpdata.atom_coeffs[i, 0] = i + 1  # type
            lmpdata.atom_coeffs[i, 1] = e  # type
            lmpdata.atom_coeffs[i, 2] = e  # symbol
            lmpdata.atom_coeffs[i, 3] = FF_ATOMS[e][1]  # mass
            lmpdata.atom_coeffs[i, 4] = FF_ATOMS[e][0]  # charge
            lmpdata.atom_coeffs[i, 7] = 0.0
            lmpdata.atom_coeffs[i, 8] = e
            lmpdata.atom_coeffs[i, 9] = ""  # molname

    def SetXYZ(self, lmpdata):
        L = lmpdata.a
        n = np.ceil(lmpdata.atoms.shape[0] ** (1.0/3.0)).astype(int)
        xi = np.linspace(0, L, n, endpoint=False)
        yi = np.linspace(0, L, n, endpoint=False)
        zi = np.linspace(0, L, n, endpoint=False)
        x, y, z = np.meshgrid(xi, yi, zi, indexing='ij')
        grid = x.reshape(-1)
        grid = np.vstack((grid, y.reshape(-1)))
        grid = np.vstack((grid, z.reshape(-1)))
        grid = grid.T
        idx = np.arange(grid.shape[0])
        np.random.shuffle(idx)
        idx = idx[:lmpdata.atoms.shape[0]]
        lmpdata.atoms[:, 4:7] = grid[idx, 0:3]

    def SetPairIJ(self, lmpdata):
        self.pairij = "\nPairIJ Coeffs\n\n"
        fmt = "{:3d} {:3d} {:16.6f} {:12.6f} {:16.6f} # {}\n"
        for i in range(0, lmpdata.atom_coeffs.shape[0]):
            for j in range(i, lmpdata.atom_coeffs.shape[0]):
                a = lmpdata.atom_coeffs[i][2]
                b = lmpdata.atom_coeffs[j][2]
                pair = "{}-{}".format(a, b)
                if pair in FF_PAIRS.keys():
                    Aij = FF_PAIRS[pair][0]
                    rhoij = FF_PAIRS[pair][1]
                    Cij = FF_PAIRS[pair][2]
                else:
                    Aij = 0.0
                    rhoij = 1.0
                    Cij = 0.0
                self.pairij += fmt.format(i+1, j+1, Aij, rhoij, Cij, pair)


def GetForceField():
    ff = ATOMS.strip().split("\n")[1:]
    ff = np.array(" ".join(ff).split(), dtype=object).reshape((-1, 4))
    ff[:, 2:4] = ff[:, 2:4].astype(float)
    ff_atoms = {}
    for f in ff:
        ff_atoms[f[0]] = [f[2], f[3]]  # charge mass
    ff = PAIR_FF.strip().split("\n")[1:]
    ff = np.array(" ".join(ff).split(), dtype=object).reshape((-1, 4))
    ff[:, 1:4] = ff[:, 1:4].astype(float)
    ev2kcalmol = 1.602176462e-19 * 6.02214129e23 * 1e-3 / 4.184
    ff[:, [1, 3]] = ff[:, [1, 3]] * ev2kcalmol  # ev -> kcal/mol
    ff_pairs = {}
    for f in ff:
        ff_pairs[f[0]] = [f[1], f[2], f[3]]  # Aij, rhoij, Cij
    return ff_atoms, ff_pairs


# Global variable: FF_ATOMS, FF_PAIRS
FF_ATOMS, FF_PAIRS = GetForceField()

lmp = LammpsData()
lmpdata = template.LammpsData()
lmpdata.target_density = args.density
lmpdata.initial_density = args.density/args.deform**3
lmp.LoadAtomsFromArg()
lmp.SetAtoms(lmpdata)
lmpdata.SetCubicLatticeFromDensity(lmpdata.initial_density)
lmp.SetXYZ(lmpdata)
lmpdata.SetLammpsData()
lmp.SetPairIJ(lmpdata)
lmpdata.molname = lmp.systemname
o = open(args.outfile + ".data", "w")
o.write(lmpdata.head)
o.write(lmpdata.lattice)
o.write(lmpdata.body)
o.write(lmp.pairij)

# debug
# lmpdata.OutputXYZ(args.outfile + ".xyz")
lmpdata.ShowMolecularInfoCell()
lmpin = template.LammpsIN(lmpdata, args.outfile)
lmpin.Glass()
o = open(args.outfile + ".in", "w")
o.write(lmpin.indata)
