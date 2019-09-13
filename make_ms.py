#!/usr/bin/env python
# Scripts for workshop on MD simulation of molten salts ans ionic liquid.
# Author: Takahiro Ohkubo <ohkubo.takahiro@faculty.chiba-u.jp>
# Version: 2019/09
# http://amorphous.tf.chiba-u.jp/MS%E8%AC%9B%E7%BF%92%E4%BC%9A/molten.html
# Copyright (C) 2019 Takahiro Ohkubo (Chiba university)

"""
mixing rule C, D for anion-anion
C = fij/sum(fij) * Cij 
D = fij/sum(fij) * Dij
"""
import os
import re
import numpy as np
import argparse
import ms_template as template


def LoadFFdatabase():
    dbfile = os.path.dirname(__file__) + "/msdatabase/ms.ff"
    body = open(dbfile).read() + "\n"
    body = re.sub("#.*?\n", "", body)
    PAIR, VDW = {}, {}
    vdw = re.search("Atom\n(.+?)\n\n", body, re.DOTALL).group(1)
    for v in vdw.split("\n"):
        d = v.split()
        # 0:charge 1:mass 2:sigma
        VDW[d[0]] = np.array(d[1:], dtype=float)
    pairij = re.search("Pair IJ(.*\n\n)", body, re.DOTALL).group(1)
    pairij = re.findall("PAIR (.+?)\n(.+?\n.+?\n.+?)\n", pairij.strip())
    for pair in pairij:
        k = pair[0].strip()
        # 0:atom_a 1:atom_b 2:A 3:sigma 4:rho 5:C 6:D
        PAIR[k] = np.empty((3, 7), dtype=object)
        ff = np.array(pair[1].split(), dtype=object).reshape((3, 6))
        PAIR[k][:, 2:7] = ff[:, 1:6]
        for i, f in enumerate(ff[:, 0]):
            PAIR[k][i, 0], PAIR[k][i, 1] = f.split("-")
            PAIR[k][i, 2:7] = PAIR[k][i, 2:7].astype(float)
        PAIR[k][:, 2] = PAIR[k][:, 2]/4.184  # A; kJ/->kcal/mol
        PAIR[k][:, 5] = PAIR[k][:, 5]/4.184  # A; kJ/->kcal/mol
        PAIR[k][:, 6] = PAIR[k][:, 6]/4.184  # A; kJ/->kcal/mol
    return PAIR, VDW


PAIR, VDW = LoadFFdatabase()
par = argparse.ArgumentParser(description="test")
for k in PAIR.keys():
    lopt = "--{}".format(k)
    shelp = "number of {}".format(k)
    par.add_argument(lopt, metavar='N', default=0, type=int, help=shelp)
par.add_argument('--density', metavar=2.0, default=2.0, type=float,
                 help="target density g/cm3")
par.add_argument('--deform', metavar=1, default=1, type=float,
                 help="expand scale for deformation of initial cell length")
par.add_argument('-o', '--outfile', required=True,
                 help="basename for output files")
args = par.parse_args()


def LoadAtomsFromArg():
    # 0:elem_a 1:elem_b 2:molname 3:num
    atoms = np.empty((0, 4), dtype=object)
    for k, value in par.parse_args()._get_kwargs():
        if (k in PAIR.keys()) and (value != 0):
            obj = re.search("([A-Z][a-z]*)([A-Z][a-z]*)", k)
            a, b = obj.group(1), obj.group(2)
            atoms = np.vstack((atoms, [a, b, k, value]))
    atoms[:, 3] = atoms[:, 3].astype(int)
    return atoms


def SetAtom(atoms, lmpdata):
    molname = ["{}{}".format(a[3], a[2]) for a in atoms]
    lmpdata.molname = "-".join(molname)
    elems, mols = [], []
    for a in atoms:
        elems += [a[0], a[1]] * a[3]
        mols += [a[2]] * 2 * a[3]
    nm = (len(elems), lmpdata.atoms.shape[1])
    # 0:id 1:molid 2:type 3:charge 4:x 5:y 6:z 7:element 8:symbol 9:ff
    lmpdata.atoms = np.empty(nm, dtype=object)
    lmpdata.atoms[:, 0] = np.arange(1, lmpdata.atoms.shape[0] + 1)
    lmpdata.atoms[:, 1] = 1
    lmpdata.atoms[:, 7] = elems
    lmpdata.atoms[:, 8] = elems
    lmpdata.atoms[:, 9] = elems
    elems_o = ["Li", "Na", "K", "Rb", "Cs", "F", "Cl", "Br", "I"]
    elems_u = np.unique(elems)
    elems = np.empty(0)
    for e in elems_o:
        if e in elems_u:
            elems = np.append(elems, e)
    # 0:type 1:symbol 2:ff 3:mass 4:charge 5:func 6:eps 7:sigma 8:symbol
    nm = (elems.shape[0], lmpdata.atom_coeffs.shape[1])
    lmpdata.atom_coeffs = np.empty(nm, dtype=object)
    for i, e in enumerate(elems):
        flag = lmpdata.atoms[:, 7] == e
        lmpdata.atoms[flag, 2] = i + 1  # type
        lmpdata.atoms[flag, 3] = VDW[e][0]  # charge
        lmpdata.atom_coeffs[i, 0] = i + 1  # type
        lmpdata.atom_coeffs[i, 1] = e  # type
        lmpdata.atom_coeffs[i, 2] = e  # symbol
        lmpdata.atom_coeffs[i, 3] = VDW[e][1]  # mass
        lmpdata.atom_coeffs[i, 4] = VDW[e][0]  # charge
        lmpdata.atom_coeffs[i, 7] = VDW[e][2]  # sigma
        lmpdata.atom_coeffs[i, 8] = e   # sigma
        lmpdata.atom_coeffs[i, 9] = ""  # molname


def SetXYZGrid(lmpdata):
    mols = int((0.5 * lmpdata.atoms.shape[0]))
    n = np.ceil(mols**(1.0/3.0)).astype(int)
    x = np.linspace(0, lmpdata.a, n, endpoint=False)
    y = np.linspace(0, lmpdata.b, n, endpoint=False)
    z = np.linspace(0, lmpdata.c, n, endpoint=False)
    dx = (x[1] - x[0])*0.5
    dy = (y[1] - y[0])*0.5
    dz = (z[1] - z[0])*0.5
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    grid = x.reshape(-1)
    grid = np.vstack((grid, y.reshape(-1)))
    grid = np.vstack((grid, z.reshape(-1)))
    grid = grid.T
    idx = np.arange(grid.shape[0])
    np.random.shuffle(idx)
    idx = idx[:mols]
    lmpdata.atoms[::2, 4:7] = grid[idx, 0:3]
    lmpdata.atoms[1::2, 4:7] = grid[idx, 0:3] + [dx, dy, dz]


def MakePairIJ(atoms, lmpdata):
    elems = lmpdata.atom_coeffs[:, 1]
    charges = lmpdata.atom_coeffs[:, 4]
    flags = {}
    flags[-2] = "--"
    flags[0] = "+-"
    flags[+2] = "++"
    pairs = np.empty((0, 5))
    for i in range(len(elems)):
        for j in range(i, len(elems)):
            s = charges[i] + charges[j]
            pairs = np.vstack(
                (pairs, [i+1, j+1, elems[i], elems[j], flags[s]]))
    # 0:type_a 1:type_b 2:elem_a 3:elem_b 4:+/- 5:Aij 6:rhoij 7:Cij 8:Dij
    coeffs = np.empty((pairs.shape[0], 9), dtype=object)
    coeffs[:, 0:5] = pairs
    pair_coeffs = {}
    rho = 0
    for a in atoms:
        pair = a[2]
        rho += a[3]/np.sum(atoms[:, 3]) / (PAIR[pair][1][4])
    rho_const = 1.0/rho
    for c in coeffs:
        idxs = np.empty(0, dtype=int)
        idxs = np.append(idxs, np.where(atoms[:, 0] == c[2])[0])
        idxs = np.append(idxs, np.where(atoms[:, 0] == c[3])[0])
        idxs = np.append(idxs, np.where(atoms[:, 1] == c[2])[0])
        idxs = np.append(idxs, np.where(atoms[:, 1] == c[3])[0])
        idxs = np.unique(idxs)
        atoms_ = atoms[idxs]
        frac_sum = np.sum(atoms_[:, 3])
        A, rho, sigma, C, D = 0, 0, 0, 0, 0
        sigma = VDW[c[2]][2] + VDW[c[3]][2]
        pair = c[2] + c[3]
        if c[4] == "++":
            for a in atoms_:
                A += a[3]/frac_sum * PAIR[a[2]][0][2]
                rho += a[3]/frac_sum * PAIR[a[2]][0][4]
                C += a[3]/frac_sum * PAIR[a[2]][0][5]
                D += a[3]/frac_sum * PAIR[a[2]][0][6]
        elif c[4] == "--":
            for a in atoms_:
                A += a[3]/frac_sum * PAIR[a[2]][2][2]
                rho += a[3]/frac_sum * PAIR[a[2]][2][4]
                C += a[3]/frac_sum * PAIR[a[2]][2][5]
                D += a[3]/frac_sum * PAIR[a[2]][2][6]
        elif c[4] == "+-":
            A = PAIR[pair][1][2]
            rho = PAIR[pair][1][4]
            C = PAIR[pair][1][5]
            D = PAIR[pair][1][6]
        else:
            print("Atomic +/- type is missing.")
            exit()
        # if you use constat rho
        # rho = rho_const
        pair_coeffs[pair] = [c[0], c[1], c[2], c[3], A, sigma, rho, C, D]
    return pair_coeffs


atoms = LoadAtomsFromArg()
lmpdata = template.LammpsData()
lmpdata.target_density = args.density
lmpdata.initial_density = args.density/args.deform**3

SetAtom(atoms, lmpdata)
lmpdata.SetCubicLatticeFromDensity(lmpdata.initial_density)
SetXYZGrid(lmpdata)
lmpdata.SetLammpsData()
pairij_coeff = MakePairIJ(atoms, lmpdata)

o = open(args.outfile + ".data", "w")
o.write(lmpdata.head)
o.write(lmpdata.lattice)
o.write(lmpdata.body)
o.write("\nPairIJ Coeffs\n\n")
fmt = "{:>3s} {:>3s} {:12.5f} {:10.5f} {:10.5f} {:12.5f} {:12.5f} # {}\n"
for k, v in pairij_coeff.items():
    s = v[2] + "-" + v[3]
    o.write(fmt.format(v[0], v[1], v[4], v[6], v[5], v[7], v[8], s))

# debug
# lmpdata.OutputXYZ(args.outfile + ".xyz")
lmpdata.ShowMolecularInfoCell()

lmpin = template.LammpsIN(lmpdata, args.outfile)
lmpin.MoltenSalts()
o = open(args.outfile + ".in", "w")
o.write(lmpin.indata)
