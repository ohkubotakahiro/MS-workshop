#!/usr/bin/env python
# Scripts for workshop on MD simulation of molten salts ans ionic liquid.
# Author: Takahiro Ohkubo <ohkubo.takahiro@faculty.chiba-u.jp>
# Version: 2019/09
# http://amorphous.tf.chiba-u.jp/MS%E8%AC%9B%E7%BF%92%E4%BC%9A/molten.html
# Copyright (C) 2019 Takahiro Ohkubo (Chiba university)

import os
import re
import glob
import numpy as np
import argparse
import ms_template as template
MAXCHAIN = 100
MAXRING = 100
ZMATDIR = os.path.dirname(__file__) + "/ildatabase/"

ZMATFILES = sorted(glob.glob(ZMATDIR + "*.zmat"))

ZMATS = [os.path.basename(s)[:-5] for s in ZMATFILES]
zfiles = np.empty((0, 3), dtype=object)
for i, zfile in enumerate(ZMATFILES):
    text = open(zfile).readline().strip()
    zfile_ = os.path.basename(zfile)[:-5]
    zfiles = np.vstack((zfiles, [zfile_, text, text[-1]]))
anions = zfiles[zfiles[:, 2] == '-', :]
cations = zfiles[zfiles[:, 2] == '+', :]
zmatfiles = np.vstack((cations, anions))

par = argparse.ArgumentParser(description="test")
for z in zmatfiles:
    lopt = "--{}".format(z[0])
    shelp = "number of {}".format(z[1])
    par.add_argument(lopt, metavar='N', default=0, type=int, help=shelp)
par.add_argument('--density', metavar=1.6, default=1.6, type=float,
                 help="target density g/cm3")
par.add_argument('--deform', metavar=2, default=2, type=float,
                 help="expand scale for deformation to build initial cell")
par.add_argument('--rcutoff', metavar=1.5, default=1.5, type=float,
                 help="cutoff distance for random atomic position")
par.add_argument('-o', '--outfile', metavar='basename', required=True,
                 help="basename for lammps input files")

args = par.parse_args()

body = open(ZMATDIR + "/il.ff").read() + "\n"
body = re.sub("#.*\n", "", body)
FF = {}
for s in ['ATOMS', 'BONDS', 'ANGLES', 'DIHEDRALS', 'IMPROPER']:
    d = re.search("{}(.*?)\n\n".format(s), body, re.DOTALL).group(1)
    data = d.strip().split("\n")
    n, m = len(data), len(data[0].split())
    f = np.array(" ".join(data).split(), dtype=object).reshape((n, m))
    FF[s] = f
FF['ATOMS'][:, [2, 3, 5, 6]] = FF['ATOMS'][:, [2, 3, 5, 6]].astype(float)
FF['BONDS'][:, 3:5] = FF['BONDS'][:, 3:5].astype(float)
FF['ANGLES'][:, 4:6] = FF['ANGLES'][:, 4:6].astype(float)
FF['DIHEDRALS'][:, 5:9] = FF['DIHEDRALS'][:, 5:9].astype(float)
FF['IMPROPER'][:, 5:9] = FF['IMPROPER'][:, 5:9].astype(float)
# kJ/mol -> kcal/mol
FF['ATOMS'][:, 6] = FF['ATOMS'][:, 6] / 4.184
FF['BONDS'][:, 4] = FF['BONDS'][:, 4] / 2 / 4.184
FF['ANGLES'][:, 5] = FF['ANGLES'][:, 5] / 2 / 4.184
FF['DIHEDRALS'][:, 5:9] = FF['DIHEDRALS'][:, 5:9] / 4.184
FF['IMPROPER'][:, 5:9] = FF['IMPROPER'][:, 5:9] / 4.184


class LoadDataFromFile():
    def __init__(self, infile):
        self.infile = infile
        self.chains = {}
        self.rings = {}
        basename, ext = os.path.splitext(infile)
        # 0:id 1:element 2:symbol 3:x 4:y 5:z
        self.data = np.empty((0, 6), dtype=object)
        self.bonds = np.empty((0, 2), dtype=object)
        if ext == ".zmat":
            self.LoadFromZmatFile()
        else:
            self.SetBonds()
        # check single molecules from bond chains
        self.SetChains()
        self.CheckBondCutoff()
        # self.SetRings()
        # print("{}: {} molecues".format(infile, len(self.molecues)))

    def CheckBondCutoff(self):
        if len(self.molecues) != 1:
            print("{}: Isolated atoms were deteceted!".format(self.infile))
            print("Please check bond cutoff distance.")
            exit()

    def SetRings(self):
        for i in range(3, MAXRING):
            self.rings[i] = np.empty((0, i), dtype=int)
        for k, lists in self.chains.items():
            if lists.shape[0] == 0:
                continue
            for ll in lists:
                if np.unique(ll).shape[0] != k:
                    u, c = np.unique(ll, return_counts=True)
                    p = u[c == 2]
                    pp = np.where(ll == p)[0]
                    flag = np.ones(ll.shape[0], dtype=bool)
                    flag[:pp[0]] = False
                    flag[pp[1]:] = False
                    n = np.sum(flag)
                    self.rings[n] = np.vstack((self.rings[n], ll[flag]))
        for i in range(3, MAXRING):
            if self.rings[i].shape[0] != 0:
                self.rings[i] = np.unique(self.rings[i], axis=0)

    def Nchain(self, bonds, chain):
        # bonds:(False;non-check, True;check) :flag 1:id_a 1:id_b
        idx_a = np.where((bonds[:, 0] == False) &
                         (bonds[:, 1] == chain[-1]))[0]
        idx_b = np.where((bonds[:, 0] == False) &
                         (bonds[:, 2] == chain[-1]))[0]
        ids_a = bonds[idx_a, 2]
        ids_b = bonds[idx_b, 1]
        ids = np.append(ids_a, ids_b)
        idx = np.append(idx_a, idx_b)
        check_ids = np.vstack((ids, idx)).T  # ids, bondのidx
        for c in check_ids:
            bonds[c[1], 0] = 1
            chain = np.append(chain, c[0])
            n = chain.shape[0]
            self.chains[n] = np.vstack((self.chains[n], chain))
            if chain.shape[0] == MAXCHAIN:
                chain = chain[:-1]
                continue
            self.Nchain(bonds, chain)
            chain = chain[:-1]

    def SetChains(self):
        self.chains = {}
        for i in range(1, MAXCHAIN+1):
            self.chains[i] = np.empty((0, i), dtype=int)
        # flag, id1, id2
        bonds = np.empty((self.bonds.shape[0], 3), dtype=object)
        bonds[:, 1:3] = self.bonds
        bonds[:, 0] = False
        atoms = np.empty((self.data.shape[0], 2), dtype=object)
        atoms[:, 1] = self.data[:, 0]
        bonds[:, 0] = False
        mols = []
        for a in atoms:
            if a[0] == True:  # check済み
                continue
            self.Nchain(bonds, np.array([a[1]]))
            ids = np.empty(0, dtype=int)
            for chain in self.chains.values():
                ids = np.append(ids, chain.reshape(-1))
            ids_u = np.unique(ids)
            atoms[ids_u - 1, 0] = True
            mols.append(ids_u)
        self.molecues = mols

    def SetBonds(self):
        if self.data.shape[0] == 1:
            return
        bonds = np.empty((0, 2), dtype=int)
        for k, v in CUTOFF.items():
            a, b = k.strip().split("-")
            idx_a = np.where(self.data[:, 1] == a)[0]
            idx_b = np.where(self.data[:, 1] == b)[0]
            dx = np.repeat([self.data[idx_a, 3]], len(idx_b), axis=0)
            dy = np.repeat([self.data[idx_a, 4]], len(idx_b), axis=0)
            dz = np.repeat([self.data[idx_a, 5]], len(idx_b), axis=0)
            dx = dx - self.data[idx_b, 3].reshape((-1, 1))
            dy = dy - self.data[idx_b, 4].reshape((-1, 1))
            dz = dz - self.data[idx_b, 5].reshape((-1, 1))
            r = np.sqrt((dx**2 + dy**2 + dz**2).astype(float))
            idxs = np.where((r < v) & (r != 0.0))
            ids_b = self.data[idx_b[idxs[0]], 0]
            ids_a = self.data[idx_a[idxs[1]], 0]
            bb = np.array([ids_a, ids_b], dtype=int).T
            bonds = np.vstack((bonds, bb))
        if bonds.shape[0] == 0:
            print("atoms={}, but no bonds!".format(self.data.shape[0]))
            print("Please check cutoff list.")
            exit()
        bonds = np.sort(bonds, axis=1)
        bonds = np.unique(bonds, axis=0)
        nm = (bonds.shape[0], self.bonds.shape[1])
        self.bonds = np.empty(nm, dtype=object)
        self.bonds[:, 0:2] = bonds

    def zmat2xyz(self, zdata):
        data = np.empty((zdata.shape[0], self.data.shape[1]), dtype=object)
        for i, z in enumerate(zdata):
            data[i, 1] = re.search("^([A-Z][a-z]*)", z[1]).group(1)
            data[i, 2] = z[1]
        # first atom at origin
        data[0, 3:6] = 0.0
        if zdata.shape[0] == 1:
            self.data = data
            return
        # second atom at distance r from first along xx
        data[1, 3:6] = 0.0
        data[1, 3] = zdata[1, 3]
        bonds = np.empty((0, 2))
        bonds = np.vstack((bonds, zdata[1, [0, 2]]))
        if zdata.shape[0] == 2:
            self.data = data
            return
        # third atom at distance r from ir forms angle a 3-ir-ia in plane xy
        r = zdata[2, 3]
        ang = zdata[2, 5] * np.pi / 180.0
        delx = data[zdata[2, 4] - 1, 3] - data[zdata[2, 2] - 1, 3]
        dely = data[zdata[2, 4] - 1, 4] - data[zdata[2, 2] - 1, 4]
        theta = np.arccos(delx / np.sqrt(delx*delx + dely*dely))
        if dely < 0.0:
            theta = 2 * np.pi - theta
        ang = theta - ang
        data[2, 3] = data[zdata[2, 2] - 1, 3] + r * np.cos(ang)
        data[2, 4] = data[zdata[2, 2] - 1, 4] + r * np.sin(ang)
        data[2, 5] = 0.0
        bonds = np.vstack((bonds, zdata[2, [0, 2]]))
        if zdata.shape[0] == 3:
            self.data = data
            return data
        for i in range(3, zdata.shape[0]):
            vB = data[zdata[i, 2] - 1, 3:6].astype(float)
            vC = data[zdata[i, 4] - 1, 3:6].astype(float)
            vD = data[zdata[i, 6] - 1, 3:6].astype(float)
            vBC = vC - vB
            vCD = vD - vC
            BC = np.linalg.norm(vBC)
            r = zdata[i, 3]
            bB = r * np.cos(zdata[i, 5] * np.pi / 180.0)  # bond-angle
            bA = r * np.sin(zdata[i, 5] * np.pi / 180.0)  # bond-angle
            aA = bA * np.sin(zdata[i, 7] * np.pi / 180.0)  # dihedral
            ba = bA * np.cos(zdata[i, 7] * np.pi / 180.0)  # dihedral
            vb = vC - vBC * ((BC - bB) / BC)
            vn = np.cross(vCD, vBC)
            vn = vn/np.linalg.norm(vn)
            vm = np.cross(vBC, vn)
            vm = vm/np.linalg.norm(vm)
            va = vb + np.dot(vm, ba)
            vA = va + np.dot(vn, aA)
            data[i, 3:6] = vA
        self.data = data
        self.data[:, 0] = np.arange(1, data.shape[0] + 1)

    def LoadFromZmatFile(self):
        body = open(self.infile).read().strip()
        body = re.sub("#.+?\n", "", body)
        lines = body.split("\n")
        lines = [v.strip() + "\n" for v in lines]
        s = "".join(lines[2:])
        s = re.search("(.+?)\n\n", s, re.DOTALL)
        variable = re.findall("(.+) *= *(.+)", body)
        data = s.group(1)
        for v in variable:
            data = re.sub(v[0].strip(), v[1].strip(), data)
        data = data.split("\n")
        data[0] = data[0] + " 0" * 6
        if len(data) > 1:
            data[1] = data[1] + " 0" * 4
        if len(data) > 2:
            data[2] = data[2] + " 0" * 2
        n = len(data[0].split())
        data = np.array(" ".join(data).split(), dtype=object)
        data = data.reshape((-1, n))
        zdata = np.empty((data.shape[0], 8), dtype=object)
        s = np.abs(7 - n)
        zdata[:, 0] = np.arange(1, zdata.shape[0] + 1)
        zdata[:, 1:8] = data[:, s:s+7]
        zdata[:, ::2] = zdata[:, ::2].astype(int)
        zdata[:, 3::2] = zdata[:, 3::2].astype(float)
        self.zmat2xyz(zdata)
        self.data[:, 0] = np.arange(1, self.data.shape[0] + 1)
        self.bonds = zdata[1:, [0, 2]].astype(object)
        bonds = re.findall("[cC][oO][nN][nN][eE][cC][tT] +(\d+) +(\d+)", body)
        bonds = np.array(bonds, dtype=int)
        if bonds.shape[0] != 0:
            self.bonds = np.vstack((self.bonds, bonds))


molecules = []
for k, value in par.parse_args()._get_kwargs():
    if (k in ZMATS) and value != 0:
        print("Building {}{}...".format(value, k))
        zfile = ZMATFILES[ZMATS.index(k)]
        f = LoadDataFromFile(zfile)
        mol = template.MOLECULE(k)
        mol.SetAtomData(f.data, f.bonds)
        mol.SetFF(FF)
        molecules.append(mol)
        for i in range(1, value):
            mol_new = template.MOLECULE(k)
            # mol_new.CopyMolecule(mol)
            molecules.append(mol_new)
print("done")
lmpdata = template.LammpsData()
lmpdata.target_density = args.density
lmpdata.initial_density = args.density/args.deform**3
lmpdata.SetMolname(molecules)
lmpdata.AppendMolecules(molecules)
lmpdata.SetCubicLatticeFromDensity(lmpdata.initial_density)
print("Setting random position (cutoff = {})...".format(args.rcutoff))
lmpdata.SetCubicRandomPosition(rcutoff=args.rcutoff)
print("done")
lmpdata.SetLammpsData()
lmpdata.SetLammpsPairCoeffs()
lmpdata.SetLammpsBondCoeffs()
lmpdata.SetLammpsAngleCoeffs()
lmpdata.SetLammpsDihedralCoeffs()
lmpdata.SetLammpsImproperCoeffs()
# debug
# lmpdata.OutputXYZ(args.outfile + ".xyz")
lmpdata.ShowMolecularInfoCell()
print("Writing input files...".format(args.rcutoff))
lmpin = template.LammpsIN(lmpdata, args.outfile)
lmpin.IonicLiquid()

o = open(args.outfile + ".in", "w")
o.write(lmpin.indata)
o = open(args.outfile + ".data", "w")
o.write(lmpdata.head)
o.write(lmpdata.lattice)
o.write(lmpdata.body)
print("done")
