#!/usr/bin/env python
# Scripts for workshop on MD simulation of molten salts ans ionic liquid.
# Author: Takahiro Ohkubo <ohkubo.takahiro@faculty.chiba-u.jp>
# Version: 2019/09
# http://amorphous.tf.chiba-u.jp/MS%E8%AC%9B%E7%BF%92%E4%BC%9A/molten.html
# Copyright (C) 2019 Takahiro Ohkubo (Chiba university)
import numpy as np

CUTOFF = {}
CUTOFF['B-F'] = 1.5
CUTOFF['C-C'] = 1.6
CUTOFF['C-N'] = 1.6
CUTOFF['C-H'] = 1.1
CUTOFF['C-F'] = 1.5
CUTOFF['C-O'] = 1.7
CUTOFF['O-H'] = 1.2
CUTOFF['N-H'] = 1.2

CUTOFF['P-F'] = 1.7
CUTOFF['P-C'] = 1.9

CUTOFF['S-C'] = 1.9
CUTOFF['S-O'] = 1.9
CUTOFF['S-N'] = 1.8
CUTOFF['S-F'] = 1.7


class MOLECULE():
    def __init__(self, molname):
        self.molname = molname
        self.chains = {}
        self.rings = {}
        # 0:id 1:molid 2:type 3:charge 4:x 5:y 6:z 7:element 8:symbol 9:ff
        self.atoms = np.empty((0, 10), dtype=object)
        # 0:id 1:type 2:id_a 3:id_b 4:symbol
        self.bonds = np.empty((0, 5), dtype=object)
        # 0:id 1:type 2:id_a 3:id_b 4:id_c 5:symbol
        self.angles = np.empty((0, 6), dtype=object)
        # 0:id 1:type 2:id_a 3:id_b 4:id_c 5:id_d 6:symbol
        self.dihedrals = np.empty((0, 7), dtype=object)
        # 0:id 1:type 2:id_a 3:id_b 4:id_c 5:id_d 6:symbol
        self.impropers = np.empty((0, 7), dtype=object)
        
        # 0:type 1:symbol 2:ff 3:mass 4:charge
        # 5:func 6:eps 7:sigma 8:elem 9:molname
        self.atom_coeffs = np.empty((0, 10), dtype=object)
        # 0:type 1:func 2:r0  3:K 4:symbol
        self.bond_coeffs = np.empty((0, 5), dtype=object)
        # 0:type 1:func 2:theta0 3:K 4:symbol
        self.angle_coeffs = np.empty((0, 5), dtype=object)
        # 0:type 1:func 2:V1 3:V2 4:V3 5:V4 6:symbol
        self.dihedral_coeffs = np.empty((0, 7), dtype=object)
        # 0:type 1:func 2:V1 3:V2 4:V3 5:V4 6:symbol
        self.improper_coeffs = np.empty((0, 7), dtype=object)

    def SetAtomData(self, data, bonds):
        nm = (data.shape[0], self.atoms.shape[1])
        self.atoms = np.empty(nm, dtype=object)
        self.atoms[:, 0] = data[:, 0]
        self.atoms[:, 4:7] = data[:, 3:6]
        self.atoms[:, 7] = data[:, 1]
        self.atoms[:, 8] = data[:, 2]
        if bonds.shape[0] != 0:
            nm = (bonds.shape[0], self.bonds.shape[1])
            self.bonds = np.empty(nm, dtype=object)
            self.bonds[:, 2:4] = bonds
        self.SetAngles()
        self.SetDihedrals()
        self.SetImpropers()

    def GetFFdata(self, blists, ff):
        n = blists.shape[1]
        coeffs = np.empty((0, ff.shape[1]), dtype=object)
        blists = blists.astype(int)
        symbols = self.atoms[blists - 1, 9]
        n = blists.shape[1]
        dummy = np.copy(ff[0, :])
        dummy[n+1:] = 0.0
        for i, s in enumerate(symbols):
            ff_a = ff[np.sum(ff[:, 0:n] == s, axis=1) == n]
            ff_b = ff[np.sum(ff[:, 0:n] == s[::-1], axis=1) == n]
            if ff_a.shape[0] == 1:
                coeffs = np.vstack((coeffs, ff_a))
            elif ff_b.shape[0] == 1:
                blists[i, :] = blists[i, ::-1]
                coeffs = np.vstack((coeffs, ff_b))
            elif ff_a.shape[0] == 0 and ff_b.shape[0] == 0:
                print("missing FF!!!; {} ({})".format(s, ff), end="")
                print("ids:", blists[i])
                dummy[:n] = s
                coeffs = np.vstack((coeffs, dummy))
            else:
                print("duplicated FF!!!; {} ()".format(s, ff), end="")
                print("ids:", blists[i])
                exit()
        symbols_ = coeffs[:, 0]
        for i in range(1, n):
            symbols_ = symbols_ + "-" + coeffs[:, i]
        return coeffs, blists, symbols_

    def SetFFBond(self, FF):
        c, b, s = self.GetFFdata(self.bonds[:, 2:4], FF['BONDS'])
        self.bonds[:, 2:4] = b
        self.bonds[:, 4] = s
        symbol, idx = np.unique(s, return_index=True)
        nm = (symbol.shape[0], self.bond_coeffs.shape[1])
        self.bond_coeffs = np.empty(nm, dtype=object)
        self.bond_coeffs[:, 1:4] = c[idx, 2:5]
        self.bond_coeffs[:, 4] = symbol
        self.bond_coeffs[:, 0] = np.arange(1, self.bond_coeffs.shape[0] + 1)
        self.bond_coeffs[:, 3] = self.bond_coeffs[:, 3]
        for s in self.bond_coeffs:
            self.bonds[self.bonds[:, 4] == s[4], 1] = s[0]
        self.bonds[:, 0] = np.arange(1, self.bonds.shape[0] + 1)

    def SetFFAngle(self, FF):
        c, b, s = self.GetFFdata(self.angles[:, 2:5], FF['ANGLES'])
        self.angles[:, 2:5] = b
        self.angles[:, 5] = s
        symbol, idx = np.unique(s, return_index=True)
        nm = (symbol.shape[0], self.angle_coeffs.shape[1])
        self.angle_coeffs = np.empty(nm, dtype=object)
        self.angle_coeffs[:, 1:4] = c[idx, 3:6]
        self.angle_coeffs[:, 4] = symbol
        self.angle_coeffs[:, 0] = np.arange(1, self.angle_coeffs.shape[0] + 1)
        for s in self.angle_coeffs:
            self.angles[self.angles[:, 5] == s[4], 1] = s[0]
        self.angles[:, 0] = np.arange(1, self.angles.shape[0] + 1)

    def SetFFDihedral(self, FF):
        c, b, s = self.GetFFdata(self.dihedrals[:, 2:6], FF['DIHEDRALS'])
        self.dihedrals[:, 2:6] = b
        self.dihedrals[:, 6] = s
        symbol, idx = np.unique(s, return_index=True)
        nm = (symbol.shape[0], self.dihedral_coeffs.shape[1])
        self.dihedral_coeffs = np.empty(nm, dtype=object)
        self.dihedral_coeffs[:, 1:6] = c[idx, 4:9]
        self.dihedral_coeffs[:, 6] = symbol
        self.dihedral_coeffs[:, 0] = np.arange(1, symbol.shape[0] + 1)
        for s in self.dihedral_coeffs:
            self.dihedrals[self.dihedrals[:, 6] == s[6], 1] = s[0]
        self.dihedral_coeffs[:, 2:6] = self.dihedral_coeffs[:, 2:6]
        self.dihedrals[:, 0] = np.arange(1, self.dihedrals.shape[0] + 1)

    def SetFFImproper(self, FF):
        c, b, s = self.GetFFdata(self.impropers[:, 2:6], FF['IMPROPER'])
        self.impropers[:, 2:6] = b
        self.impropers[:, 6] = s
        symbol, idx = np.unique(s, return_index=True)
        nm = (symbol.shape[0], self.improper_coeffs.shape[1])
        self.improper_coeffs = np.empty(nm, dtype=object)
        self.improper_coeffs[:, 1:6] = c[idx, 4:9]
        self.improper_coeffs[:, 6] = symbol
        self.improper_coeffs[:, 0] = np.arange(1, symbol.shape[0] + 1)
        for s in self.improper_coeffs:
            self.impropers[self.impropers[:, 6] == s[6], 1] = s[0]
        self.impropers[:, 0] = np.arange(1, self.impropers.shape[0] + 1)

    def SetFF(self, FF):
        ff = FF['ATOMS']
        symbols = np.unique(self.atoms[:, 8])
        coeffs = np.empty(self.atom_coeffs.shape[1], dtype=object)
        for i, a in enumerate(symbols):
            ffi = ff[ff[:, 0] == a]
            if ffi.shape[0] != 1:
                print("WARINIG!!! duplicated or no FF; {}".format(a))
                exit()
            flags = self.atoms[:, 8] == a
            charge = float(ffi[0, 3])
            type_num = i + 1
            coeffs[0] = type_num
            coeffs[1:8] = ffi[0, :]
            coeffs[8] = self.atoms[flags, 7][0]  # element
            self.atoms[flags, 2] = type_num
            self.atoms[flags, 3] = charge
            self.atoms[flags, 9] = coeffs[2]
            self.atom_coeffs = np.vstack((self.atom_coeffs, coeffs))
        self.atom_coeffs[:, 7] = self.atom_coeffs[:, 7]
        self.atom_coeffs[:, 9] = self.molname
        self.SetFFBond(FF)
        self.SetFFAngle(FF)
        self.SetFFDihedral(FF)
        self.SetFFImproper(FF)

    def SetImpropers(self):
        angles = self.angles[:, 2:5].astype(int)
        ang, ar, c = np.unique(
            angles[:, 1], return_counts=True, return_index=True)
        ang = angles[ar[c == 3]]
        impropers = self.AddBondsLists(ang, improper=True)
        nm = (impropers.shape[0], self.impropers.shape[1])
        self.impropers = np.empty(nm, dtype=object)
        self.impropers[:, 2:6] = impropers[:, ::-1]

    def SetDihedrals(self):
        dihedrals = self.AddBondsLists(self.angles[:, 2:5])
        nm = (dihedrals.shape[0], self.dihedrals.shape[1])
        self.dihedrals = np.empty(nm, dtype=object)
        self.dihedrals[:, 2:6] = dihedrals

    def SetAngles(self):
        angles = self.AddBondsLists(self.bonds[:, 2:4])
        nm = angles.shape[0], self.angles.shape[1]
        self.angles = np.empty(nm, dtype=object)
        self.angles[:, 2:5] = angles

    def CheckBonds(self, chain, b, tail=False):
        add_blist = np.empty(0, dtype=int)
        id_a = self.bonds[self.bonds[:, 2] == b][:, 3]
        id_b = self.bonds[self.bonds[:, 3] == b][:, 2]
        ids = np.append(id_a, id_b)
        ids = np.unique(ids)
        for i in ids:
            if np.sum(chain == i) == 0:
                add_blist = np.append(add_blist, i)
        add_blist = add_blist.reshape((-1, 1))
        blist = np.repeat([chain], add_blist.shape[0], axis=0)
        if tail is False:
            blist = np.hstack((add_blist, blist))
        else:
            blist = np.hstack((blist, add_blist))
        return blist

    def AddBondsLists(self, chains, improper=False):
        blists = np.empty((0, chains.shape[1] + 1))
        for chain in chains:
            if improper is False:
                blist = self.CheckBonds(chain, chain[0], tail=False)
                blists = np.vstack((blists, blist))
                blist = self.CheckBonds(chain, chain[-1], tail=True)
                blists = np.vstack((blists, blist))
            else:
                blist = self.CheckBonds(chain, chain[1], tail=True)
                blists = np.vstack((blists, blist))
        blists = blists.astype(int)
        if blists.shape[0] == 0:
            return blists
        if improper is True:
            blists = blists[:, [0, 2, 1, 3]]
        else:
            idx = blists[:, 0] > blists[:, -1]
            blists[idx, :] = blists[idx, ::-1]
            blists = np.unique(blists, axis=0)
        return blists

    def ShowMolecularInfo(self):
        molname = self.molname
        atoms = self.atoms.shape[0]
        bonds = self.bonds.shape[0]
        angles = self.angles.shape[0]
        dihedrals = self.dihedrals.shape[0]
        impropers = self.impropers.shape[0]
        print("{}: {} atoms".format(molname, atoms))
        print("{}: {} bonds".format(molname, bonds))
        print("{}: {} angles".format(molname, angles))
        print("{}: {} dihedrals".format(molname, dihedrals))
        print("{}: {} impropers".format(molname, impropers))

    def OutputXYZ(self, outfile):
        o = open(outfile, "w")
        o.write("{}\n\n".format(self.atoms.shape[0]))
        fmt = "{:2s} {:12.6f} {:12.6f} {:12.6f} # {:4d} {}\n"
        for a in self.atoms:
            o.write(fmt.format(a[7], a[4], a[5], a[6], a[0], a[8]))
        print(outfile, "was created.")


class LammpsData(MOLECULE):
    def __init__(self):
        super().__init__("")
        self.a = None
        self.b = None
        self.c = None
        self.alpha = 90
        self.beta = 90
        self.gamma = 90

    def SetMolname(self, molecules):
        self.molnames = np.array([m.molname for m in molecules], dtype=object)
        molnames_u, cnt = np.unique(self.molnames, return_counts=True)
        molname = ["{}{}".format(cnt[i], m) for i, m in enumerate(molnames_u)]
        self.molname = "-".join(molname)
        self.molname_u = molnames_u

    def AddMolType(self, m):
        self.UpdateMolType(m)
        self.atom_coeffs = np.vstack((self.atom_coeffs, m.atom_coeffs))
        self.bond_coeffs = np.vstack((self.bond_coeffs, m.bond_coeffs))
        self.angle_coeffs = np.vstack((self.angle_coeffs, m.angle_coeffs))
        self.dihedral_coeffs = np.vstack(
            (self.dihedral_coeffs, m.dihedral_coeffs))
        self.improper_coeffs = np.vstack(
            (self.improper_coeffs, m.improper_coeffs))

    def UpdateType(self, mol):
        if self.atoms.shape[0] != 0 and self.atom_coeffs.shape[0] != 0:
            mol.atoms[:, 2] += self.atom_coeffs[-1, 0]
        if self.bonds.shape[0] != 0:
            mol.bonds[:, 1] += self.bond_coeffs[-1:, 0]
        if self.angles.shape[0] != 0:
            mol.angles[:, 1] += self.angle_coeffs[-1, 0]
        if self.dihedrals.shape[0] != 0:
            mol.dihedrals[:, 1] += self.dihedral_coeffs[-1, 0]
        if self.impropers.shape[0] != 0:
            mol.impropers[:, 1] += self.improper_coeffs[-1, 0]

    def AppendMolecules(self, molecules):
        for molname in self.molname_u:
            if self.atoms.shape[0] == 0:
                idmax, molidmax = 0, 0
            else:
                idmax, molidmax = self.atoms[-1, 0], self.atoms[-1, 1]
            idxs = np.where(self.molnames == molname)[0]
            mol = molecules[idxs[0]]
            self.UpdateType(mol)
            self.AddMolType(mol)
            m = len(idxs)
            ids = np.arange(0, m * mol.atoms.shape[0], mol.atoms.shape[0])
            ids = ids + idmax
            # atom
            ids_ = np.repeat(ids, mol.atoms.shape[0])
            atoms = np.tile(mol.atoms, (m, 1))
            atoms[:, 0] = atoms[:, 0] + ids_
            molids = np.arange(1, m + 1) + molidmax
            molids = np.repeat(molids, mol.atoms.shape[0])
            atoms[:, 1] = molids
            self.atoms = np.vstack((self.atoms, atoms))
            # bonds
            bonds = np.tile(mol.bonds, (m, 1))
            ids_ = np.repeat(ids, mol.bonds.shape[0])
            ids__ = np.repeat(ids_.reshape(-1, 1), 2, axis=1)
            bonds[:, 2:4] = bonds[:, 2:4] + ids__
            self.bonds = np.vstack((self.bonds, bonds))
            # angles
            angles = np.tile(mol.angles, (m, 1))
            ids_ = np.repeat(ids, mol.angles.shape[0])
            ids__ = np.repeat(ids_.reshape(-1, 1), 3, axis=1)
            angles[:, 2:5] = angles[:, 2:5] + ids__
            self.angles = np.vstack((self.angles, angles))
            # dihedrals
            dihedrals = np.tile(mol.dihedrals, (m, 1))
            ids_ = np.repeat(ids, mol.dihedrals.shape[0])
            ids__ = np.repeat(ids_.reshape(-1, 1), 4, axis=1)
            dihedrals[:, 2:6] = dihedrals[:, 2:6] + ids__
            self.dihedrals = np.vstack((self.dihedrals, dihedrals))
            # impropers
            impropers = np.tile(mol.impropers, (m, 1))
            ids_ = np.repeat(ids, mol.impropers.shape[0])
            ids__ = np.repeat(ids_.reshape(-1, 1), 4, axis=1)
            impropers[:, 2:6] = impropers[:, 2:6] + ids__
            self.impropers = np.vstack((self.impropers, impropers))
        # self.bonds[:, 0] = np.arange(1, self.bonds.shape[0] + 1)
        # self.angles[:, 0] = np.arange(1, self.angles.shape[0] + 1)
        # self.dihedrals[:, 0] = np.arange(1, self.dihedrals.shape[0] + 1)
        # self.impropers[:, 0] = np.arange(1, self.impropers.shape[0] + 1)

    def CalcDensity(self):
        W = 0.0
        symbol, idx = np.unique(self.atom_coeffs[:, 2], return_index=True)
        mass = self.atom_coeffs[idx, 3]
        for a, m in zip(symbol, mass):
            W = W + np.sum(self.atoms[:, 9] == a) * m
        W = W / 6.02214129e23
        V = self.M[0, 0] * self.M[1, 1] * self.M[2, 2] * 1e-24
        rho = W/V  # g/cm3
        return rho

    def SetCubicRandomPosition(self, rcutoff=1.5):
        L = np.array([self.a, self.b, self.c])
        molids, idxs = np.unique(self.atoms[:, 1], return_index=True)
        xyz = self.atoms[self.atoms[:, 1] == molids[0], 4:7]
        for i, m in enumerate(molids[1:]):
            xyz_origin = self.atoms[self.atoms[:, 1] == m, 4:7]
            while True:
                xyz_i = xyz_origin + L * np.random.random(3)
                xi = xyz_i[:, 0]
                yi = xyz_i[:, 1]
                zi = xyz_i[:, 2]
                x = np.repeat(xi.reshape((-1, 1)), xyz.shape[0], axis=1)
                y = np.repeat(yi.reshape((-1, 1)), xyz.shape[0], axis=1)
                z = np.repeat(zi.reshape((-1, 1)), xyz.shape[0], axis=1)
                dx = (x - xyz[:, 0]).astype(float)
                dy = (y - xyz[:, 1]).astype(float)
                dz = (z - xyz[:, 2]).astype(float)
                dx = dx - np.rint(dx/L[0]) * L[0]
                dy = dy - np.rint(dy/L[1]) * L[1]
                dz = dz - np.rint(dz/L[2]) * L[2]
                r = np.sqrt(dx**2 + dy**2 + dz**2)
                if np.all(r > rcutoff):
                    print("{}/{} {}".format(i+2,
                                            molids.shape[0], self.molnames[i]))
                    xyz = np.vstack((xyz, xyz_i))
                    break
        self.atoms[:, 4:7] = xyz

    def SetCubicLatticeFromDensity(self, density):
        self.alpha = 90
        self.beta = 90
        self.gamma = 90
        rho = density
        W = 0.0
        n = 0
        symbol, idx = np.unique(self.atom_coeffs[:, 2], return_index=True)
        mass = self.atom_coeffs[idx, 3]
        for a, m in zip(symbol, mass):
            n = n + np.sum(self.atoms[:, 9] == a)
            W = W + np.sum(self.atoms[:, 9] == a) * m
        L = (W / 6.02214129e23 / rho * 1e+24)**(1.0/3.0)
        self.a = L
        self.b = L
        self.c = L
        self.SetCellVector()
        lattice = ""
        lattice += "0.0 {} xlo xhi\n".format(L)
        lattice += "0.0 {} ylo yhi\n".format(L)
        lattice += "0.0 {} zlo zhi\n".format(L)
        self.lattice = lattice

    def UpdateMolType(self, m):
        # coeffsを持っているときだけupdateする
        # copyしたmolはcoeffsを持っていない。
        if m.atoms.shape[0] != 0 and self.atom_coeffs.shape[0] != 0:
            m.atom_coeffs[:, 0] += self.atom_coeffs[-1, 0]
        if m.bond_coeffs.shape[0] != 0 and self.bond_coeffs.shape[0] != 0:
            m.bond_coeffs[:, 0] += self.bond_coeffs[-1, 0]
        if m.angle_coeffs.shape[0] != 0 and self.angle_coeffs.shape[0] != 0:
            m.angle_coeffs[:, 0] += self.angle_coeffs[-1, 0]
        if m.dihedral_coeffs.shape[0] != 0 and \
           self.dihedral_coeffs.shape[0] != 0:
            m.dihedral_coeffs[:, 0] += self.dihedral_coeffs[-1, 0]
        if m.improper_coeffs.shape[0] != 0 and \
           self.improper_coeffs.shape[0] != 0:
            m.improper_coeffs[:, 0] += self.improper_coeffs[-1, 0]

    def SetLammpsData(self):
        head = "molecular system: {}\n".format(self.molname)
        body = ""
        head += "{} atoms\n".format(self.atoms.shape[0])
        head += "{} bonds\n".format(self.bonds.shape[0])
        head += "{} angles\n".format(self.angles.shape[0])
        head += "{} dihedrals\n".format(self.dihedrals.shape[0])
        head += "{} impropers\n".format(self.impropers.shape[0])
        head += "{} atom types\n".format(self.atom_coeffs.shape[0])
        head += "{} bond types\n".format(self.bond_coeffs.shape[0])
        head += "{} angle types\n".format(self.angle_coeffs.shape[0])
        head += "{} dihedral types\n".format(self.dihedral_coeffs.shape[0])
        head += "{} improper types\n".format(self.improper_coeffs.shape[0])
        head += "\n"
        body += "\nMasses\n\n"
        fmt = "{:6d} {:10.6f} # {:4s} {:4s} {:s}\n"
        for m in self.atom_coeffs:
            body += fmt.format(m[0], m[3], m[1], m[2], m[9])
        body += "\nAtoms\n\n"
        fmt1 = "{:6d} {:6d} {:6d} {:10.6f} "
        fmt2 = "{:10.6f} {:10.6f} {:10.6f} # {:s}\n"
        for a in self.atoms:
            body += fmt1.format(a[0], a[1], a[2], a[3])
            body += fmt2.format(a[4], a[5], a[6], a[9])
        self.head = head
        self.body = body

    def SetLammpsPairCoeffs(self):
        self.body += "\nPair Coeffs\n\n"
        fmt = "{:6d} {:10.6f} {:10.6f} # {:s}\n"
        for m in self.atom_coeffs:
            self.body += fmt.format(m[0], m[7], m[6], m[2])

    def SetLammpsPairIJCoeffs(self, coeffs):
        self.body += "\nPair Coeffs\n\n"
        fmt = "{:6d} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f} # {:s}\n"
        for m in self.coeffs:
            self.body += fmt.format(m[0], m[1], m[2], m[2])

    def SetLammpsBondCoeffs(self):
        # Bonds
        if self.bonds.shape[0] != 0:
            self.body += "\nBonds\n\n"
            fmt = "{:6d} {:6d} {:6d} {:6d} # {:s}\n"
            for b in self.bonds:
                self.body += fmt.format(b[0], b[1], b[2], b[3], b[4])
            self.body += "\nBond Coeffs\n\n"
            fmt = "{:6d} {:10.6f} {:10.6f} # {:s}\n"
            for m in self.bond_coeffs:
                self.body += fmt.format(m[0], m[3], m[2], m[4])

    def SetLammpsAngleCoeffs(self):
        # Angles
        if self.angles.shape[0] != 0:
            self.body += "\nAngles\n\n"
            fmt = "{:6d} {:6d} {:6d} {:6d} {:6d} # {:s}\n"
            for b in self.angles:
                self.body += fmt.format(b[0], b[1], b[2], b[3], b[4], b[5])
            self.body += "\nAngle Coeffs\n\n"
            fmt = "{:6d} {:10.6f} {:10.6f} # {:s}\n"
            for m in self.angle_coeffs:
                self.body += fmt.format(m[0], m[3], m[2], m[4])

    def SetLammpsDihedralCoeffs(self):
        # Dihedrals
        if self.dihedrals.shape[0] != 0:
            self.body += "\nDihedrals\n\n"
            fmt = "{:6d} {:6d} {:6d} {:6d} {:6d} {:6d} # {:s}\n"
            for b in self.dihedrals:
                self.body += fmt.format(b[0], b[1],
                                        b[2], b[3], b[4], b[5], b[6])
            self.body += "\nDihedral Coeffs\n\n"
            fmt = "{:6d} {:10.6f} {:10.6f} {:10.6f} {:10.6f} # {:s}\n"
            for m in self.dihedral_coeffs:
                self.body += fmt.format(m[0], m[2], m[3], m[4], m[5], m[6])

    def SetLammpsImproperCoeffs(self):
        # Impropers
        if self.impropers.shape[0] != 0:
            self.body += "\nImpropers\n\n"
            fmt = "{:6d} {:6d} {:6d} {:6d} {:6d} {:6d} # {:s}\n"
            # I,J,K,L
            for b in self.impropers:
                self.body += fmt.format(b[0], b[1],
                                        b[4], b[3], b[2], b[5], b[6])
            self.body += "\nImproper Coeffs\n\n"
            fmt = "{:6d} {:10.6f} {:d} {:d} # {:s}\n"
            for m in self.improper_coeffs:
                self.body += fmt.format(m[0], m[3], -1, 2, m[6])

    def SetCellVector(self):
        a, b, c = self.a, self.b, self.c
        alpha = self.alpha * np.pi/180
        beta = self.beta * np.pi/180
        gamma = self.gamma * np.pi/180
        v1 = [a, 0, 0]
        v2 = [b*np.cos(gamma), b*np.sin(gamma), 0]
        v3 = [c*np.cos(beta),
              c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma),
              c*np.sqrt(1 + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma)
                        - np.cos(alpha)**2 -
                        np.cos(beta)**2-np.cos(gamma)**2)/np.sin(gamma)]
        self.M = np.array([v1, v2, v3])

    def ShowMolecularInfoCell(self):
        self.ShowMolecularInfo()
        rho = self.CalcDensity()
        print("initial density: {:.5f} g/cm3".format(rho))
        print("target density: {:.5f} g/cm3".format(self.target_density))
        total_charge = np.round(np.sum(self.atoms[:, 3]), 5)
        print("total charge: {:.5f}".format(total_charge))
        for a in self.atom_coeffs:
            n = np.sum(self.atoms[:, 8] == a[1])
            if a[1] != a[2]:
                print("{:>6d}: {}({}) {}".format(n, a[1], a[2], a[9]))
            else:
                print("{:>6d}: {} {}".format(n, a[1], a[9]))


class LammpsIN():
    def __init__(self, mol, outfile):
        self.mol = mol
        self.basename = outfile
        self.molname = mol.molname
        self.elems = " ".join(mol.atom_coeffs[:, 8])
        self.fix_shake = ""
        self.fix_deform = ""
        self.unfix_deform = ""
        self.indata = ""
        self.SetFixShake(self.mol)
        self.SetFixDeform(self.mol)

    def MoltenSalts(self):
        s = ms_bmh.format(molname=self.molname,
                          basename=self.basename,
                          fix_shake=self.fix_shake,
                          fix_deform=self.fix_deform,
                          elems=self.elems,
                          unfix_deform=self.unfix_deform)
        self.indata = s

    def Glass(self):
        s = glass_buck.format(molname=self.molname,
                              basename=self.basename,
                              fix_deform=self.fix_deform,
                              elems=self.elems,
                              unfix_deform=self.unfix_deform)
        self.indata = s
        

    def IonicLiquid(self):
        s = il_ljcut.format(molname=self.molname,
                            basename=self.basename,
                            fix_shake=self.fix_shake,
                            fix_deform=self.fix_deform,
                            elems=self.elems,
                            unfix_deform=self.unfix_deform)
        self.indata = s

    def SetFixDeform(self, mol):
        if mol.target_density == mol.initial_density:
            return
        scale = mol.target_density/mol.initial_density
        scale = scale ** (1/3)
        s = "fix               DEFORM all deform 1 &\n" + " " * 18
        s += "x scale {:.4f} ".format(1/scale)
        s += "y scale {:.4f} ".format(1/scale)
        s += "z scale {:.4f}".format(1/scale)
        self.fix_deform = s
        self.unfix_deform = "unfix             DEFORM"

    def SetFixShake(self, mol):
        btypes = mol.bond_coeffs[mol.bond_coeffs[:, 1] == 'cons', 0]
        if btypes.shape[0] != 0:
            btypes = ["{}".format(v) for v in btypes]
            btypes = " ".join(btypes)
            s = "fix               SHAKE all shake 0.0001 20 0 b "
            s += btypes
            self.fix_shake = s


il_ljcut = """# {molname}
units             real
atom_style        full
pair_style        lj/cut/coul/long 12.0 12.0
pair_modify       mix geometric tail yes
kspace_style      ewald 1.0e-5

special_bonds     lj/coul 0.0 0.0 0.5
bond_style        harmonic
angle_style       harmonic
dihedral_style    opls
improper_style    cvff
read_data         {basename}.data
replicate         1 1 1

thermo            100
thermo_style      custom step temp etotal press vol density

timestep          1.0
velocity          all create 2000 7777

timestep          1
dump              1 all custom 100 {basename}.lammpstrj &
                  id mol type element x y z
dump_modify       1 element {elems} sort id

# neigh_modify      every 1 delay 0 check yes
# minimize          1.0e-4 1.0e-6 100 1000
# neigh_modify      every 1 delay 10 check yes
# reset_timestep    0

{fix_shake}
{fix_deform}
fix               1 all nvt temp 2000 2000 100
run               1000
{unfix_deform}

kspace_style      pppm 1.0e-5
run               3000
fix               1 all nvt temp 2000 300 10
run               3000
unfix             1
fix               1 all npt temp 300 300 100 iso 1 1 100
run               10000
write_data        {basename}_eq.data
"""

ms_bmh = """# {molname}
units             real
atom_style        full
pair_style        born/coul/long 10.0 10.0
kspace_style      ewald 1.0e-5
variable          T equal 1200

read_data         {basename}.data
replicate         1 1 1

thermo            100
thermo_style      custom step temp etotal press vol density

timestep          1.0
velocity          all create 2000 7777

timestep          1
dump              1 all custom 100 {basename}.lammpstrj &
                    id mol type element x y z
dump_modify       1 element {elems} sort id

# neigh_modify      every 1 delay 0 check yes
# minimize          1.0e-4 1.0e-6 100 1000
# neigh_modify      every 1 delay 10 check yes
# reset_timestep    0

fix               1 all nvt temp 2000 2000 100
run               5000
{fix_deform}
fix               1 all nvt temp 2000 ${{T}} 100
run               5000
{unfix_deform}
kspace_style      pppm 1.0e-5

unfix             1
fix               1 all npt temp ${{T}} ${{T}} 100 iso 1 1 100
run               10000
write_data        {basename}_eq.data
"""

glass_buck = """
units             real
atom_style        full
pair_style        buck/coul/long 11.0 11.0
read_data         {basename}.data
replicate         1 1 1
kspace_style      pppm 1.0e-5
thermo            100
thermo_style      custom step temp etotal press vol density

dump              1 all custom 10 {basename}.lammpstrj id type element x y z
dump_modify       1 element {elems} sort id

velocity          all create 3000 777 dist gaussian

timestep          1.0
{fix_deform}
fix               1 all nvt temp 3000 3000 10
{unfix_deform}
run               10000
unfix             1

fix               1 all npt temp 3000 3000 100 iso 0 0 1000
run               50000
unfix             1
fix               1 all npt temp 3000 300 100 iso 0 0 1000
run               50000
unfix             1
fix               1 all npt temp 300 300 100 iso 0.0 0.0 1000
run               50000
"""
