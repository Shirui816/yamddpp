## for lammps data file  for atom style without charge #modified by lsj  from MDAnalysis.topology.LAMMPSParser.DATAParser
SECTIONS = set([
    'Atoms',  # Molecular topology sections
    'Velocities',
    'Masses',
    'Ellipsoids',
    'Lines',
    'Triangles',
    'Bodies',
    'Bonds',  # Forcefield sections
    'Angles',
    'Dihedrals',
    'Impropers',
    'Pair',
    'Pair LJCoeffs',
    'Bond Coeffs',
    'Angle Coeffs',
    'Dihedral Coeffs',
    'Improper Coeffs',
    'BondBond Coeffs',  # Class 2 FF sections
    'BondAngle Coeffs',
    'MiddleBondTorsion Coeffs',
    'EndBondTorsion Coeffs',
    'AngleTorsion Coeffs',
    'AngleAngleTorsion Coeffs',
    'BondBond13 Coeffs',
    'AngleAngle Coeffs',
])
# We usually check by splitting around whitespace, so check
# if any SECTION keywords will trip up on this
# and add them
for val in list(SECTIONS):
    if len(val.split()) > 1:
        SECTIONS.add(val.split()[0])

HEADERS = set([
    'atoms',
    'bonds',
    'angles',
    'dihedrals',
    'impropers',
    'atom types',
    'bond types',
    'angle types',
    'dihedral types',
    'improper types',
    'extra bond per atom',
    'extra angle per atom',
    'extra dihedral per atom',
    'extra improper per atom',
    'extra special per atom',
    'ellipsoids',
    'lines',
    'triangles',
    'bodies',
    'xlo xhi',
    'ylo yhi',
    'zlo zhi',
    'xy xz yz',
])


def iterdata(filename):
    with open(filename, 'r') as f:
        for line in f:
            line = line.partition('#')[0].strip()
            if line:
                yield line


import numpy as np


class lmp_data(object):
    def __init__(self, fname):
        self.fname = fname
        """Split a data file into dict of header and sections

		Returns
		-------
		header - dict of header section: value
		sections - dict of section name: content
		"""
        f = list(iterdata(fname))
        starts = [i for i, line in enumerate(f) if line.split()[0] in SECTIONS]
        starts += [None]

        self.header = {}
        for line in f[:starts[0]]:
            for token in HEADERS:
                if line.endswith(token):
                    self.header[token] = line.split(token)[0]
                    continue

        self.sections = {f[l]: f[l + 1:starts[i + 1]] for i, l in enumerate(starts[:-1])}

        self.particles = None
        self.offset = 0
        self.pnum = 0

        if ("Atoms" in self.sections):
            p0 = {}
            self.pnum = len(self.sections["Atoms"])
            for line in self.sections["Atoms"]:
                linel = line.split()
                tag = int(linel[0])
                p0[tag] = linel[1:]
            idxl = sorted(list(p0.keys()))
            self.offset = idxl[0]
            self.particles = np.array([p0[idx] for idx in idxl])

    def get_velocity(self):
        if ("Velocities" in self.sections):
            v0 = {}
            for line in self.sections["Velocities"]:
                linel = line.split()
                tag = int(linel[0])
                v0[tag] = np.array(linel[1:]).astype(np.float32)
            idxl = sorted(list(v0.keys()))
            return (np.array([v0[idx] for idx in idxl]))

    def get_top(self, tag):  # Bonds,Angles,Dihedrals
        if (tag in self.sections):
            top = []
            for line in self.sections[tag]:
                linel = np.array(line.split()[1:]).astype(np.int64)
                top.append(linel)
            top = np.array(top)
            # print(top)
            top[:, 1:] -= self.offset
            type_offset = top[:, 0].min()
            # print(type_offset,self.offset)
            top[:, 0] -= type_offset
            return (top)

    def get_mass_pt(self):
        if ("Masses" in self.sections):
            mass_t = {}
            for line in self.sections["Masses"]:
                linel = line.split()
                mass_t[int(linel[0])] = float(linel[1])
            return (mass_t)

    def get_box(self):
        x = self.header["xlo xhi"].split()
        y = self.header["ylo yhi"].split()
        z = self.header["zlo zhi"].split()
        Lx = abs(float(x[0]) - float(x[1]))
        Ly = abs(float(y[0]) - float(y[1]))
        Lz = abs(float(z[0]) - float(z[1]))
        return (np.array((Lx, Ly, Lz)))

    # the below is for atom style without charge
    def get_img(self):
        if (self.particles[0].shape[0] >= 8):
            return (self.particles[:, 5:8].astype(np.int32))

    def get_pos(self):
        return (self.particles[:, 2:5].astype(np.float64))

    def get_RealP(self):
        pos = self.get_pos()
        img = self.get_img()
        box = self.get_box()
        return (pos + img * box)

    def get_type(self):
        return (self.particles[:, 1].astype(np.int32))

    def get_mass(self):
        mass_t = self.get_mass_pt()
        type = self.get_type()
        massl = []
        for tp in type:
            massl.append(mass_t[tp])
        return (massl)
