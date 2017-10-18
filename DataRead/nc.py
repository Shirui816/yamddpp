## for lammps netcdf4 file #by lsj 
class m_nc_mod(object):
    def __init__(self, fname):
        try:
            import netCDF4 as netcdf
        except ImportError as e:
            raise "'m_nc' requires the 'netCDF4' package"
        self.nc = netcdf.Dataset(fname, "a")
        self.pp = self.nc.variables

    def cr_iamge(self):
        if ("ix" in self.pp and "iy" in self.pp and "iz" in self.pp):
            image = self.nc.createVariable("image", "i4", ("frame", "atom", "spatial"))
            image[:, :, 0] = self.pp["ix"][:][:]
            image[:, :, 1] = self.pp["iy"][:][:]
            image[:, :, 2] = self.pp["iz"][:][:]
            return (image)


class nc_image(object):
    def __init__(self, nc_pp):
        try:
            import numpy as np
        except ImportError as e:
            raise "'nc_image' requires the 'numpy' package"
        if ("ix" in nc_pp and "iy" in nc_pp and "iz" in nc_pp):
            self.ix = nc_pp["ix"]
            self.iy = nc_pp["iy"]
            self.iz = nc_pp["iz"]
            self.numframes = self.ix.shape[0]
            self.np = np
        else:
            raise "'nc_image' requires ix,iy,iz all in the 'nc_pp'"

    def __getitem__(self, frame):
        if (type(frame) != int):
            raise TypeError
        if (frame < 0):
            frame = len(self) + frame
        if (frame < 0) or (frame >= len(self)):
            raise IndexError
        return (self.np.stack((self.ix[frame], self.iy[frame], self.iz[frame]), axis=1))

    def __len__(self):
        return self.numframes


class nc_RealP(object):
    def __init__(self, nc):
        self.pos = nc.get_pos()
        self.img = nc.get_img()
        self.box = nc.get_box()
        if (not self.pos):
            raise ("Failed in getting position")
        if (not self.img):
            raise ("Failed in getting image")
        if (not self.box):
            raise ("Failed in getting box")
        self.numframes = self.pos.shape[0]

    def __getitem__(self, frame):
        if (type(frame) != int):
            raise TypeError
        if (frame < 0):
            frame = len(self) + frame
        if (frame < 0) or (frame >= len(self)):
            raise IndexError
        return ((self.pos[frame] + self.box[frame] * self.img[frame]))

    def __len__(self):
        return self.numframes


class m_nc(object):
    def __init__(self, fname):
        try:
            import netCDF4 as netcdf
        except ImportError as e:
            raise "'m_nc' requires the 'netCDF4' package"
        self.nc = netcdf.Dataset(fname)
        self.fname = fname
        self.natoms = len(self.nc.dimensions['atom'])
        self.numframes = len(self.nc.dimensions['frame'])
        self.pp = self.nc.variables
        self.pp_keys = list(self.pp.keys())
        print("Avaliable pp:", self.pp_keys)

    def get_pos(self):
        if ("coordinates" in self.pp):
            return (self.pp['coordinates'])

    def get_img(self):
        if ("image" in self.pp):
            return (self.pp['image'])
        else:
            return (nc_image(self.pp))

    def get_mass(self):
        if ("mass" in self.pp):
            return (self.pp['mass'])

    def get_box(self):
        if ('cell_lengths' in self.nc.variables):
            cell_lengths = self.nc.variables['cell_lengths']
            return (cell_lengths)

    def get_time(self):
        if ('time' in self.nc.variables):
            time = self.nc.variables['time']
            return (time)

    def close(self):
        self.nc.close()
