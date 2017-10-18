## for lammps h5md file #by lsj
class h5md(object):
	#possible value for h5md:{box,position,image,velocity,force,species(type),charge}
	def __init__(self,fname):
		try:
			import h5py
		except ImportError as e:
			raise "'m_nc' requires the 'h5py' package"			
		self.pp=h5py.File(fname, 'r') ["particles"]["all"]
		self.fname=fname
		print("Avaliable pp:",list(self.pp.keys()))
	def get_box(self):
		if("box" in self.pp):
			return(self.pp["box"]["edges"]["value"])
	def get_pos(self):
		if("position" in self.pp):
			return(self.pp["position"]["value"])
	def get_img(self):
		if("image" in self.pp):
			return(self.pp["image"]["value"])
	def get_velocity(self):
		if("velocity" in self.pp):
			return(self.pp["velocity"]["value"])	
	def get_type(self):
		if("species" in self.pp):
			return(self.pp["species"]["value"])
	def close():
		self.pp.close()