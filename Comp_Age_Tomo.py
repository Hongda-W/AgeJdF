"""
Compare the 2-station tomography result with the age-dependent velocity result
to see if the surface wave velocity in the oceanic plate is age-dependent.
"""
import dispDBase
import raytomo
import numpy as np
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import pycpt
import obspy
from obspy.geodetics import locations2degrees, degrees2kilometers
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf, interp2d
from matplotlib.path import Path
import h5py

class CompSurfVel(h5py.File):
	
	def _init_parameters(self,pers=np.array([])):
		"""Initiate several parameters
		"""
		self.attrs.create(name = 'tomo_f', data = np.string_('./ray_tomo_JdF_ocean.h5'))
		self.attrs.create(name = 'dataid', data = np.string_('qc_run_0')) #dataid in the ray_tomo*.h5 file for the dataset of interest
		self.attrs.create(name = 'age_h5', data = np.string_('./age_model_1803.h5'))
		self.attrs.create(name = 'minlon', data = 227., dtype='f')
		self.attrs.create(name = 'maxlon', data = 238., dtype='f')
		self.attrs.create(name = 'minlat', data = 43., dtype='f')
		self.attrs.create(name = 'maxlat', data = 50., dtype='f')
		if pers.size==0:
			pers = np.append( np.arange(7.)*2.+6., np.arange(4.)*3.+20.)
		for per in pers:
			self.create_group(name='%g_sec'%( per ))
		self.attrs.create(name = 'prd_arr', data = pers, dtype='f')
		self.poly_lst = [(360.-126.729,49.292),(360.-125.021,48.105),(360.-124.299,44.963),(360.-124.7,43.203),\
			(360.-126.545,43.586),(360.-131.406,44.8),(360.-130.97,47.),(360.-129.821,49.12)]
		try:
			self.Rbf_func
		except:
			self._get_Rbf_func()
		return
	
	def _get_Rbf_func(self,sed_file='/work3/wang/code_bkup/ToolKit/Models/SedThick/sedthick_world_v2.xyz'):
		thk_xyz = np.loadtxt(sed_file)
		minlat = self.attrs['minlat']
		maxlat = self.attrs['maxlat']
		minlon = self.attrs['minlon']
		maxlon = self.attrs['maxlon']
		lat_pss = np.logical_and(thk_xyz[:,1]>minlat, thk_xyz[:,1]<maxlat)
		lon_pss = np.logical_and(thk_xyz[:,0]>minlon, thk_xyz[:,0]<maxlon)
		pss = np.logical_and(lat_pss, lon_pss)
		sed_lat = thk_xyz[:,1][pss]
		sed_lon = thk_xyz[:,0][pss]
		sed_thk = thk_xyz[:,2][pss]
		self.Rbf_func = Rbf(sed_lat,sed_lon,sed_thk,norm=cal_dist)
		pass
		
	def get_tomo_data(self,threshold=20.):
		"""Get tomography dataset from h5 file
		Parameters:
		tomo_f        --  h5 file that contains the tomography results
		period        --  period of interest
		threshold     --  path density threshold for forming mask
		"""
		dset = raytomo.RayTomoDataSet(self.attrs['tomo_f'])
		for prd in self.attrs['prd_arr']:
			group = self['%g_sec'%( prd )]
			dset.get_data4plot(dataid=self.attrs['dataid'].decode('utf-8'), period=prd)
			pdens = dset.pdens
			mask_pdens = dset.pdens < threshold
			tomo_data = np.ma.masked_array(dset.vel_iso, mask=mask_pdens)
			group.create_dataset(name='tomo_data', data=dset.vel_iso) # phase velocity map
			group.create_dataset(name='tomo_data_msk', data=mask_pdens) # save the mask array seperately. h5 file doesn't support masked array
			group.create_dataset(name='latArr', data=dset.latArr)
			group.create_dataset(name='lonArr', data=dset.lonArr)
		return
	
	def get_res_data(self):
		dset = raytomo.RayTomoDataSet(self.attrs['tomo_f'])
		for period in self.attrs['prd_arr']:
			group = self['%g_sec'%( period )]
			group.create_dataset(name='res_data', data=dset[self.attrs['dataid'].decode('utf-8')+'/%g_sec'%( period )]['residual'].value)
		return
	
	def get_age_arr(self):
		"""Get age array for the grids of the tomography result
		"""
		dset = dispDBase.dispASDF(self.attrs['age_h5'])
		minlat = self.attrs['minlat']
		maxlat = self.attrs['maxlat']
		minlon = self.attrs['minlon']
		maxlon = self.attrs['maxlon']
		dset.set_poly(self.poly_lst,minlon,minlat,maxlon,maxlat)
		dset.read_age_mdl()
		for period in self.attrs['prd_arr']:
			group = self['%g_sec'%( period )]
			lons_orig = group['lonArr'].value
			lons = lons_orig.reshape(lons_orig.size)
			lats = group['latArr'].value.reshape(lons_orig.size)
			cords_vec = np.vstack((lats,lons)).T
			age_Arr = dset.get_ages(cords_vec).reshape(lons_orig.shape)
			mask_age = age_Arr > 180.
			group.create_dataset(name='age_Arr', data=age_Arr)
			group.create_dataset(name='age_Arr_msk', data=mask_age)
		return
		
	def get_sed_thk(self):
		"""Get the sediment thickness for the grids of the tomography result
		"""
		for period in self.attrs['prd_arr']:
			group = self['%g_sec'%( period )]
			sed_Arr = self.Rbf_func(group['latArr'].value, group['lonArr'].value)
			group.create_dataset(name='sed_Arr', data=sed_Arr)
			group.create_dataset(name='sed_Arr_msk', data=group['tomo_data_msk'].value)
		pass
	
	def get_wtr_dep(self,dep_f='./etopo.xyz'):
		""" Get water depth for the grids of the tomography result
		"""
		dep_xyz = np.loadtxt(dep_f)
		dep_lon = dep_xyz[:,0].reshape(self['6_sec']['lonArr'].value.shape[::-1])
		dep_lat = dep_xyz[:,1].reshape(self['6_sec']['lonArr'].value.shape[::-1])
		dep_data = dep_xyz[:,2].reshape(self['6_sec']['lonArr'].value.shape[::-1])
		if not ((self['6_sec']['lonArr'].value - dep_lon.T).sum() == 0 and (self['6_sec']['latArr'].value - dep_lat.T).sum() == 0):
			raise AttributeError('The local etopo file is not compatible with the tomography result')
		for period in self.attrs['prd_arr']:
			group = self['%g_sec'%( period )]
			dep_Arr = dep_data.T
			group.create_dataset(name='dep_Arr', data=dep_Arr)
			group.create_dataset(name='dep_Arr_msk', data=dep_Arr>0)
		pass
	
	def get_c(self,wave='phase'):
		"""Ger c0, c1, c2 for the given period
		"""
		dset = dispDBase.dispASDF(self.attrs['age_h5'])
		# dset = dispDBase.dispASDF('age_model_new.h5')
		for period in self.attrs['prd_arr']:
			str_per = str(int(period)).zfill(2)
			group = self['%g_sec'%( period )]
			c_dic = dset.auxiliary_data.FitResult[str_per][wave].parameters
			c0 = c_dic['c0']
			c1 = c_dic['c1']
			c2 = c_dic['c2']
			age_vel = c0 + c1*np.sqrt(group['age_Arr'].value) + c2*group['age_Arr'].value
			group.create_dataset(name='age_vel', data=age_vel)
			group.create_dataset(name='age_vel_msk', data=group['age_Arr_msk'].value)
			group.attrs.create(name = 'c0', data = c0, dtype='f')
			group.attrs.create(name = 'c1', data = c1, dtype='f')
			group.attrs.create(name = 'c2', data = c2, dtype='f')
		return
	
	def _get_basemap(self, projection='lambert', geopolygons=None, resolution='i', bound=True, hillshade=False):
		"""Get basemap for plotting results
		"""
		# fig=plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
		minlat = self.attrs['minlat']
		maxlat = self.attrs['maxlat']
		minlon = self.attrs['minlon']
		maxlon = self.attrs['maxlon']
		lat_centre = (maxlat+minlat)/2.0
		lon_centre = (maxlon+minlon)/2.0
		if projection=='merc':
			m=Basemap(projection='merc', llcrnrlat=minlat-5., urcrnrlat=maxlat+5., llcrnrlon=minlon-5.,
					  urcrnrlon=maxlon+5., lat_ts=20, resolution=resolution)
			# m.drawparallels(np.arange(minlat,maxlat,dlat), labels=[1,0,0,1])
			# m.drawmeridians(np.arange(minlon,maxlon,dlon), labels=[1,0,0,1])
			m.drawparallels(np.arange(-80.0,80.0,2.0), dashes=[2,2], labels=[1,0,0,0], fontsize=12)
			m.drawmeridians(np.arange(-170.0,170.0,2.0), dashes=[2,2], labels=[0,0,1,0], fontsize=12)
			m.drawstates(color='g', linewidth=2.)
		elif projection=='global':
			m=Basemap(projection='ortho',lon_0=lon_centre, lat_0=lat_centre, resolution=resolution)
			# m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,1])
			# m.drawmeridians(np.arange(-170.0,170.0,10.0), labels=[1,0,0,1])
		elif projection=='regional_ortho':
			m1 = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution='l')
			m = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution=resolution,\
				llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/3.5)
			m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,0],  linewidth=2,  fontsize=20)
			# m.drawparallels(np.arange(-90.0,90.0,30.0),labels=[1,0,0,0], dashes=[10, 5], linewidth=2,  fontsize=20)
			# m.drawmeridians(np.arange(10,180.0,30.0), dashes=[10, 5], linewidth=2)
			m.drawmeridians(np.arange(-170.0,170.0,10.0),  linewidth=2)
		elif projection=='lambert':
			distEW, az, baz=obspy.geodetics.gps2dist_azimuth(minlat, minlon, minlat, maxlon) # distance is in m
			distNS, az, baz=obspy.geodetics.gps2dist_azimuth(minlat, minlon, maxlat+2., minlon) # distance is in m
			m = Basemap(width=distEW, height=distNS, rsphere=(6378137.00,6356752.3142), resolution='l', projection='lcc',\
				lat_1=minlat, lat_2=maxlat, lon_0=lon_centre, lat_0=lat_centre)
			m.drawparallels(np.arange(-80.0,80.0,2.0), linewidth=1, dashes=[2,2], labels=[1,0,0,0], fontsize=12)
			m.drawmeridians(np.arange(-170.0,170.0,2.0), linewidth=1, dashes=[2,2], labels=[0,0,1,0], fontsize=12)
			# m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=0.5, dashes=[2,2], labels=[1,0,0,0], fontsize=5)
			# m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=0.5, dashes=[2,2], labels=[0,0,0,1], fontsize=5)
		m.drawcoastlines(linewidth=1.0)
		m.drawcountries(linewidth=1.0)
		m.drawstates(linewidth=1.0)
		# m.drawmapboundary(fill_color=[1.0,1.0,1.0])
		# m.fillcontinents(lake_color='#99ffff',zorder=0.2)
		# m.drawlsmask(land_color='0.8', ocean_color='#99ffff')
		m.drawmapboundary(fill_color="white")
		if bound:
			try:
				# m.readshapefile('/projects/howa1663/Code/ToolKit/Models/Plates/PB2002_boundaries', name='PB2002_boundaries', drawbounds=True, linewidth=1, color='orange') # draw plate boundary on basemap
				m.readshapefile('/work3/wang/code_bkup/AgeJdF/Plates/PB2002_boundaries', name='PB2002_boundaries', drawbounds=True, \
						linewidth=1, color='orange')
			except IOError:
				print("Couldn't read shape file! Continue without drawing plateboundaries")
		try:
			geopolygons.PlotPolygon(inbasemap=m)
		except:
			pass
		if hillshade:
			from netCDF4 import Dataset
			from matplotlib.colors import LightSource
			etopo1 = Dataset('/work2/wang/Code/ToolKit/ETOPO1_Ice_g_gmt4.grd','r')
			zz = etopo1.variables["z"][:]
			llons = etopo1.variables["x"][:]
			west = llons<0 # mask array with negetive longitudes
			west = 360.*west*np.ones(len(llons))
			llons = llons+west
			llats = etopo1.variables["y"][:]
			etopoz = zz[(llats>(minlat-2))*(llats<(maxlat+2)), :]
			etopoz = etopoz[:, (llons>(minlon-2))*(llons<(maxlon+2))]
			llats = llats[(llats>(minlat-2))*(llats<(maxlat+2))]
			llons = llons[(llons>(minlon-2))*(llons<(maxlon+2))]
			ls = LightSource(azdeg=315, altdeg=45)
			etopoZ = m.transform_scalar(etopoz, llons-360*(llons>180)*np.ones(len(llons)), llats, etopoz.shape[0], etopoz.shape[1])
			ls = LightSource(azdeg=315, altdeg=45)
			m.imshow(ls.hillshade(etopoZ, vert_exag=1.),cmap='gray')
		return m	

	def plot_age(self,period=6., projection='lambert',geopolygons=None, showfig=True, vmin=0, vmax=None, hillshade=True):
		"""Plot age map
		"""
		if hillshade:
			alpha = 0.5
		else:
			alpha =1.
		m = self._get_basemap(projection=projection, geopolygons=geopolygons,hillshade=hillshade)
		group = self['%g_sec'%( period )]
		x, y = m(group['lonArr'].value, group['latArr'].value)
		my_cmap = pycpt.load.gmtColormap('./cv.cpt')
		age_Arr = group['age_Arr'].value
		age_Arr_msk = group['age_Arr_msk'].value
		if vmin == None:
			vmin = np.nanmin(age_Arr[~age_Arr_msk])
			vmin = np.floor(vmin/5.)*5.
		if vmax == None:
			vmax = np.nanmax(age_Arr[~age_Arr_msk])
			vmax = np.ceil(vmax/5.)*5.
		im = m.pcolormesh(x, y, np.ma.masked_array(age_Arr, mask=age_Arr_msk), cmap=my_cmap, shading='gouraud', vmin=vmin, vmax=vmax, alpha=alpha)
		cb = m.colorbar(im, "bottom", size="3%", pad='2%', format='%d')
		cb.set_label('Age (Ma)', fontsize=12, rotation=0)
		cb.set_alpha(1)
		cb.draw_all()
		ax = plt.gca() # only plot the oceanic part for JdF
		# ax.set_xlim(right=x_max)
		if showfig:
			plt.show()
		return
	
	def plot_sed(self,period=6.,projection='lambert',geopolygons=None, showfig=True, vmin=0, vmax=None, hillshade=True):
		"""Plot sedimentary thickness map
		"""
		if hillshade:
			alpha = 0.5
		else:
			alpha =1.
		m = self._get_basemap(projection=projection, geopolygons=geopolygons,hillshade=hillshade)
		group = self['%g_sec'%( period )]
		x, y = m(group['lonArr'].value, group['latArr'].value)
		my_cmap = pycpt.load.gmtColormap('./cv.cpt')
		sed_Arr = group['sed_Arr'].value
		sed_Arr_msk = group['sed_Arr_msk'].value
		if vmin == None:
			vmin = np.nanmin(sed_Arr[~sed_Arr_msk])
			vmin = np.floor(vmin/5.)*5.
		if vmax == None:
			vmax = np.nanmax(sed_Arr[~sed_Arr_msk])
			vmax = np.ceil(vmax/5.)*5.
		im = m.pcolormesh(x, y, np.ma.masked_array(sed_Arr,mask=sed_Arr_msk), cmap=my_cmap, shading='gouraud', vmin=vmin, vmax=vmax, alpha=alpha)
		cb = m.colorbar(im, "bottom", size="3%", pad='2%', format='%d')
		cb.set_label('Sediment thickness (m)', fontsize=12, rotation=0)
		cb.set_alpha(1)
		cb.draw_all()
		ax = plt.gca() # only plot the oceanic part for JdF
		# ax.set_xlim(right=x_max)
		if showfig:
			plt.show()
		return
	
	def plot_tomo_vel(self, period=6., projection='lambert',geopolygons=None, showfig=True, vmin=None, vmax=None, sta=True, hillshade=False):
		"""Plot velocity map from tomography result
		"""
		if hillshade:
			alpha = 0.5
		else:
			alpha =1.
		m = self._get_basemap(projection=projection, geopolygons=geopolygons,hillshade=hillshade)
		group = self['%g_sec'%( period )]
		x, y = m(group['lonArr'].value, group['latArr'].value)
		my_cmap = pycpt.load.gmtColormap('./cv.cpt')
		tomo_data = group['tomo_data'].value
		tomo_data_msk = group['tomo_data_msk'].value
		if vmin == None:
			vmin = np.nanmin(tomo_data[~tomo_data_msk])
			vmin = np.ceil(vmin*20.)/20.
		if vmax == None:
			vmax = np.nanmax(tomo_data[~tomo_data_msk])
			vmax = np.floor(vmax*20.)/20.
		im = m.pcolormesh(x, y, np.ma.masked_array(tomo_data,mask=tomo_data_msk), cmap=my_cmap, shading='gouraud', vmin=vmin, vmax=vmax, alpha=alpha)
		cb = m.colorbar(im, "bottom", size="3%", pad='2%', format='%.2f')
		cb.set_label('vel (km/s)', fontsize=12, rotation=0)
		cb.set_alpha(1)
		cb.draw_all()
		ax = plt.gca() # only plot the oceanic part for JdF
		if sta:
			self.sta_on_plot(ax,m,period)
		# ax.set_xlim(right=x_max)
		fig = plt.gcf()
		fig.suptitle(str(period)+' sec', fontsize=14,y=0.95)
		if showfig:
			plt.show()
		pass
	
	def plot_age_vel(self, period=6., projection='lambert',geopolygons=None, showfig=True, vmin=None, vmax=None, sta=True, hillshade=False):
		"""Plot age-dependent velocity map
		"""
		if hillshade:
			alpha = 0.5
		else:
			alpha =1.
		m = self._get_basemap(projection=projection, geopolygons=geopolygons,hillshade=hillshade)
		group = self['%g_sec'%( period )]
		x, y = m(group['lonArr'].value, group['latArr'].value)
		my_cmap = pycpt.load.gmtColormap('./cv.cpt')
		age_vel = group['age_vel'].value
		age_vel_msk = group['age_vel_msk'].value
		if vmin == None:
			vmin = np.nanmin(age_vel[~age_vel_msk])
			vmin = np.ceil(vmin*20.)/20.
		if vmax == None:
			vmax = np.nanmax(age_vel[~age_vel_msk])
			vmax = np.floor(vmax*20.)/20.
		im = m.pcolormesh(x, y, np.ma.masked_array(age_vel,mask=age_vel_msk), cmap=my_cmap, shading='gouraud', vmin=vmin, vmax=vmax, alpha=alpha)
		cb = m.colorbar(im, "bottom", size="3%", pad='2%', format='%.2f')
		cb.set_label('vel (km/s)', fontsize=12, rotation=0)
		cb.set_alpha(1)
		cb.draw_all()
		ax = plt.gca() # only plot the oceanic part for JdF
		if sta:
			self.sta_on_plot(ax,m,period)
		# ax.set_xlim(right=x_max)
		fig = plt.gcf()
		fig.suptitle(str(period)+' sec', fontsize=14,y=0.95)
		if showfig:
			plt.show()
		pass

	def plot_diff(self, period=6., projection='lambert',geopolygons=None, showfig=True, vmin=None, vmax=None, sta=True, hillshade=False, pct=True):
		"""Plot the difference between tomography result and age-dependent velocity map
		"""
		group = self['%g_sec'%( period )]
		age_vel = group['age_vel'].value
		age_vel_msk = group['age_vel_msk'].value
		tomo_data = group['tomo_data'].value
		tomo_data_msk = group['tomo_data_msk'].value
		mask = np.logical_or(age_vel_msk, tomo_data_msk)
		if pct:
			data = (tomo_data - age_vel) / age_vel* 100
			cb_label = 'vel difference (%)'
		else:
			data = tomo_data - age_vel
			cb_label = 'vel difference (km/s)'
		plt_data = np.ma.masked_array(data, mask=mask)
		if hillshade:
			alpha = 0.5
		else:
			alpha =1.
		m = self._get_basemap(projection=projection, geopolygons=geopolygons,hillshade=hillshade)
		x, y = m(group['lonArr'].value, group['latArr'].value)
		my_cmap = pycpt.load.gmtColormap('./cv.cpt')
		if vmin == None:
			vmin = np.nanmin(plt_data[~mask])
			vmin = np.ceil(vmin*20.)/20.
		if vmax == None:
			vmax = np.nanmax(plt_data[~mask])
			vmax = np.floor(vmax*20.)/20.
		im = m.pcolormesh(x, y, plt_data, cmap=my_cmap, shading='gouraud', vmin=vmin, vmax=vmax, alpha=alpha)
		cb = m.colorbar(im, "bottom", size="3%", pad='2%', format='%.2f')
		cb.set_label(cb_label, fontsize=12, rotation=0)
		cb.set_alpha(1)
		cb.draw_all()
		ax = plt.gca() # only plot the oceanic part for JdF
		# ax.set_xlim(right=x_max)
		if sta:
			self.sta_on_plot(ax,m,period)
		fig = plt.gcf()
		fig.suptitle(str(period)+' sec', fontsize=14,y=0.95)
		if showfig:
			plt.show()
		pass
	
	def plot_age_curve(self, period=6., showfig=True):
		"""Plot vel vs. age for the tomography result
		"""
		group = self['%g_sec'%( period )]
		tomo_data = group['tomo_data'].value
		tomo_data_msk = group['tomo_data_msk'].value
		age_Arr = group['age_Arr'].value
		age_Arr_msk = group['age_Arr_msk'].value
		mask = np.logical_or(tomo_data_msk, age_Arr_msk)
		vel_vec = tomo_data[~mask]
		age_vec = age_Arr[~mask]
		plt.plot(age_vec, vel_vec, 'r.')
		ages = np.linspace(0,age_vec.max(),100)
		vels = group.attrs['c0']+group.attrs['c1']*np.sqrt(ages)+group.attrs['c2']*ages
		plt.plot(ages, vels, 'b-')
		plt.xlim(xmin=0)
		plt.xlabel('Age (Ma)', fontsize=14)
		plt.ylabel('vel (km/s)', fontsize=14)
		fig = plt.gcf()
		fig.suptitle(str(period)+' sec', fontsize=14)
		if showfig:
			plt.show()
		pass
	
	def plot_sed_curve(self, period=6., showfig=True):
		"""Plot vel vs. sediment thickness for the tomography result
		"""
		group = self['%g_sec'%( period )]
		tomo_data = group['tomo_data'].value
		mask = group['tomo_data_msk'].value
		sed_Arr = group['sed_Arr'].value
		vel_vec = tomo_data[~mask]
		sed_vec = sed_Arr[~mask]
		plt.plot(sed_vec, vel_vec, 'r.')
		plt.xlim(xmin=0)
		plt.xlabel('Sediment thickness (m)', fontsize=14)
		plt.ylabel('vel (km/s)', fontsize=14)
		fig = plt.gcf()
		fig.suptitle(str(period)+' sec', fontsize=14)
		if showfig:
			plt.show()
		pass
	
	def plot_dep_curve(self, period=6., showfig=True):
		"""Plot tomography vel results vs. oceanic water depth
		"""
		group = self['%g_sec'%( period )]
		tomo_data = group['tomo_data'].value
		dep_Arr = group['dep_Arr'].value
		mask = np.logical_or(group['tomo_data_msk'].value, group['dep_Arr_msk'].value)
		vel_vec = tomo_data[~mask]
		dep_vec = dep_Arr[~mask]
		plt.plot(dep_vec, vel_vec, 'r.')
		plt.xlim(xmax=0)
		plt.xlabel('Water depth (m)', fontsize=14)
		plt.ylabel('vel (km/s)', fontsize=14)
		fig = plt.gcf()
		fig.suptitle(str(period)+' sec', fontsize=14)
		if showfig:
			plt.show()
		pass

	def sta_on_plot(self, ax, basemap, period, ms=6):
		"""Add stations on plot
		"""
		group = self['%g_sec'%( period )]
		if group['res_data'].value.size == 0:
			self.get_res_data()
		cords1 = group['res_data'].value[:,1:3]
		cords2 = group['res_data'].value[:,3:5]
		sta_cords,cnts = np.unique(np.concatenate((cords1,cords2),axis=0), return_counts=True, axis=0)
		x, y = basemap(sta_cords[:,1], sta_cords[:,0])
		ax.plot(x,y,'^', color='gray',alpha=.5, ms=ms)
		pass
	
def cal_dist(origs,dests):
	""" Calculate the great-circle distance between points on geographic map
	Parameters:
		origs -- 2*N array, with row 0 latitude, row 1 longitude
		dests -- 2*N array, with row 0 latitude, row 1 longitude
	"""
	radius = 6371.009 # km
	if origs.ndim:
		lat1, lon1 = origs
		lat2, lon2 = dests
	else:
		lat1 = origs[0,:]
		lon1 = origs[1,:]
		lat2 = dests[0,:]
		lon2 = dests[1,:]
	dlat = (lat2-lat1) / 180. * np.pi
	dlon = (lon2-lon1) / 180. * np.pi
	a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(lat1 / 180. * np.pi) \
		* np.cos(lat2 / 180. * np.pi) * np.sin(dlon/2) * np.sin(dlon/2)
	c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
	return radius * c