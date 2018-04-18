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
from scipy.interpolate import Rbf, interp1d, interp2d, NearestNDInterpolator, griddata
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
			try:
				self.create_group(name='%g_sec'%( per ))
			except:
				pass
		self.attrs.create(name = 'prd_arr', data = pers, dtype='f')
		self.poly_lst = [(360.-126.729,49.292),(360.-125.021,48.105),(360.-124.299,44.963),(360.-124.7,43.203),\
			(360.-126.545,43.586),(360.-131.406,44.8),(360.-130.97,47.),(360.-129.821,49.12)]
		try:
			self.Rbf_func
		except:
			self._get_Rbf_func()
		return
	
	def _get_Rbf_func(self,sed_file='/work3/wang/code_bkup/ToolKit/Models/SedThick/sedthick_world_v2.xyz'):
		""" Get the interpolation function for the sediment thickness data using radial basis function
		"""
		thk_xyz = np.loadtxt(sed_file)
		minlat = self.attrs['minlat']
		maxlat = self.attrs['maxlat']
		minlon = self.attrs['minlon']
		maxlon = self.attrs['maxlon']
		lat_pss = np.logical_and(thk_xyz[:,1]>=minlat, thk_xyz[:,1]<=maxlat)
		lon_pss = np.logical_and(thk_xyz[:,0]>=minlon, thk_xyz[:,0]<=maxlon)
		pss = np.logical_and(lat_pss, lon_pss)
		sed_lat = thk_xyz[:,1][pss]
		sed_lon = thk_xyz[:,0][pss]
		sed_thk = thk_xyz[:,2][pss]
		self.Rbf_func = Rbf(sed_lat,sed_lon,sed_thk,norm=cal_dist)
		pass
	
	def _get_age_func(self)	:
		""" Get the function from Nearest interpolation for the oceanic age field
		"""
		age_nc_Arr = self['age_nc_Arr'].value
		age_lon_Vec = self['age_lon_Vec'].value
		age_lat_Vec = self['age_lat_Vec'].value
		xx, yy = np.meshgrid(age_lon_Vec, age_lat_Vec) # xx for longitude, yy for latitude
		xx = xx.reshape(xx.size) #nearest
		yy = yy.reshape(yy.size)
		x = np.column_stack((xx,yy))
		y = age_nc_Arr.reshape(age_nc_Arr.size)
		self.age_func = NearestNDInterpolator(x,y,rescale=False)
		return
		
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
	
	def get_age_arr(self,renew=False):
		"""Get age array for the grids of the tomography result
		"""
		dset = dispDBase.dispASDF(self.attrs['age_h5'])
		minlat = self.attrs['minlat']
		maxlat = self.attrs['maxlat']
		minlon = self.attrs['minlon']
		maxlon = self.attrs['maxlon']
		dset.set_poly(self.poly_lst,minlon,minlat,maxlon,maxlat)
		dset.read_age_mdl()
		self.create_dataset(name='age_nc_Arr', data=dset.age_data)
		self.create_dataset(name='age_lon_Vec', data=dset.age_lon)
		self.create_dataset(name='age_lat_Vec', data=dset.age_lat)
		for period in self.attrs['prd_arr']:
			group = self['%g_sec'%( period )]
			lons = group['lonArr'].value
			lats = group['latArr'].value
			age_Arr = dset.get_ages(lons.reshape(lons.size),lats.reshape(lats.size))
			age_Arr = age_Arr.reshape(lats.shape)
			mask_age = age_Arr > 180.
			if renew:
				del group['age_Arr']
				del group['age_Arr_msk']
			group.create_dataset(name='age_Arr', data=age_Arr)
			group.create_dataset(name='age_Arr_msk', data=mask_age)
		return
	
	def _cal_age_grad(self):
		""" Calculate the age gradient vector on original age grid points
		"""
		age_nc_Arr = self['age_nc_Arr'].value
		age_nc_Arr_msk = age_nc_Arr > 180.
		age_lon_Vec = self['age_lon_Vec'].value
		age_lat_Vec = self['age_lat_Vec'].value
		deriv_lat, deriv_lon = np.gradient(age_nc_Arr, age_lat_Vec, age_lon_Vec)
		msk_left = np.zeros(age_nc_Arr_msk.shape)
		msk_right = np.zeros(age_nc_Arr_msk.shape)
		msk_up = np.zeros(age_nc_Arr_msk.shape)
		msk_down = np.zeros(age_nc_Arr_msk.shape)
		msk_left[:,0:-1] = age_nc_Arr_msk[:,1:]
		msk_right[:,1:] = age_nc_Arr_msk[:,0:-1]
		msk_up[0:-1,:] = age_nc_Arr_msk[1:,:]
		msk_down[1:,:] = age_nc_Arr_msk[0:-1,:]
		deriv_msk = np.array(msk_left+msk_right+msk_up+msk_down,dtype=bool)
		self.create_dataset(name='deriv_lat', data=deriv_lat)
		self.create_dataset(name='deriv_lon', data=deriv_lon)
		self.create_dataset(name='deriv_msk', data=deriv_msk)
		pass
		
	def get_age_grad(self,renew=False):
		""" get the age gradient at each grid point
		"""
		try:
			driv_lat = self['deriv_lat'].value
			driv_lon = self['deriv_lon'].value
			driv_msk = self['deriv_msk'].value
		except:
			self._cal_age_grad()
		deriv_lat = self['deriv_lat'].value
		deriv_lon = self['deriv_lon'].value
		deriv_msk = self['deriv_msk'].value
		age_lon_Vec = self['age_lon_Vec'].value
		age_lat_Vec = self['age_lat_Vec'].value
		xx, yy = np.meshgrid(age_lon_Vec, age_lat_Vec) # xx for longitude, yy for latitude
		xx = xx.reshape(xx.size)
		yy = yy.reshape(yy.size)
		f_deriv_lat = NearestNDInterpolator(np.column_stack((xx,yy)),deriv_lat.reshape(deriv_lat.size),rescale=False)
		f_deriv_lon = NearestNDInterpolator(np.column_stack((xx,yy)),deriv_lon.reshape(deriv_lon.size),rescale=False)
		f_deriv_msk = NearestNDInterpolator(np.column_stack((xx,yy)),deriv_msk.reshape(deriv_msk.size),rescale=False)
		for period in self.attrs['prd_arr']:
			group = self['%g_sec'%( period )]
			lons_orig = group['lonArr'].value
			lons = lons_orig.reshape(lons_orig.size)
			lats = group['latArr'].value.reshape(lons_orig.size)
			deriv_lat_Arr = f_deriv_lat(np.column_stack((lons,lats))).reshape(lons_orig.shape)
			deriv_lon_Arr = f_deriv_lon(np.column_stack((lons,lats))).reshape(lons_orig.shape)
			deriv_msk_Arr = f_deriv_msk(np.column_stack((lons,lats))).reshape(lons_orig.shape)
			if renew:
				del group['age_deriv_lat_Arr']
				del group['age_deriv_lon_Arr']
				del group['age_deriv_msk_Arr']
			group.create_dataset(name='age_deriv_lat_Arr', data=deriv_lat_Arr)
			group.create_dataset(name='age_deriv_lon_Arr', data=deriv_lon_Arr)
			group.create_dataset(name='age_deriv_msk_Arr', data=deriv_msk_Arr)
		pass
	
	def _cons_traj(self,lon,lat,period):
		""" Based on the given longitude & latitude, construct a trajectory for connnecting age gradient vectors on the map.
		Doesn't work very well. 
		Parameter:
					lon,lat -- longitude and latitude of the input point
					period  -- the period of the interested dataset
		
		Return: return the mask array for selected grids
		"""
		group = self['%g_sec'%( period )]
		lonArr = group['lonArr'].value
		latArr = group['latArr'].value
		ageArr = group['age_Arr'].value
		age_drv_lon = group['age_deriv_lon_Arr']
		age_drv_lat = group['age_deriv_lat_Arr']
		age_drv_msk = group['age_deriv_msk_Arr']
		tomo_data_msk = group['tomo_data_msk'].value
		mask_init = np.logical_or(tomo_data_msk, age_drv_msk)
		mask = np.array(np.ones(lonArr.shape),dtype=bool)
		i_s, j_s = self._find_nearest_grid(lon,lat,period) # find the 1st and 2nd index of the start point on grid
		if mask_init[i_s,j_s]:
			raise ValueError("The starting point is out bounds of the resulting map")
		mask[i_s,j_s] = False
		# go along increasing age direction
		i,j = _walk(age_drv_lon,age_drv_lat,i_s,j_s,1)
		while (0<i<lonArr.shape[0] and 0<j<lonArr.shape[1]):
			if mask_init[i,j]:
				break
			else:
				mask[i,j] = False
			i,j = _walk(age_drv_lon,age_drv_lat,i,j,1)
			
		# go along descending age direction
		i_2,j_2 = _walk(age_drv_lon,age_drv_lat,i_s,j_s,-1)
		mask[i_2,j_2] = False
		while (0<i_2<lonArr.shape[0] and 0<j_2<lonArr.shape[1]):
			if mask_init[i_2,j_2]:
				break
			else:
				mask[i_2,j_2] = False
			age_b = ageArr[i_2,j_2]
			i_b,j_b = i_2,j_2
			i_2,j_2 = _walk(age_drv_lon,age_drv_lat,i_2,j_2,-1)
			if ageArr[i_2,j_2] > age_b: # find the youngest age
				i_3,j_3 = i_b,j_b
				break
		try:
			i_3
		except NameError:
			return mask
		
		# go along increasing age from the found youngest age
		while (0<i_3<lonArr.shape[0] and 0<j_3<lonArr.shape[1]):
			if mask_init[i_3,j_3]:
				break
			else:
				mask[i_3,j_3] = False
			i_3,j_3 = _walk(age_drv_lon,age_drv_lat,i_3,j_3,1)
		return mask
	
	def _cons_traj_stream(self,period=6.,lon=232.,lat=48.,N=20,projection='lambert'):
		""" Construct trajectory from the stream plot. The output are coordinates for the points interpolated along the desired stream line
		"""
		strm = self.plot_age_deriv(period=period, lon=lon, lat=lat, streamline=True, projection=projection, showfig=False)
		m = self._get_basemap(projection=projection)
		plt.close('all')
		path_pnts_arr = np.vstack(strm.lines.get_segments()) 
		lons, lats = m(path_pnts_arr[::2,0], path_pnts_arr[::2,1],inverse=True) # every point on the path was repeated at leat once
		try:
			self.age_func
		except:
			self._get_age_func()
		dist = np.sqrt((lons[1:] - lons[:-1]) ** 2 + (lats[1:] - lats[:-1]) ** 2)
		transp = np.append(np.array(True),(dist > dist.max() / 100))
		ind = np.argsort(lons[transp])
		f = interp1d(lons[transp][ind],lats[transp][ind],kind='cubic')
		lons_out = np.linspace(lons[transp].min(), lons[transp].max(), N)
		lats_out = f(lons_out)
		lons_out[lons_out<0] = lons_out[lons_out<0] + 360.
		ages = self.age_func(np.column_stack((lons_out,lats_out)))
		mask = ages < 0.5 # discard point whose age is smaller than 0.5 myrs
		return lons_out[~mask], lats_out[~mask]

	def _find_nearest_grid(self,lon,lat,period):
		""" Find the grid point on the result lat&lon grid that's closest to the given coordinates
		"""
		group = self['%g_sec'%( period )]
		lonArr = group['lonArr'].value
		latArr = group['latArr'].value
		diff_Arr = np.dstack((lonArr, latArr)) - np.array([lon, lat]) # 3-d array ( , ,2)
		diff_Arr[:,:,0] = diff_Arr[:,:,0] * np.cos(lat/180.*np.pi)
		dist_sq = np.sum(diff_Arr**2,axis=-1)
		ind1, ind2 = np.where(dist_sq == np.min(dist_sq))
		return ind1[0], ind2[0]
		
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
		if not ((self['6_sec']['lonArr'].value - dep_lon.T).sum() < 0.01 and (self['6_sec']['latArr'].value - dep_lat.T).sum() < 0.01):
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
	
	def read_curve_Ye(self, fname='./Curve_Ye.lst'):
		""" Read Ye's age-dependent phase velocity curve from his 2013 paper, Fig 3
		"""
		try:
			self.create_group(name='Curve_Ye')
		except:
			pass
		group = self['Curve_Ye']
		f = open(fname, 'r')
		lines = f.readlines()
		for line in lines:
			try:
				age, vel = line.strip().split()
				arr = np.append(arr, np.array([[float(age.replace(',','')), float(vel)]]), axis=0)
			except ValueError:
				period = line.strip().replace('# ','').replace(' sec phase','')
				if bool(period):
					arr = np.array([[],[]]).T
					name = period+'_sec_phase'
				else:
					group.create_dataset(name=name, data=arr)
		try:
			group[name]
		except:
			group.create_dataset(name=name, data=arr)
		pass
	
	def get_traj_data(self,lon1=232,lat1=48.,lon2=232.,lat2=46.5,lon3=232.,lat3=45.,N=20):
		""" Get data along trajecoty, not done
		Parameters:
		"""
		group = self.create_group(name='3_trajs')
		
		try:
			self.age_func
		except:
			self._get_age_func()
		for period in self.attrs['prd_arr']:
			lons_out1, lats_out1 = self._cons_traj_stream(period=period,lon=lon1,lat=lat1,N=N)
			lons_out2, lats_out2 = self._cons_traj_stream(period=period,lon=lon2,lat=lat2,N=N)
			lons_out3, lats_out3 = self._cons_traj_stream(period=period,lon=lon3,lat=lat3,N=N)
			ages1 = self.age_func(np.column_stack((lons_out1,lats_out1)))
			ages2 = self.age_func(np.column_stack((lons_out2,lats_out2)))
			ages3 = self.age_func(np.column_stack((lons_out3,lats_out3)))
			seds1 = self.Rbf_func(lats_out1, lons_out1)
			seds2 = self.Rbf_func(lats_out2, lons_out2)
			seds3 = self.Rbf_func(lats_out3, lons_out3)
			group_per = self['%g_sec'%( period )]
			tomo_data = group_per['tomo_data'].value
			tomo_data_msk = group_per['tomo_data_msk'].value
			lonArr = group_per['lonArr'].value
			latArr = group_per['latArr'].value
			x1 = lonArr[~tomo_data_msk]
			y1 = latArr[~tomo_data_msk]
			z1 = tomo_data[~tomo_data_msk]
			vels1 = griddata(np.column_stack((x1,y1)), z1, (lons_out1, lats_out1), method='linear', fill_value=0.)
			vels2 = griddata(np.column_stack((x1,y1)), z1, (lons_out2, lats_out2), method='linear', fill_value=0.)
			vels3 = griddata(np.column_stack((x1,y1)), z1, (lons_out3, lats_out3), method='linear', fill_value=0.)
			mask1 = np.logical_or(ages1>180, vels1<0.5)
			mask2 = np.logical_or(ages2>180, vels2<0.5)
			mask3 = np.logical_or(ages3>180, vels3<0.5)
			subgrp = group.create_group('%g_sec'%( period ))
			subgrp.create_dataset(name='traj_N', data=np.column_stack((lons_out1,lats_out1)))
			subgrp.create_dataset(name='traj_M', data=np.column_stack((lons_out2,lats_out2)))
			subgrp.create_dataset(name='traj_S', data=np.column_stack((lons_out3,lats_out3)))
			subgrp.create_dataset(name='mask_N', data=mask1)
			subgrp.create_dataset(name='mask_M', data=mask2)
			subgrp.create_dataset(name='mask_S', data=mask3)
			subgrp.create_dataset(name='vels_N', data=vels1)
			subgrp.create_dataset(name='vels_M', data=vels2)
			subgrp.create_dataset(name='vels_S', data=vels3)
			try:
				curve_ye = self['Curve_Ye']['%g_sec_phase'%( period )].value
				subgrp.create_dataset(name="Ye_2013", data=curve_ye)
			except:
				pass
			try:
				curve_ye_1 = self['Curve_Ye']['%g_sec_phase'%( period-1 )].value
				subgrp.create_dataset(name="Ye_%g_sec"%(period-1), data=curve_ye_1)
			except:
				pass
			dep_data = group_per['dep_Arr'].value
			dep_data_msk = group_per['dep_Arr_msk'].value
			x_dep = lonArr[~dep_data_msk]
			y_dep = latArr[~dep_data_msk]
			z_dep = dep_data[~dep_data_msk]
			deps1 = griddata(np.column_stack((x_dep,y_dep)), z_dep, (lons_out1, lats_out1), method='linear', fill_value=0.)
			deps2 = griddata(np.column_stack((x_dep,y_dep)), z_dep, (lons_out2, lats_out2), method='linear', fill_value=0.)
			deps3 = griddata(np.column_stack((x_dep,y_dep)), z_dep, (lons_out3, lats_out3), method='linear', fill_value=0.)
			subgrp.create_dataset(name='deps_N', data=deps1)
			subgrp.create_dataset(name='deps_M', data=deps2)
			subgrp.create_dataset(name='deps_S', data=deps3)
			subgrp.create_dataset(name='seds_N', data=seds1)
			subgrp.create_dataset(name='seds_M', data=seds2)
			subgrp.create_dataset(name='seds_S', data=seds3)
			subgrp.create_dataset(name='ages_N', data=ages1)
			subgrp.create_dataset(name='ages_M', data=ages2)
			subgrp.create_dataset(name='ages_S', data=ages3)
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
				#m.readshapefile('/work3/wang/code_bkup/AgeJdF/Plates/PB2002_boundaries', name='PB2002_boundaries', drawbounds=True, \
				#		linewidth=1, color='orange')
				m.readshapefile('/work3/wang/code_bkup/ToolKit/Models/UT_Plates/ridge',name='ridge',drawbounds=True, linewidth=1, color='orange')
				m.readshapefile('/work3/wang/code_bkup/ToolKit/Models/UT_Plates/trench',name='trench',drawbounds=True, linewidth=1, color='orange')
				m.readshapefile('/work3/wang/code_bkup/ToolKit/Models/UT_Plates/transform',name='transform',drawbounds=True, linewidth=1, color='orange')
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

	def plot_age(self,period=6., projection='lambert',geopolygons=None, showfig=True, vmin=0, vmax=None, hillshade=False):
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
		return m
	
	def plot_sed(self,period=6.,projection='lambert',geopolygons=None, showfig=True, vmin=0, vmax=None, hillshade=False):
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
	
	def plot_dep(self,period=6.,projection='lambert',geopolygons=None, showfig=True, vmin=None, vmax=None, hillshade=False):
		"""Plot water depth map
		"""
		if hillshade:
			alpha = 0.5
		else:
			alpha =1.
		m = self._get_basemap(projection=projection, geopolygons=geopolygons,hillshade=hillshade)
		group = self['%g_sec'%( period )]
		x, y = m(group['lonArr'].value, group['latArr'].value)
		my_cmap = pycpt.load.gmtColormap('/work3/wang/code_bkup/ToolKit/Models/ETOPO1/ETOPO1.cpt')
		dep_Arr = group['dep_Arr'].value
		if vmin == None:
			vmin = np.nanmin(dep_Arr)
			vmin = np.floor(vmin/100.)*100.
		if vmax == None:
			vmax = np.nanmax(dep_Arr)
			vmax = np.ceil(vmax/100.)*100.
		im = m.pcolormesh(x, y, dep_Arr, cmap=my_cmap, shading='gouraud', vmin=vmin, vmax=vmax, alpha=alpha)
		cb = m.colorbar(im, "bottom", size="3%", pad='2%', format='%d')
		cb.set_label('Topography (m)', fontsize=12, rotation=0)
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
		#x_cobb,y_cobb = m(-130.0000,46.0000) #cobb hotspot
		#ax.plot(x_cobb,y_cobb,'o', color='white',alpha=.6,ms=8,mec='white')
		if sta:
			self.sta_on_plot(ax,m,period)
		# ax.set_xlim(right=x_max)
		fig = plt.gcf()
		fig.suptitle(str(period)+' sec', fontsize=14,y=0.95)
		if showfig:
			plt.show()
		return m
	
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
	
	def plot_age_curve_traj(self, period=6., lon=323, lat=48, showfig=True):
		"""Plot vel vs. age for the tomography result
		"""
		group = self['%g_sec'%( period )]
		mask = self._cons_traj(lon=lon,lat=lat,period=period)
		tomo_data = group['tomo_data'].value
		age_Arr = group['age_Arr'].value
		vel_vec = tomo_data[~mask]
		age_vec = age_Arr[~mask]
		plt.plot(age_vec, vel_vec, 'r.')
		plt.xlim(xmin=0)
		plt.xlabel('Age (Ma)', fontsize=14)
		plt.ylabel('vel (km/s)', fontsize=14)
		fig = plt.gcf()
		fig.suptitle(str(period)+' sec', fontsize=14)
		if showfig:
			plt.show()
		pass
	
	def plot_age_curve_traj_2(self, period=6., lon=323, lat=48, N=20, showfig=True):
		"""Plot vel vs. age for the tomography result
		"""
		lons_out, lats_out = self._cons_traj_stream(period=period,lon=lon,lat=lat,N=N)
		try:
			self.age_func
		except:
			self._get_age_func()
		ages = self.age_func(np.column_stack((lons_out,lats_out)))
		group = self['%g_sec'%( period )]
		tomo_data = group['tomo_data'].value
		tomo_data_msk = group['tomo_data_msk'].value
		lonArr = group['lonArr'].value
		latArr = group['latArr'].value
		x1 = lonArr[~tomo_data_msk]
		y1 = latArr[~tomo_data_msk]
		z1 = tomo_data[~tomo_data_msk]
		# f_tomo = interp2d(x1, y1, z1, fill_value=0.)
		# vels = (f_tomo(lons_out, lats_out)).diagonal()
		vels = griddata(np.column_stack((x1,y1)), z1, (lons_out, lats_out), method='linear', fill_value=0.)
		mask = np.logical_or(ages>180, vels==0)
		fig1 = plt.figure(1)
		plt.plot(ages[~mask], vels[~mask], 'r.')
		plt.xlim(xmin=0)
		plt.xlabel('Age (Ma)', fontsize=14)
		plt.ylabel('vel (km/s)', fontsize=14)
		fig1.suptitle(str(period)+' sec', fontsize=14)
		if showfig:
			fig1.show()
		fig2 = plt.figure(2)
		m = self.plot_age(period=period, projection='lambert',geopolygons=None, showfig=False, vmin=0, vmax=None, hillshade=False)
		x2, y2 = m(lons_out, lats_out)
		ax = fig2.gca()
		ax.plot(x2[~mask], y2[~mask],color='gray')
		ax.plot(x2[~mask], y2[~mask],'.', color='gray')
		fig2.show()
		fig3 = plt.figure(3)
		m = self.plot_tomo_vel(period=period, projection='lambert',geopolygons=None, showfig=False, vmin=None, vmax=None, sta=True, hillshade=False)
		x3, y3 = m(lons_out, lats_out)
		ax3 = fig3.gca()
		ax3.plot(x3[~mask], y3[~mask],color='gray')
		ax3.plot(x3[~mask], y3[~mask],'.',color='gray')
		fig3.show()
		pass
	
	def plot_3_traj(self, period=6.):
		""" Plot the 3 North, Middle and South trajatory on top of tomography result &/or age map
		"""
		subgrp = self['3_trajs']['%g_sec'%( period )]
		lons_out1 = subgrp['traj_N'].value[:,0]
		lats_out1 = subgrp['traj_N'].value[:,1]
		lons_out2 = subgrp['traj_M'].value[:,0]
		lats_out2 = subgrp['traj_M'].value[:,1]
		lons_out3 = subgrp['traj_S'].value[:,0]
		lats_out3 = subgrp['traj_S'].value[:,1]
		mask1 = subgrp['mask_N'].value
		mask2 = subgrp['mask_M'].value
		mask3 = subgrp['mask_S'].value
		vels1 = subgrp['vels_N'].value
		vels2 = subgrp['vels_M'].value
		vels3 = subgrp['vels_S'].value
		deps1 = subgrp['deps_N'].value
		deps2 = subgrp['deps_M'].value
		deps3 = subgrp['deps_S'].value
		ages1 = subgrp['ages_N'].value
		ages2 = subgrp['ages_M'].value
		ages3 = subgrp['ages_S'].value
		seds1 = subgrp['seds_N'].value
		seds2 = subgrp['seds_M'].value
		seds3 = subgrp['seds_S'].value
		
		fig1 = plt.figure(1)
		m = self.plot_age(period=period, projection='lambert',geopolygons=None, showfig=False, vmin=0, vmax=None, hillshade=False)
		x1_1, y1_1 = m(lons_out1, lats_out1)
		x1_2, y1_2 = m(lons_out2, lats_out2)
		x1_3, y1_3 = m(lons_out3, lats_out3)
		ax1 = fig1.gca()
		ax1.plot(x1_1[~mask1], y1_1[~mask1],marker='.', linestyle='-', color='gray')
		ax1.plot(x1_2[~mask2], y1_2[~mask2],marker='.', linestyle='-', color='gray')
		ax1.plot(x1_3[~mask3], y1_3[~mask3],marker='.', linestyle='-', color='gray')
		fig1.show()
		
		fig2 = plt.figure(2)
		m = self.plot_tomo_vel(period=period, projection='lambert',geopolygons=None, showfig=False, vmin=None, vmax=None, sta=True, hillshade=False)
		x2_1, y2_1 = m(lons_out1, lats_out1)
		x2_2, y2_2 = m(lons_out2, lats_out2)
		x2_3, y2_3 = m(lons_out3, lats_out3)
		ax2 = fig2.gca()
		ax2.plot(x2_1[~mask1], y2_1[~mask1],marker='.', linestyle='-', color='gray')
		ax2.plot(x2_2[~mask2], y2_2[~mask2],marker='.', linestyle='-', color='gray')
		ax2.plot(x2_3[~mask3], y2_3[~mask3],marker='.', linestyle='-', color='gray')
		fig2.show()
		
		fig3 = plt.figure(3)
		vel_min = np.append(np.append(vels1[~mask1],vels2[~mask2]),vels3[~mask3]).min()
		vel_max = np.append(np.append(vels1[~mask1],vels2[~mask2]),vels3[~mask3]).max()
		plt.plot(ages1[~mask1], vels1[~mask1], marker='o', linestyle='-', color='red', label='N')
		plt.plot(ages2[~mask2], vels2[~mask2], marker='o', linestyle='-', color='green', label='M')
		plt.plot(ages3[~mask3], vels3[~mask3], marker='o', linestyle='-', color='blue', label='S')
		try:
			curve_ye = subgrp["Ye_2013"].value
			plt.plot(curve_ye[:,0], curve_ye[:,1], color='gray', label='Ye_2013')
		except:
			try:
				curve_ye = subgrp["Ye_%g_sec"%(period-1)]
				plt.plot(curve_ye[:,0], curve_ye[:,1], color='gray', label="Ye_%g_sec"%(period-1))
			except:
				pass
		plt.legend(fontsize=12)
		plt.xlim(xmin=0.5)
		plt.ylim(np.floor(vel_min*2.)/2.,np.ceil(vel_max*2.)/2.)
		plt.xlabel('Age (Ma)', fontsize=14)
		plt.ylabel('vel (km/s)', fontsize=14)
		fig3.suptitle(str(period)+" sec", fontsize=14)
		fig3.show()
		
		fig4 = plt.figure(4)
		dep_min = np.append(np.append(deps1,deps2),deps3).min()
		dep_max = np.append(np.append(deps1,deps2),deps3).max()
		plt.plot(ages1[~mask1], deps1[~mask1], marker='o', linestyle='-', color='red', label='N')
		plt.plot(ages2[~mask2], deps2[~mask2], marker='o', linestyle='-', color='green', label='M')
		plt.plot(ages3[~mask3], deps3[~mask3], marker='o', linestyle='-', color='blue', label='S')
		plt.legend(fontsize=12)
		plt.xlim(xmin=0.5)
		plt.ylim(np.floor(dep_min/100)*100,np.ceil(dep_max/100)*100)
		plt.xlabel('Age (Ma)', fontsize=14)
		plt.ylabel('Water depth (m)', fontsize=14)
		fig4.suptitle(str(period)+" sec", fontsize=14)
		fig4.show()
		
		fig5 = plt.figure(5)
		sed_min = np.append(np.append(seds1,seds2),seds3).min()
		sed_max = np.append(np.append(seds1,seds2),seds3).max()
		plt.plot(ages1[~mask1], seds1[~mask1], marker='o', linestyle='-', color='red', label='N')
		plt.plot(ages2[~mask2], seds2[~mask2], marker='o', linestyle='-', color='green', label='M')
		plt.plot(ages3[~mask3], seds3[~mask3], marker='o', linestyle='-', color='blue', label='S')
		plt.legend(fontsize=12)
		plt.xlim(xmin=0.5)
		plt.ylim(np.floor(sed_min/100)*100,np.ceil(sed_max/100)*100)
		plt.xlabel('Age (Ma)', fontsize=14)
		plt.ylabel('Sediment thickness (m)', fontsize=14)
		fig5.suptitle(str(period)+" sec", fontsize=14)
		fig5.show()
		pass
	
	
	def plot_age_deriv(self, period=6., vec_s=0.05, sparse=10, lon=232., lat=46., streamline=False, projection='lambert',geopolygons=None, showfig=True, vmin=None, vmax=None, sta=True, hillshade=False):
		"""Plot age derivative as vectors on the map
			vec_s   --  the size of the vectors drawn on the map (length of plot width)
			sparse  --  how sparse for drawing the vectors
			lon     --  longitude of starting point on stream line, only works when streamline == True
			lat     --  latitude of starting point on stream line, only works when streamline == True
		"""
		if hillshade:
			alpha = 0.5
		else:
			alpha = 1.
		group = self['%g_sec'%( period )]
		deriv_lat_Arr = group['age_deriv_lat_Arr'].value
		deriv_lon_Arr = group['age_deriv_lon_Arr'].value
		deriv_msk_Arr = group['age_deriv_msk_Arr'].value
		age_Arr = group['age_Arr'].value
		age_Arr_msk = group['age_Arr_msk'].value
		mask = np.logical_or(deriv_msk_Arr, age_Arr_msk)
		m = self._get_basemap(projection=projection, geopolygons=geopolygons,hillshade=hillshade)
		x, y = m(group['lonArr'].value, group['latArr'].value)
		my_cmap = pycpt.load.gmtColormap('./cv.cpt')
		if vmin == None:
			vmin = np.nanmin(age_Arr[~age_Arr_msk])
			vmin = np.ceil(vmin)
		if vmax == None:
			vmax = np.nanmax(age_Arr[~age_Arr_msk])
			vmax = np.floor(vmax)
		im = m.pcolormesh(x, y, np.ma.masked_array(age_Arr,mask=age_Arr_msk), cmap=my_cmap, shading='gouraud', vmin=vmin, vmax=vmax, alpha=alpha)
		cb = m.colorbar(im, "bottom", size="3%", pad='2%', format='%d')
		cb.set_label("Age (myr)", fontsize=12, rotation=0)
		cb.set_alpha(1)
		cb.draw_all()
		ax = plt.gca()
		if streamline:
			rEarth = 6370997.0 # Earth radius
			seed = np.array(m(lon, lat))
			dx = deriv_lon_Arr * rEarth * np.cos(group['latArr'].value * np.pi / 180)
			dy = deriv_lat_Arr * rEarth
			strm = ax.streamplot(x[0,:], y[:,0], dx, dy, color='gray', linewidth=1, start_points=seed.reshape(1,2))
			ax.plot(seed[0], seed[1], marker='o',color='gray')
		else:
			llons = group['lonArr'].value[~mask]
			llats = group['latArr'].value[~mask]
			dlon = deriv_lon_Arr[~mask]
			dlat = deriv_lat_Arr[~mask]
			dabs = np.sqrt(dlon[::sparse]**2 + dlat[::sparse]**2)
			m.quiver(llons[::sparse], llats[::sparse], dlon[::sparse]/dabs, dlat[::sparse]/dabs, latlon=True, pivot='tail',scale=1./vec_s)
		if sta:
			self.sta_on_plot(ax,m,period)
		fig = plt.gcf()
		fig.suptitle(str(period)+' sec', fontsize=14,y=0.95)
		if showfig:
			plt.show()
		else:
			plt.close('all')
		if streamline:
			return strm
		pass
	
	def plot_age_traj(self, period=6., lon=232., lat=46., N=30, streamline=False, vec=False, projection='lambert',geopolygons=None, showfig=True, vmin=None, vmax=None, sta=True, hillshade=False):
		"""Plot a trajectory of varying age based on a starting point
			lon, lat   --  longitude & latitude of the starting point
		"""
		if hillshade:
			alpha = 0.5
		else:
			alpha = 1.
		group = self['%g_sec'%( period )]
		age_Arr = group['age_Arr'].value
		age_Arr_msk = group['age_Arr_msk'].value
		m = self._get_basemap(projection=projection, geopolygons=geopolygons,hillshade=hillshade)
		if streamline:
			llons, llats = self._cons_traj_stream(period=period,lon=lon,lat=lat,N=N,projection=projection)
			m = self._get_basemap(projection=projection, geopolygons=geopolygons,hillshade=hillshade)
			x_traj, y_traj = m(llons,llats)
		else:
			mask = self._cons_traj(lon=lon,lat=lat,period=period)
			llons = group['lonArr'].value[~mask]
			llats = group['latArr'].value[~mask]
			if vec:
				deriv_lat_Arr = group['age_deriv_lat_Arr'].value
				deriv_lon_Arr = group['age_deriv_lon_Arr'].value
				dlon = deriv_lon_Arr[~mask]
				dlat = deriv_lat_Arr[~mask]
				dabs = np.sqrt(dlon**2 + dlat**2)
				m.quiver(llons, llats, dlon/dabs, dlat/dabs, latlon=True, pivot='tail')
			x_traj, y_traj = m(llons, llats)
			ind = np.argsort(x_traj)
			x_traj = x_traj[ind]
			y_traj = y_traj[ind]
		x, y = m(group['lonArr'].value, group['latArr'].value)
		my_cmap = pycpt.load.gmtColormap('./cv.cpt')
		if vmin == None:
			vmin = np.nanmin(age_Arr[~age_Arr_msk])
			vmin = np.ceil(vmin)
		if vmax == None:
			vmax = np.nanmax(age_Arr[~age_Arr_msk])
			vmax = np.floor(vmax)
		im = m.pcolormesh(x, y, np.ma.masked_array(age_Arr,mask=age_Arr_msk), cmap=my_cmap, shading='gouraud', vmin=vmin, vmax=vmax, alpha=alpha)
		cb = m.colorbar(im, "bottom", size="3%", pad='2%', format='%d')
		cb.set_label("Age (myr)", fontsize=12, rotation=0)
		cb.set_alpha(1)
		cb.draw_all()
		ax = plt.gca()
		ax.plot(x_traj, y_traj,color='gray')
		if sta:
			self.sta_on_plot(ax,m,period)
		fig = plt.gcf()
		fig.suptitle(str(period)+' sec', fontsize=14,y=0.95)
		if showfig:
			plt.show()
		return

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

def _walk(drv_lon,drv_lat,i,j,inc):
		""" Walk along the gradient direction of (i,j)
		Parameter:
					drv_lon  --  the array for partial derivtive with respect to longitude
					drv_lat  --  the array for partial derivtive with respect to latitude
					i,j      --  index for the point of interest
					inc      --  1 for inceasing direction, -1 for descending diretion 
		"""
		if drv_lon[i,j] == 0:
			if drv_lat[i,j] > 0:
				i += inc
			else:
				i += -inc
		elif drv_lon[i,j] > 0:
			angle = np.arctan(drv_lat[i,j]/drv_lon[i,j])
			if np.abs(angle) <= np.pi/8.:
				j += inc
			elif np.pi/8. < angle < 3.*np.pi/8.:
				j += inc
				i += inc
			elif -3.*np.pi/8. < angle < -np.pi/8.:
				j += inc
				i += -inc
			elif angle >= 3.*np.pi/8.:
				i += inc
			else:
				i += -inc
		else:
			angle = np.arctan(drv_lat[i,j]/drv_lon[i,j])
			if np.abs(angle) <= np.pi/8.:
				j += -inc
			elif np.pi/8. < angle < 3.*np.pi/8.:
				j += -inc
				i += -inc
			elif -3.*np.pi/8. < angle < -np.pi/8.:
				j += -inc
				i += inc
			elif angle >= 3.*np.pi/8.:
				i += -inc
			else:
				i += inc
		return i,j