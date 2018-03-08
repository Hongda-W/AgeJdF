"""
Write the station pair paths for the final usable FTAN results for a certain period.
"""
import dispDBase
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import optimize

def func(age_data,c0,c1,c2):
	return c0+c1*np.sqrt(age_data)+c2*age_data

def sq_misfit(z,*params):
	"""Calculate total squared misfit. [Ye et al, GGG, 2013, eq(3)]
	Parameters:  z   -- (c0,c1,c2), the coefficients we want to invert
	        params -- dist, array (Npairs,); d_dist, array(Nparis,); V, array(Npairs,);
	                  ages_lst, list of arrays, Npairs number of arrays with different sizes;
	"""
	c0, c1, c2 = z # period dependent coefficients
	dist, d_dist, V, ages_lst = params
	if not len(ages_lst)==dist.size & dist.size==d_dist.size & V.size==dist.size:
			raise AttributeError('The number of inter-station paths are incompatible for inverting c0, c1 & c2')
	t_path = np.zeros(dist.shape)
	for i, ages in enumerate(ages_lst):
		age_inc = (ages[0:-1] + ages[1:])/2
		d_d = np.repeat(d_dist[i], age_inc.size)
		v_ages = c0+c1*np.sqrt(age_inc)+c2*age_inc
		v_ages[v_ages==0] = d_dist[i]/9999. # if model gives a 0 velocity change it to a small number to avoid 0 division
		t_path[i] = (d_d/v_ages).sum()
	sq_misfit = ((dist/t_path - V) **2).sum()
	return sq_misfit

T = 10.
UC = 'phase' # flag for chosing group or phase velocity measurement
c0,c1,c2 = 3.9, -0.15, 0.05


if __name__ == '__main__':
	dset = dispDBase.dispASDF('./age_model.h5')
	lst = [(360.-131.108, 47.443),(360.-127.51, 49.382),(360-124.01, 49.282),(360.-124.1, 43.2),(360.-126.76, 43.77),(360.-130.91,44.57),(360.-133.57,45.57)]
	dset.set_poly(lst,228,43,237,49)
	dset.read_age_mdl()
	per = str(int(T)).zfill(2) # reformat the period to a 2-character string
	# outname = "path_list_"+per+"_"+UC
	# outfile = open(outname,'w')
	path_list = dset.auxiliary_data.FinalStas[per][UC].data.value # path list every row contains netcode and stacode for the two stations
	age_out = np.array([])
	vel_out = np.array([])
	dist_arr = np.array([])
	d_dist_arr = np.array([])
	vel_arr = np.array([])
	ages_lst = []
	for path in path_list:
		lat1 = dset.waveforms[path[0].decode("ASCII")+'.'+path[1].decode("ASCII")].coordinates['latitude']
		lon1 = dset.waveforms[path[0].decode("ASCII")+'.'+path[1].decode("ASCII")].coordinates['longitude']
		lat2 = dset.waveforms[path[2].decode("ASCII")+'.'+path[3].decode("ASCII")].coordinates['latitude']
		lon2 = dset.waveforms[path[2].decode("ASCII")+'.'+path[3].decode("ASCII")].coordinates['longitude']
		gc_path, dist = dispDBase.get_gc_path(lon1,lat1,lon2,lat2,3)
		gc_points = np.array(gc_path)
		ages = dset.get_ages(gc_points[:,::-1])
		ages_lst.append(ages)
		ages_inc = (ages[0:-1] + ages[1:])/2
		v_s = c0 + c2*ages_inc + c1*np.sqrt(ages_inc) # model input
		vel_avg = (ages_inc.size)/((1/v_s).sum()) # the theoretical velocity measurement for the path
		dist_arr = np.append(dist_arr, dist)
		d_dist_arr = np.append(d_dist_arr,dist/(ages_inc.size))
		vel_arr = np.append(vel_arr, vel_avg)
		# age_avg = ages.mean()
		# v_s = c0 + c2*ages + c1*np.sqrt(ages)
		# v_s[0] = 2*v_s[0]
		# v_s[-1] = 2*v_s[-1]
		# vel_avg = (N-1)/((1/v_s).sum())
		# age_out = np.append(age_out,age_avg)
		# vel_out = np.append(vel_out,vel_avg)
	params = (dist_arr, d_dist_arr, vel_arr, ages_lst)
	cranges = (slice(0.5,4,0.1), slice(-0.5,0.5,0.05), slice(-0.2,0.2,0.02))
	resbrute = optimize.brute(sq_misfit,cranges,args=params,full_output=True,finish=None)
	c0_out, c1_out, c2_out = resbrute[0]
	# popt, pcov = curve_fit(func,age_out,vel_out)
	# plt.plot(age_out,vel_out,'r.', label='Data')
	age_syn = np.linspace(0,10,100)
	plt.plot(age_syn,c0_out+c1_out*np.sqrt(age_syn)+c2_out*age_syn,'y-',label='Fit')
	plt.plot(age_syn,c0+c2*age_syn+c1*np.sqrt(age_syn),'b-',label='Input')
	plt.legend(fontsize=14)
	ax = plt.gca()
	# ax.set_xlim(0,np.ceil(age_out.max()))
	ax.set_xlim(0,10)
	ax.set_ylabel("velocity (km/s)",fontsize=14)
	ax.set_xlabel("Age (Ma)",fontsize=14)
	plt.title("Synthetic test",fontsize=14)
	plt.show()
	
	