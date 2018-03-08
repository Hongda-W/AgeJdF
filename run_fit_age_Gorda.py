import dispDBase
import numpy as np
import matplotlib.pyplot as plt

dset = dispDBase.dispASDF('./age_model_Gorda_new.h5')
lst = [(360.-130.155,44.05),(360.-128.48,43.703),(360.-126.224,43.046),(360.-124.9,42.745),(360.-123.65,42.611),(360.-124.33,40.52),(360.-127.56,40.57),(360.-130.96,40.68)]
dset.set_poly(lst,230,40,237,44)
dset.read_age_mdl()
dset.read_stations(stafile='/scratch/summit/howa1663/CC_JdF/XCORR/station_7D.lst',source='CIEI',chans=['BHZ', 'BHE', 'BHN'])
dset.read_paths(disp_dir='/scratch/summit/howa1663/CC_JdF/XCORR/FTAN_4_All',res=3.5)
dset.intp_disp(pers=np.append( np.arange(6.)*2.+6., np.arange(4.)*3.+18.))
pers = np.append( np.arange(6.)*2.+6., np.arange(4.)*3.+18.)
for per in pers:
    dset.fit_Harmon(int(per),vel_type='phase')
    dset.fit_Harmon(int(per),vel_type='group')
# dset.fit_Harmon(10,vel_type='group')
# dset.plot_vel_age(10,vel_type='group')
# dset.plot_paths(10,vel_type='group')
# dset.plot_all_vel(pers=np.array([6,8,10,14]),vel_type='group')
