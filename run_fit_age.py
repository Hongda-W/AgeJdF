import dispDBase
import numpy as np
import matplotlib.pyplot as plt

dset = dispDBase.dispASDF('./age_model_180321.h5')
# lst = [(360.-131.108, 47.443),(360.-127.51, 49.382),(360-124.01, 49.282),(360.-124.1, 43.2),(360.-126.76, 43.77),(360.-130.91,44.57),(360.-133.57,45.57)]
lst = [(360.-126.729,49.292),(360.-125.021,48.105),(360.-124.299,44.963),(360.-124.7,43.203),(360.-126.545,43.586),(360.-131.406,44.8),(360.-130.97,47.),(360.-129.821,49.12)]
dset.set_poly(lst,228,43,237,49)
dset.read_age_mdl()
dset.read_stations(stafile='/work3/wang/JdF/station_7D.lst',source='CIEI',chans=['BHZ', 'BHE', 'BHN'])
dset.read_paths(disp_dir='/work3/wang/JdF/FTAN_4_All',res=3.5)
dset.intp_disp(pers = np.append( np.arange(7.)*2.+6., np.arange(4.)*3.+20.))
pers = np.append( np.arange(7.)*2.+6., np.arange(4.)*3.+20.)
dset.write_2_sta_in(outdir='/work3/wang/JdF/Input_4_Ray', pers=pers, cut=0.8)
# for per in pers:
#    if per < 20:
#        continue
#     dset.fit_Harmon(int(per),vel_type='phase')
#     dset.fit_Harmon(int(per),vel_type='group')
# dset.plot_stations(ppoly=False)
# dset.plot_age_topo(10,vel_type='phase')
# dset.plot_vel_topo(10,vel_type='phase')
# dset.plot_vel_age(10,vel_type='phase')
# dset.plot_paths(6,vel_type='phase')
# dset.plot_all_vel(pers=np.array([6,8,10,12,18]),vel_type='group')
#dset.get_vel_age(3.9,-0.15,0.05,3.75,3.9)
