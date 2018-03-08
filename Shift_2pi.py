import dispDBase
import numpy as np
import matplotlib.pyplot as plt

dset = dispDBase.dispASDF('./age_model.h5')
age_phv = dset.auxiliary_data.FitResult['08']['phase'].data.value
c0 = dset.auxiliary_data.FitResult['08']['phase'].parameters['c0']
c1 = dset.auxiliary_data.FitResult['08']['phase'].parameters['c1']
c2 = dset.auxiliary_data.FitResult['08']['phase'].parameters['c2']
T = 8.
period = 14.
str_per = str(int(period)).zfill(2)
ap_per = dset.auxiliary_data.FitResult[str_per]['phase'].data.value
diffs = age_phv[:,1] - (c0+c1*np.sqrt(age_phv[:,0])+c2*age_phv[:,0])
ind = diffs < -diffs.std()
sta_pairs = dset.auxiliary_data.FinalStas['08']['phase'].data.value
ages = np.array([])
phVs = np.array([])
for idx, sta_pair in enumerate(sta_pairs[ind]):
	dist = dset.auxiliary_data.DISPArray[sta_pair[0]][sta_pair[1]][sta_pair[2]][sta_pair[3]].parameters['dist']
	DISP_interp = dset.auxiliary_data.DISPinterp[sta_pair[0]][sta_pair[1]][sta_pair[2]][sta_pair[3]].data.value
	phv_T = age_phv[ind,1][idx]
	v0 = c0+c1*np.sqrt(age_phv[ind,0][idx])+c2*age_phv[ind,0][idx]
	n = round((dist/phv_T-dist/v0)/T)
	DISP_interp[2,:] = dist/(dist/DISP_interp[2,:]-n*DISP_interp[0,:])
	ind2 = np.where(DISP_interp[0,:] == period)[0]
	if ind2.size == 1:
		ages = np.append(ages,dset.auxiliary_data.AgeGc[sta_pair[0]][sta_pair[1]][sta_pair[2]][sta_pair[3]].parameters['age_avg'])
		phVs = np.append(phVs,DISP_interp[2,ind2])
ind3 = [age_i in ages for age_i in ap_per[:,0]]
plt.plot(ap_per[np.logical_not(ind3),0], ap_per[np.logical_not(ind3),1], 'r.')
plt.plot(ages, phVs, 'g.')
t0 = np.linspace(0,10,50)
# plt.plot(t0,c0+c1*np.sqrt(t0)+c2*t0, 'b-')
plt.xlim(xmin=0.)
plt.xlabel('age (ma)')
plt.ylabel('km/s')
plt.show()
