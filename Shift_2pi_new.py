import dispDBase
import numpy as np
import matplotlib.pyplot as plt

def get_fit(age_avgs,V):
	fits = np.polyfit(np.sqrt(age_avgs),V,2)
	p = np.poly1d(fits)
	t0 = np.linspace(0,np.max(age_avgs),100)
	predict_V = p(np.sqrt(t0))
	return t0,predict_V

dset = dispDBase.dispASDF('./age_model.h5')
colors = ['orange','blue','wheat','green','red']
age_phv_1 = dset.auxiliary_data.FitResult['18']['phase'].data.value
plt.plot(age_phv_1[:,0],age_phv_1[:,1],'.',color=colors[0],label=str(18)+' sec')
t0, pred_V = get_fit(age_phv_1[:,0],age_phv_1[:,1])
plt.plot(t0,pred_V,color=colors[0])
age_phv_2 = dset.auxiliary_data.FitResult['12']['phase'].data.value
plt.plot(age_phv_2[:,0],age_phv_2[:,1],'.',color=colors[1],label=str(12)+' sec')
t0, pred_V = get_fit(age_phv_2[:,0],age_phv_2[:,1])
plt.plot(t0,pred_V,color=colors[1])
done = np.append(age_phv_1[:,0],age_phv_2[:,0])
age_phv_3 = dset.auxiliary_data.FitResult['10']['phase'].data.value
str_per = str(int(10.)).zfill(2)
ap_per = dset.auxiliary_data.FitResult[str_per]['phase'].data.value
c0 = dset.auxiliary_data.FitResult['10']['phase'].parameters['c0']
c1 = dset.auxiliary_data.FitResult['10']['phase'].parameters['c1']
c2 = dset.auxiliary_data.FitResult['10']['phase'].parameters['c2']
diffs = age_phv_3[:,1] - (c0+c1*np.sqrt(age_phv_3[:,0])+c2*age_phv_3[:,0])
ind = diffs < -diffs.std()
sta_pairs = dset.auxiliary_data.FinalStas[str_per]['phase'].data.value
ages = np.array([])
phVs = np.array([])
ages_10_8 = np.array([])
phVs_10_8 = np.array([])
ages_10_6 = np.array([])
phVs_10_6 = np.array([])
for idx, sta_pair in enumerate(sta_pairs[ind]):
	if dset.auxiliary_data.AgeGc[sta_pair[0]][sta_pair[1]][sta_pair[2]][sta_pair[3]].parameters['age_avg'] in done:
		continue
	dist = dset.auxiliary_data.DISPArray[sta_pair[0]][sta_pair[1]][sta_pair[2]][sta_pair[3]].parameters['dist']
	DISP_interp = dset.auxiliary_data.DISPinterp[sta_pair[0]][sta_pair[1]][sta_pair[2]][sta_pair[3]].data.value
	phv_T = age_phv_3[ind,1][idx]
	v0 = c0+c1*np.sqrt(age_phv_3[ind,0][idx])+c2*age_phv_3[ind,0][idx]
	n = round((dist/phv_T-dist/v0)/10.)
	DISP_interp[2,:] = dist/(dist/DISP_interp[2,:]-n*DISP_interp[0,:])
	ind2 = np.where(DISP_interp[0,:] == 10)[0]
	if ind2.size == 1:
		ages = np.append(ages,dset.auxiliary_data.AgeGc[sta_pair[0]][sta_pair[1]][sta_pair[2]][sta_pair[3]].parameters['age_avg'])
		phVs = np.append(phVs,DISP_interp[2,ind2])
	ind2 = np.where(DISP_interp[0,:] == 8)[0]
	if ind2.size == 1:
		ages_10_8 = np.append(ages_10_8,dset.auxiliary_data.AgeGc[sta_pair[0]][sta_pair[1]][sta_pair[2]][sta_pair[3]].parameters['age_avg'])
		phVs_10_8 = np.append(phVs_10_8,DISP_interp[2,ind2])
	ind2 = np.where(DISP_interp[0,:] == 6)[0]
	if ind2.size == 1:
		ages_10_6 = np.append(ages_10_6,dset.auxiliary_data.AgeGc[sta_pair[0]][sta_pair[1]][sta_pair[2]][sta_pair[3]].parameters['age_avg'])
		phVs_10_6 = np.append(phVs_10_6,DISP_interp[2,ind2])
ind3 = [age_i in ages for age_i in ap_per[:,0]]
plt.plot(ap_per[np.logical_not(ind3),0], ap_per[np.logical_not(ind3),1], '.',color=colors[2],label=str(10)+' sec')
plt.plot(ages, phVs, '.',color=colors[2])
age_out = np.append(ap_per[np.logical_not(ind3),0],ages)
phV_out = np.append(ap_per[np.logical_not(ind3),1],phVs)
ind_out = np.logical_and(phV_out>0.25,phV_out<4.5)
t0, pred_V = get_fit(age_out[ind_out],phV_out[ind_out])
plt.plot(t0,pred_V,color=colors[2])
done = np.append(done,ap_per[:,0])

age_phv_4 = dset.auxiliary_data.FitResult['08']['phase'].data.value
str_per = str(int(8.)).zfill(2)
ap_per = dset.auxiliary_data.FitResult[str_per]['phase'].data.value
c0 = dset.auxiliary_data.FitResult[str_per]['phase'].parameters['c0']
c1 = dset.auxiliary_data.FitResult[str_per]['phase'].parameters['c1']
c2 = dset.auxiliary_data.FitResult[str_per]['phase'].parameters['c2']
diffs = age_phv_4[:,1] - (c0+c1*np.sqrt(age_phv_4[:,0])+c2*age_phv_4[:,0])
ind = diffs < -diffs.std()
sta_pairs = dset.auxiliary_data.FinalStas[str_per]['phase'].data.value
ages = np.array([])
phVs = np.array([])
ages_8_6 = np.array([])
phVs_8_6 = np.array([])
for idx, sta_pair in enumerate(sta_pairs[ind]):
	if dset.auxiliary_data.AgeGc[sta_pair[0]][sta_pair[1]][sta_pair[2]][sta_pair[3]].parameters['age_avg'] in done:
		continue
	dist = dset.auxiliary_data.DISPArray[sta_pair[0]][sta_pair[1]][sta_pair[2]][sta_pair[3]].parameters['dist']
	DISP_interp = dset.auxiliary_data.DISPinterp[sta_pair[0]][sta_pair[1]][sta_pair[2]][sta_pair[3]].data.value
	phv_T = age_phv_4[ind,1][idx]
	v0 = c0+c1*np.sqrt(age_phv_4[ind,0][idx])+c2*age_phv_4[ind,0][idx]
	n = round((dist/phv_T-dist/v0)/8.)
	DISP_interp[2,:] = dist/(dist/DISP_interp[2,:]-n*DISP_interp[0,:])
	ind2 = np.where(DISP_interp[0,:] == 8)[0]
	if ind2.size == 1:
		ages = np.append(ages,dset.auxiliary_data.AgeGc[sta_pair[0]][sta_pair[1]][sta_pair[2]][sta_pair[3]].parameters['age_avg'])
		phVs = np.append(phVs,DISP_interp[2,ind2])
	ind2 = np.where(DISP_interp[0,:] == 6)[0]
	if ind2.size == 1:
		ages_8_6 = np.append(ages_10_6,dset.auxiliary_data.AgeGc[sta_pair[0]][sta_pair[1]][sta_pair[2]][sta_pair[3]].parameters['age_avg'])
		phVs_8_6 = np.append(phVs_10_6,DISP_interp[2,ind2])
ind3 = [age_i in ages for age_i in ap_per[:,0]]
ind4 = [age_i in ages_10_8 for age_i in ap_per[:,0]]
plt.plot(ap_per[np.logical_not(np.logical_or(ind3,ind4)),0], ap_per[np.logical_not(np.logical_or(ind3,ind4)),1], '.',color=colors[3])
plt.plot(ages_10_8,phVs_10_8,'.',color=colors[3])
plt.plot(ages, phVs, '.',color=colors[3],label=str(8)+' sec')
age_out = np.append(ap_per[np.logical_not(np.logical_or(ind3,ind4)),0],ages_10_8)
age_out = np.append(age_out,ages)
phV_out = np.append(ap_per[np.logical_not(np.logical_or(ind3,ind4)),1],phVs_10_8)
phV_out = np.append(phV_out,phVs)
ind_out = np.logical_and(phV_out>0.25,phV_out<4.5)
t0, pred_V = get_fit(age_out[ind_out],phV_out[ind_out])
plt.plot(t0,pred_V,color=colors[3])
done = np.append(done,ap_per[:,0])

age_phv_5 = dset.auxiliary_data.FitResult['06']['phase'].data.value
str_per = str(int(6.)).zfill(2)
ap_per = dset.auxiliary_data.FitResult[str_per]['phase'].data.value
c0 = dset.auxiliary_data.FitResult[str_per]['phase'].parameters['c0']
c1 = dset.auxiliary_data.FitResult[str_per]['phase'].parameters['c1']
c2 = dset.auxiliary_data.FitResult[str_per]['phase'].parameters['c2']
diffs = age_phv_5[:,1] - (c0+c1*np.sqrt(age_phv_5[:,0])+c2*age_phv_5[:,0])
ind = diffs < -diffs.std()
sta_pairs = dset.auxiliary_data.FinalStas[str_per]['phase'].data.value
ages = np.array([])
phVs = np.array([])
for idx, sta_pair in enumerate(sta_pairs[ind]):
	if dset.auxiliary_data.AgeGc[sta_pair[0]][sta_pair[1]][sta_pair[2]][sta_pair[3]].parameters['age_avg'] in done:
		continue
	dist = dset.auxiliary_data.DISPArray[sta_pair[0]][sta_pair[1]][sta_pair[2]][sta_pair[3]].parameters['dist']
	DISP_interp = dset.auxiliary_data.DISPinterp[sta_pair[0]][sta_pair[1]][sta_pair[2]][sta_pair[3]].data.value
	phv_T = age_phv_5[ind,1][idx]
	v0 = c0+c1*np.sqrt(age_phv_5[ind,0][idx])+c2*age_phv_5[ind,0][idx]
	n = round((dist/phv_T-dist/v0)/6.)
	DISP_interp[2,:] = dist/(dist/DISP_interp[2,:]-n*DISP_interp[0,:])
	ind2 = np.where(DISP_interp[0,:] == 6)[0]
	if ind2.size == 1:
		ages = np.append(ages,dset.auxiliary_data.AgeGc[sta_pair[0]][sta_pair[1]][sta_pair[2]][sta_pair[3]].parameters['age_avg'])
		phVs = np.append(phVs,DISP_interp[2,ind2])
ind3 = [age_i in ages for age_i in ap_per[:,0]]
ind4 = [age_i in ages_10_6 for age_i in ap_per[:,0]]
ind5 = [age_i in ages_8_6 for age_i in ap_per[:,0]]
indd = np.logical_or(np.logical_or(ind3,ind4),ind5)
plt.plot(ap_per[np.logical_not(indd),0], ap_per[np.logical_not(indd),1], '.',color=colors[4])
plt.plot(ages_10_6,phVs_10_6,'.',color=colors[4])
plt.plot(ages_8_6,phVs_8_6,'.',color=colors[4])
plt.plot(ages, phVs, '.',color=colors[4],label=str(6)+' sec')
age_out = np.append(ap_per[np.logical_not(indd),0],ages_10_6)
age_out = np.append(age_out,ages_8_6)
age_out = np.append(age_out,ages)
phV_out = np.append(ap_per[np.logical_not(indd),1],phVs_10_6)
phV_out = np.append(phV_out,phVs_8_6)
phV_out = np.append(phV_out,phVs)
ind_out = np.logical_and(phV_out>0.25,phV_out<4.5)
t0, pred_V = get_fit(age_out[ind_out],phV_out[ind_out])
plt.plot(t0,pred_V,color=colors[4])

plt.legend(loc='best',fontsize=20)
plt.title('Phase velocity vs. oceanic age',fontsize=20)
plt.xlabel('age (ma)',fontsize=20)
plt.ylabel('km/s',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(xmin=0)
plt.ylim(ymin=0.1,ymax=4.2)
plt.show()