#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:54:11 2020

@author: lxmera
"""

### Pipeline functions

def downloadH5(DIR_F):
    import os
    if os.path.isfile(DIR_F+'/LAYER4-C_COM-Saggital-0.h5'):
        print('The model weights have already been downloaded')
    else:
        os.system('wget https://github.com/LxMera/Convolutional-Neural-Network-for-the-classification-of-independent-components-of-rs-fMRI/raw/master/LAYER4-C_COM-Saggital-0.h5 -P '+DIR_F)
   
    if os.path.isfile(DIR_F+'/automaticclassificationcnn.py'):
        print('The script for automatic classification has already been downloaded')
    else:
        os.system('wget https://www.dropbox.com/s/3d1x9z04pdjqf13/automaticclassificationcnn.py?dl=1 -P '+DIR_F)
        os.system('mv '+DIR_F+'/automaticclassificationcnn.py?dl=1 '+DIR_F+'/automaticclassificationcnn.py')    
    Path=DIR_F+'/LAYER4-C_COM-Saggital-0.h5'
    return Path

def downloadAROMA(DIR_F):
    import os
    if os.path.isfile(DIR_F+'/ICA-AROMA-master.zip'):
        print('The ICA-AROMA-master for denoising has already been downloaded')
    else:
        print('Descargando')
        os.system('wget https://www.dropbox.com/s/ivnba3q69frn0pm/ICA-AROMA-master.zip?dl=1 -P '+DIR_F)
        os.system('mv '+DIR_F+'/ICA-AROMA-master.zip?dl=1 '+DIR_F+'/ICA-AROMA-master.zip') 
    
    if os.path.exists(DIR_F+'/ICA-AROMA-master'):
        print('The ICA-AROMA-master has already been unzipped')
    else:
        os.system('unzip '+DIR_F+'/ICA-AROMA-master.zip -d '+DIR_F)
    path_aroma=DIR_F+'/ICA-AROMA-master/ICA_AROMA.py'
    return path_aroma
    
def smoothNi(PATH_GZ, fwhm):
    from nilearn import image
    import os

    F_smooth=image.smooth_img(PATH_GZ,fwhm=fwhm)
    OutFile='s'+PATH_GZ[PATH_GZ.rfind('/')+1:]
    F_smooth.to_filename(OutFile)
    
    out_file=os.path.abspath(OutFile)
    return out_file

def filtered(PATH_GZ, Time_R):
    from nilearn.input_data import NiftiMasker
    from nilearn.signal import butterworth
    import os
        
    masker = NiftiMasker()   
    signal = masker.fit_transform(PATH_GZ)    
    X_filtered = butterworth(signals=signal, sampling_rate=1./Time_R, high_pass=0.01, copy=True)
    fmri_filtered = masker.inverse_transform(X_filtered)
    OutFile='f'+PATH_GZ[PATH_GZ.rfind('/')+1:]
    fmri_filtered.to_filename(OutFile)
    
    out_file=os.path.abspath(OutFile)
    return out_file

def bandpass_filter(files, lowpass_freq, highpass_freq, fs):
    from nipype.utils.filemanip import split_filename, filename_to_list
    import os
    import nibabel as nb
    import numpy as np
    
    if highpass_freq>lowpass_freq:
        print('Fatal Error: highpass freq > lowpass freq ')
    
    out_files = []
    for filename in filename_to_list(files):
        path, name, ext = split_filename(filename)
        out_file = os.path.join(os.getcwd(), name + '_bp' + ext)
        img = nb.load(filename)
        timepoints = img.shape[-1]
        F = np.zeros((timepoints))
        lowidx = int(timepoints / 2) + 1
        if lowpass_freq > 0:
            lowidx = np.round(float(lowpass_freq) / fs * timepoints)
        highidx = 0
        if highpass_freq > 0:
            highidx = np.round(float(highpass_freq) / fs * timepoints)
        
        if int(lowidx)>int(timepoints/2):
            if timepoints%2==1:
                lowidx=int(timepoints/2)+1
            else:
                lowidx=int(timepoints/2)
                
        F[int(highidx):int(lowidx)] = 1
                 
        F = ((F + F[::-1]) > 0).astype(int)
        data = img.get_data()
        if np.all(F == 1):
            filtered_data = data
        else:
            filtered_data = np.real(np.fft.ifftn(np.fft.fftn(data) * F))
        img_out = nb.Nifti1Image(filtered_data, img.affine, img.header)
        img_out.to_filename(out_file)
        out_files.append(out_file)
  
    if np.shape(out_files)[0] > 1:
        print('Error: there are more than one file')
    return out_files[0]

def MCflirt2(in_file, dof=6):
    import os
    import glob
    print(in_file)
    file=in_file[in_file.rfind('/')+1:]
    if in_file[-2:]=='gz':
        mean_ing=file[:-7]+'_mean_reg'+file[-7:]
    else:
        mean_ing=file[:-4]+'_mean_reg'+file[-4:]
    
    mean_ing=os.path.abspath(mean_ing)
    os.system('mcflirt -in '+in_file+' -dof '+str(dof)+' -meanvol -plots')
    print('mcflirt -in '+in_file+' -dof '+str(dof)+' -meanvol -plots')
    
    dirr=in_file[:in_file.rfind('/')+1]
    mean_img=glob.glob(dirr+'*mcf_mean_reg.nii.gz')[0]
    out_file=glob.glob(dirr+'*mcf.nii.gz')[0]
    par_file=glob.glob(dirr+'*mcf.par')[0]
    return out_file, mean_img, par_file

def SelecICA(in_dir, datapy):
    import glob
    import numpy as np
    
    mec_mix=glob.glob(in_dir+'/*_mix')[0]
    noise=list(np.load(datapy))    
    return mec_mix, noise

def get_wm(files):
    return files[-1]

def autoCNN(SUB, DirMod, DirPy):
    import os
    path_ac=os.getcwd()
    print('work address on node ', path_ac)
    print('Starting process RCBP')
    print('Reduction by Consecutive Binary Patterns')
    print('...................................................................................................................................................................................')
    os.system('cp '+DirPy+' '+path_ac)
    os.system('python -c "import automaticclassificationcnn as auto; auto.classificationIC_by_CNN(\''+SUB+'\',\''+DirMod+'\')"')
    ANT=path_ac[:path_ac.rfind('/')] 
    
    val=os.path.isfile(ANT+'/ResultsClassification/auto_labels_noise.txt')
    if val:
        print("Done")
    else:
        print('Fatal error: auto_labels_noise.txt not found')
    
    tex=ANT+'/ResultsClassification/auto_labels_noise.txt'
    npy=ANT+'/ResultsClassification/data.npy'
    keyOut='Done'
    
    return tex, npy, keyOut

def mostrar(uno, dos, tres):
    print('LEONEL 1',uno)
    print('LEONEL 2',dos)
    print('LEONEL 3',tres)
    
def Ica_Aroma(path_aroma, in_file, mat_file, par_file, tr):
    import os
    path_ac=os.getcwd()
    print('------------------------------- RUNNING ICA-AROMA -----------------------------') 
    print('--------------- ICA-based Automatic Removal Of Motion Artifacts ---------------')    
    print('python '+path_aroma+' -tr '+str(tr)+' -den both -i '+in_file+' -affmat '+mat_file+' -mc '+par_file+' -o '+path_ac+'/ICA_AROMA  -overwrite')
    os.system('python '+path_aroma+' -tr '+str(tr)+' -den both -i '+in_file+' -affmat '+mat_file+' -mc '+par_file+' -o '+path_ac+'/ICA_AROMA  -overwrite')
    if os.path.exists(path_ac+'/ICA_AROMA'):
        print('The folder ICA_AROMA  was created ')
    else:
        print('The folder ICA_AROMA  was NOT created ')
     
    file_aggr=path_ac+'/ICA_AROMA/denoised_func_data_aggr.nii.gz'
    file_nonaggr=path_ac+'/ICA_AROMA/denoised_func_data_nonaggr.nii.gz'
    
    if os.path.isfile(file_aggr) and os.path.isfile(file_aggr):
        print('-------------------------- Successfully Finished-----------------------------------')
        return file_aggr, file_nonaggr
    else:
        print('Fatal ERROR: Denoising data was NOT created')
        
    

def global_S(in_file):
    import numpy as np
    import nibabel as nb
    from scipy import signal
    import os
    
    data4D=nb.load(in_file).get_fdata()
    print(np.shape(data4D))    
    Global_signal=signal.detrend(np.mean(data4D, axis=(0,1,2)))
    
    path_ac=os.getcwd()
    path_global=path_ac+"/Global_signal.txt"
    
    sigtx = open(path_global, "w")
    sigtx.write("Global_Signal" + os.linesep)
    siz=np.shape(Global_signal)[0]
    for i in range(siz-1):
      sigtx.write(str(Global_signal[i])+os.linesep)
    sigtx.write(str(Global_signal[siz-1]))
    sigtx.close()
    return path_global

def GLM2(in_file, regressor, ref_name):
    import os
    path_ac=os.getcwd()
    file=in_file[in_file.rfind('/'):]
    
    if in_file[-2:]=='gz':
        out_file=path_ac+file[:-7]+ref_name+'_GML-out_file.nii.gz'
        out_res=path_ac+file[:-7]+'_'+regressor[regressor.rfind('/')+1:][:-4]+ref_name+'_GML.nii.gz'
    else:
        out_file=path_ac+file[:-4]+ref_name+'_GML-out_file.nii.gz'
        out_res=path_ac+file[:-4]+'_'+regressor[regressor.rfind('/')+1:][:-4]+ref_name+'_GML.nii.gz'
            
    print('fsl_glm -i '+in_file+' -d '+regressor+' -o '+out_file+' --out_res='+out_res)
    os.system('fsl_glm -i '+in_file+' -d '+regressor+' -o '+out_file+' --out_res='+out_res)
    
    if os.path.isfile(out_file) and os.path.isfile(out_res):
        print('Denoising GLM - Successfully Finished')
    else:
        print('Fatal error: Outputs were NOT created')
    
    return out_file, out_res

def jump_detecter(in_par, degr=3):
    from scipy import signal, interpolate
    import scipy.io as sio
    import numpy as np
    import scipy
    import os
    
    signalx=np.loadtxt(in_par)
    binaT=np.zeros((np.shape(signalx)[0]))
    peaksT=[]
    for jx in range(6):
        sign=signalx[:,jx]    
        d_sign=np.diff(signal.detrend(sign), degr)
        bina=np.array(d_sign>3*np.std(d_sign))
        bina=scipy.ndimage.binary_dilation(bina, iterations=degr)
        tamno=np.shape(sign)[0]
        tamnb=np.shape(bina)[0]
        f = interpolate.interp1d(np.arange(tamnb), bina)
        bina2=f(np.linspace(0,tamnb-1,tamno))
        bina2=bina2>=0.5
        bina2=np.append(np.append(False,bina2),False)
        flac=np.convolve(bina2,[1,-1], 'same')[1:-1]
        up=np.where(flac==1)[0]
        down=np.where(flac==-1)[0]
        peaksUp=[]
        peaksDo=[]
        for ix in enumerate(up):
            peaksUp.append(up[ix[0]]+np.argmax(sign[up[ix[0]]:down[ix[0]]]))
            peaksDo.append(up[ix[0]]+np.argmin(sign[up[ix[0]]:down[ix[0]]]))
        binaT=binaT+bina2[1:-1]
        peaksT.append(list([peaksUp, peaksDo]))
    Resul=os.getcwd()#+'-Results'
    out_bin=Resul+'/peaks.npy'
    out_peaks=Resul+'/peaks_list.mat'
    np.save(out_bin, binaT)
    sio.savemat(out_peaks,{'peaks_list': peaksT})
    return out_bin, out_peaks

def scrubbing_vol(in_file, outliers, peaks, method='both', thres=3):
    import os
    import numpy as np
    import nibabel as nb
    from nilearn.image import new_img_like

    comple=nb.load(in_file)
    out=np.loadtxt(outliers)
    pea=np.load(peaks)

    if method=='both':
        Bolp=pea<thres
        for jx in out:
            Bolp[int(jx)]=False
        dele=Bolp
    if method=='outliers':
        Bolp=pea>=0
        for jx in out:
            Bolp[int(jx)]=False
        dele=Bolp
    if method=='peaks':
        dele=pea<thres
    
    print('The size was reduced by scrubbing to',np.sum(dele),'time points')
    matrix=comple.get_fdata()
    matrixN=matrix[:,:,:,dele]
    out_nii=new_img_like(comple, matrixN)

    Resul=os.getcwd()#+'-Results'
    name=in_file[in_file.rfind('/')+1:][:in_file[in_file.rfind('/')+1:].find('.')]

    out_file=Resul+'/'+name+'_'+method+'_scrub.nii.gz'
    out_nii.to_filename(out_file)
  
    return out_file

def DownloadAAL3(PATH):
    import os
    from nilearn import datasets
    
    if os.path.isfile(PATH+'/AAL3_for_SPM12.tar.gz'):
        print('The atlas AAL3 has already been downloaded')
    else:
        os.system('wget https://www.oxcns.org/AAL3_for_SPM12.tar.gz -P '+PATH)
        
    if os.path.exists(PATH+'/AAL3'):
        print('The atlas AAL3 has already been unzipped')
    else:
        os.system('tar -zxvf '+PATH+'/AAL3_for_SPM12.tar.gz -C '+PATH)
        
    if os.path.exists(PATH+'/AAL3/AAL3.mat'):
        print('The atlas labels AAL have already been downloaded')
    else:
        os.system('wget https://www.dropbox.com/s/eeullhxfv8tk6fg/AAL3.mat?dl=1 -P '+PATH+'/AAL3')
        os.system('mv '+PATH+'/AAL3/AAL3.mat?dl=1 '+PATH+'/AAL3/AAL3.mat')  
        
    ###################
    A_HOx='/home/lxmera/nilearn_data/fsl/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz'
    if os.path.isfile(A_HOx):
        print('The atlas Harvard-Oxford has already been downloaded')
    else:
        ###############################
        if os.path.exists('/home/lxmera'):
            datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        else:
            print('####################################')
            print('#                                  #')
            print('#       CAMBIA EL USUARIO          #')
            print('#                                  #')
            print('####################################')
            
    mat2=A_HOx[:A_HOx.rfind('/')]+'/labelsHof.mat'  
    if os.path.isfile(mat2):
        print('The atlas labels Harvard-Oxford have already been downloaded')
    else:
        os.system('wget https://www.dropbox.com/s/t0keqsapcbdl10b/labelsHof.mat?dl=1 -P '+A_HOx[:A_HOx.rfind('/')])
        os.system('mv '+A_HOx[:A_HOx.rfind('/')]+'/labelsHof.mat?dl=1 '+A_HOx[:A_HOx.rfind('/')]+'/labelsHof.mat')
    
    A_MSDL='/home/lxmera/nilearn_data/msdl_atlas/MSDL_rois/msdl_rois.nii'
    if os.path.isfile(A_MSDL):
        print('The atlas MSDL has already been downloaded')
    else:
        ###############################
        if os.path.exists('/home/lxmera'):
            datasets.fetch_atlas_msdl()
        else:
            print('####################################')
            print('#                                  #')
            print('#       CAMBIA EL USUARIO          #')
            print('#                                  #')
            print('####################################')
            
    mat3=A_MSDL[:A_MSDL.rfind('/')]+'/labelsMSDL.mat'  
    if os.path.isfile(mat3):
        print('The atlas labels MSDL have already been downloaded')
    else:
        os.system('wget https://www.dropbox.com/s/j18tleliudcx2yn/labelsMSDL.mat?dl=1 -P '+A_MSDL[:A_MSDL.rfind('/')])
        os.system('mv '+A_MSDL[:A_MSDL.rfind('/')]+'/labelsMSDL.mat?dl=1 '+A_MSDL[:A_MSDL.rfind('/')]+'/labelsMSDL.mat')
           
        
    atlas=PATH+'/AAL3/AAL3.nii.gz'
    mat=PATH+'/AAL3/AAL3.mat'   
    return atlas, mat, A_HOx, mat2, A_MSDL, mat3

def sujetos():
    import nibabel as nb
    ANAT='/home/lxmera/neuro3/data/ds000133/sub-01/ses-pre/anat/sub-01_ses-pre_T1w.nii.gz'
    FUNC='/home/lxmera/neuro3/data/ds000133/sub-01/ses-pre/func/sub-01_ses-pre_task-rest_run-01_bold.nii.gz'
    print(nb.load(ANAT).shape)
    print(nb.load(FUNC).shape)
    return ANAT, FUNC

def texto(uno,  dos, atlas):
    from nilearn import plotting
    import os
    print('Anatomica ', uno)
    print('Funcional ', dos)
    
    ##################################
    Resul=os.getcwd()+'-Results'
    os.system('mkdir '+Resul) 
    ################################## 
    
    plot_atlas=plotting.plot_roi(atlas)
    plot_atlas.savefig(Resul+'/AtlasAAL3.svg')
    
def series_times_ROI(Maps, func, typeF):
    from nilearn.input_data import NiftiLabelsMasker, NiftiMapsMasker
    from nilearn import plotting
    import scipy.io as sio
    import numpy as np
    import os    
    ##################################
    Resul=os.getcwd()#+'-Results'
    n_map=Maps[Maps.rfind('/')+1:][:Maps[Maps.rfind('/')+1:].find('.')]
    n_plot='empty_plot'
    #os.system('mkdir '+Resul) 
    ##################################    
    if typeF=='Labels':
        masker = NiftiLabelsMasker(labels_img=Maps, standardize=True)
        plot_atlas=plotting.plot_roi(Maps)
        n_plot=Resul+'/Atlas_'+n_map+'_'+typeF+'.svg'
        plot_atlas.savefig(n_plot)        
    if typeF=='Maps':
        masker = NiftiMapsMasker(maps_img=Maps, standardize=True, memory='nilearn_cache', verbose=5)
        
    time_series = masker.fit_transform(func)
    print('Shape of serial times ', np.shape(time_series))     
    out_mat=Resul+'/Time_series_'+n_map+'_'+typeF+'.mat'
    sio.savemat(out_mat, {'time_series': time_series})
       
    return out_mat, n_plot

def Functional_Connectivity(Time_s, in_mat, typeF, kind):
    from nilearn.connectome import ConnectivityMeasure
    from nilearn import plotting
    import scipy.io as sio
    import numpy as np
    import os
    
    ##################################
    Resul=os.getcwd()#+'-Results'
    n_time=Time_s[Time_s.rfind('/')+1:][:Time_s[Time_s.rfind('/')+1:].find('.')]
    n_plot2='empty_plot'
    #os.system('mkdir '+Resul) 
    ##################################
    
    time_series=sio.loadmat(Time_s)['time_series']
    data=sio.loadmat(in_mat)
    labels=data['labels']
    
    correlation_measure = ConnectivityMeasure(kind=kind)
    correlation_matrix = correlation_measure.fit_transform([time_series])[0] 
    np.fill_diagonal(correlation_matrix, 0)
    
    if typeF=='Labels':
        vec_size=data['size'][0]
        indx=np.argsort(vec_size)[-np.shape(time_series)[1]:]
        indx=np.sort(indx)
        labels=labels[indx]
    if typeF=='Maps':
        coord=data['region_coords']
        plot_conne=plotting.plot_connectome(correlation_matrix, coord, edge_threshold="80%", colorbar=True)
        n_plot2=Resul+'/ConnectomePlotMDLS.svg'
        plot_conne.savefig(n_plot2)
    
    size_f=int((np.shape(time_series)[0]**(1/7))*30/2.064782369420003)
    ima=plotting.plot_matrix(correlation_matrix, figure=(size_f, size_f), labels=labels, colorbar=True, vmax=0.8, vmin=-0.8)
    n_plot=Resul+'/Correlation_matrix_'+kind+'_'+n_time+'.svg'       
    out_mat=Resul+'/Correlation_matrix_'+kind+'_'+n_time+'.mat'
    ima.figure.savefig(n_plot) 
    sio.savemat(out_mat, {'Correlation': correlation_matrix, 'labels': labels}) 
    return out_mat, n_plot, n_plot2

def Calculate_ALFF_fALFF(slow, ASamplePeriod, Time_s, plots=False):
    import os
    import math
    import scipy
    import numpy as np
    import scipy.io as sio
    import matplotlib.pyplot as plt
    
    slow=(slow-2)    
    AllVolume=sio.loadmat(Time_s)['time_series']   
    row, col=np.shape(AllVolume)
    names=['slow_2', 'slow_3', 'slow_4', 'slow_5']
    SlowHigh=[0.25, 0.198, 0.073, 0.027]
    SlowLow=[0.198, 0.073, 0.027, 0.01]

    HighCutoff=SlowHigh[slow] #the High edge of the pass band
    LowCutoff=SlowLow[slow]   #the low edge of the pass band
    
    sampleFreq 	 = 1/ASamplePeriod
    sampleLength = row
    
    p=1
    while True:
        if 2**p >= sampleLength:
            break
        else:
            p=p+1
    #paddedLength = 2**(nextpow2(sampleLength))
    paddedLength = 2**(p)
    
    if (LowCutoff >= sampleFreq/2): # All high included
        idx_LowCutoff = paddedLength/2 + 1;
    else: # high cut off, such as freq > 0.01 Hz
        idx_LowCutoff = math.ceil(LowCutoff * paddedLength * ASamplePeriod + 1);
    # Change from round to ceil: idx_LowCutoff = round(LowCutoff *paddedLength *ASamplePeriod + 1);

    if (HighCutoff>=sampleFreq/2)and(HighCutoff==0):# All low pass
        idx_HighCutoff = paddedLength/2 + 1;
    else: #Low pass, such as freq < 0.08 Hz
        idx_HighCutoff =  np.fix (HighCutoff *paddedLength *ASamplePeriod + 1);
    # Change from round to fix: idx_HighCutoff	=round(HighCutoff *paddedLength *ASamplePeriod + 1);
    #Zero Padding
    a = np.zeros((paddedLength - sampleLength,len(AllVolume[2])))
    AllVolume = np.concatenate((AllVolume, a), axis=0)
    
    
    print('\t Performing FFT ...');
    
    AllVolume=np.transpose(AllVolume)
    AllVolume = 2*np.true_divide(abs(scipy.fft(AllVolume)),sampleLength);
    AllVolume=np.transpose(AllVolume)
    
    print('Calculating ALFF for slow', slow+2,' ...')    
    ALFF_2D = np.mean(AllVolume[idx_LowCutoff:int(idx_HighCutoff)], axis=0)
    
    print('Calculating fALFF for slow', slow+2,' ...')
    num = np.sum(AllVolume[(idx_LowCutoff):int(idx_HighCutoff)],axis=0,dtype=float)
    den = np.sum(AllVolume[2:int(paddedLength/2 + 1)],axis=0,dtype=float)
    fALFF_2D =  num/den
       
    metricas = np.concatenate((ALFF_2D, fALFF_2D), axis=0).reshape((2,col))
    
    if plots:
        plt.figure()
        plt.title('Power Spectral Density')
        freq=np.arange(0.0, 1/ASamplePeriod, 1/(ASamplePeriod*np.shape(AllVolume)[0]))
        plt.plot(freq,AllVolume)
        
        plt.figure()
        plt.title('ALFF')
        plt.plot(metricas[0,:])
        
        plt.figure()
        plt.title('fALFF')
        plt.plot(metricas[1,:]) 
    print('...done')
    
    ##################################
    Resul=os.getcwd()#+'-Results'
    #os.system('mkdir '+Resul) 
    ##################################
    out_mat=Resul+'/ALFF_and_fALFF_'+names[slow]+'.mat'
    sio.savemat(out_mat, {'ALFF': metricas[0], 'fALFF': metricas[1]}) 
    return out_mat

def Integrate(t1, t2, t3):
    Time_files=[]
    Time_files.append(t1)
    Time_files.append(t2)
    Time_files.append(t3)
    return Time_files

def Calculate_ReHo(func, nneigh, help_reho=False):
    if help_reho:
        from nipype.interfaces import afni
        afni.ReHo.help()
    import os
    Resul=os.getcwd()#+'-Results'
    n_func=func[func.rfind('/')+1:][:func[func.rfind('/')+1:].find('.')]
    out_ReHo=Resul+'/'+n_func+'_ReHo_'+str(nneigh)+'.nii.gz'
    print('3dReHo -prefix '+out_ReHo+' -inset '+func+' -nneigh '+str(nneigh))
    os.system('3dReHo -prefix '+out_ReHo+' -inset '+func+' -nneigh '+str(nneigh))
    if os.path.isfile(out_ReHo):
        print('....ReHo done')
    else:
        print('Fatal error: The ReHo file was NOT created')    
    return out_ReHo

def get_graph(Mat_D, Threshold, percentageConnections=False, complet=False):
    import scipy.io as sio
    import numpy as np
    import networkx as nx
    import pandas as pd
    import os
    Data=sio.loadmat(Mat_D)
    matX=Data['Correlation']#[:tamn,:tamn]
    labels=Data['labels']
    print(np.shape(matX))
    print(np.shape(labels))
    print(np.min(matX), np.max(matX))
    
    if percentageConnections:
        if percentageConnections>0 and percentageConnections<1:
            for i in range(-100,100):
                per=np.sum(matX>i/100.)/np.size(matX)
                if per<=Threshold:
                    Threshold=i/100.
                    break
            print(Threshold)                
        else:
            print('The coefficient is outside rank')            
    
    #Lista de conexion del grafo
    row, col=np.shape(matX)
    e=[]
    for i in range(1,row):
      for j in range(i):
          if complet:
              e.append((labels[i],labels[j],matX[i,j]))
          else:
              if matX[i,j]>Threshold:
                  e.append((labels[i],labels[j],matX[i,j])) 
                  
    print(np.shape(e)[0], int(((row-1)*row)/2))
    
    #Generar grafo
    G=nx.Graph()
    G.add_weighted_edges_from(e)
    labelNew=list(G.nodes)
    
    #Metricas por grafo (ponderados)
    Dpc=nx.degree_pearson_correlation_coefficient(G, weight='weight')
    cluster=nx.average_clustering(G, weight='weight')
    
    #No ponderados
    estra=nx.estrada_index(G)
    tnsity=nx.transitivity(G)
    conNo=nx.average_node_connectivity(G)
    ac=nx.degree_assortativity_coefficient(G)
       
    
    #Metricas por nodo
    tam=15
    BoolCenV=False
    BoolLoad=False
    alpha=0.1
    beta=1.0
    
    katxCN=nx.katz_centrality_numpy(G, alpha=alpha, beta=beta, weight='weight')    
    bcen=nx.betweenness_centrality(G, weight='weight')
    av_nd=nx.average_neighbor_degree(G, weight='weight')
    ctr=nx.clustering(G, weight='weight')
    ranPaN=nx.pagerank_numpy(G, weight='weight')    
    Gol_N=nx.hits_numpy(G)    
    Dgc=nx.degree_centrality(G)
    cl_ce=nx.closeness_centrality(G)
    cluster_Sq=nx.square_clustering(G)
    centr=nx.core_number(G)
    cami=nx.node_clique_number(G)
    camiN=nx.number_of_cliques(G)
    trian=nx.triangles(G)
    colorG=nx.greedy_color(G)
    try:
        cenVNum=nx.eigenvector_centrality_numpy(G,weight='weight')
        tam=tam+1
        BoolCenV=True
    except TypeError:
        print("La red es muy pequeña y no se puede calcular este parametro gil")
    except:
        print ('NetworkXPointlessConcept: graph null')
    if Threshold>0:
        carga_cen=nx.load_centrality(G, weight='weight') #Pesos  positivos
        BoolLoad=True
        tam=tam+1
    #katxC=nx.katz_centrality(G, alpha=alpha, beta=beta, weight='weight')
    #cenV=nx.eigenvector_centrality(G,weight='weight')
    #cenV=nx.eigenvector_centrality(G,weight='weight')
    #Golp=nx.hits(G)
    #Gol_si=nx.hits_scipy(G) 
    #ranPa=nx.pagerank(G, weight='weight')
    #ranPaS=nx.pagerank_scipy(G, weight='weight')
    
    
    matrix_datos=np.zeros((tam,np.shape(labelNew)[0]))
    tam=15
    print(np.shape(matrix_datos))
    lim=np.shape(labelNew)[0]
    for i in range(lim):
      roi=labelNew[i]
      #print(roi)      
      matrix_datos[0,i]=katxCN[roi]
      matrix_datos[1,i]=bcen[roi]
      matrix_datos[2,i]=av_nd[roi]
      matrix_datos[3,i]=ctr[roi]
      matrix_datos[4,i]=ranPaN[roi]
      matrix_datos[5,i]=Gol_N[0][roi]
      matrix_datos[6,i]=Gol_N[1][roi]
      matrix_datos[7,i]=Dgc[roi]
      matrix_datos[8,i]=cl_ce[roi]
      matrix_datos[9,i]=cluster_Sq[roi]
      matrix_datos[10,i]=centr[roi]
      matrix_datos[11,i]=cami[roi]
      matrix_datos[12,i]=camiN[roi]
      matrix_datos[13,i]=trian[roi]
      matrix_datos[14,i]=colorG[roi]
      if BoolCenV:
          matrix_datos[15,i]=cenVNum[roi]
          tam=tam+1
      if BoolLoad:
          matrix_datos[16,i]=carga_cen[roi]
          tam=tam+1                
      #matrix_datos[0,i]=katxC[roi]
      #matrix_datos[2,i]=cenV[roi]
      #matrix_datos[7,i]=Golp[0][roi]
      #matrix_datos[9,i]=Gol_si[0][roi]
      #matrix_datos[10,i]=Golp[1][roi]
      #matrix_datos[12,i]=Gol_si[1][roi]
      #matrix_datos[22,i]=ranPa[roi]
      #matrix_datos[24,i]=ranPaS[roi]
    FuncName=['degree_pearson_correlation_coefficient', 'average_clustering', 'estrada_index', 'transitivity', 'average_node_connectivity', 'degree_assortativity_coefficient', 'katz_centrality_numpy', 'betweenness_centrality', 'average_neighbor_degree', 'clustering', 'pagerank_numpy', 'hits_numpy0', 'hits_numpy1','degree_centrality', 'closeness_centrality', 'square_clustering', 'core_number', 'node_clique_number', 'number_of_cliques', 'triangles', 'greedy_color','eigenvector_centrality_numpy', 'load_centrality']
    frame=pd.DataFrame(matrix_datos)
    frame.columns=labelNew
    frame.index=FuncName[6:tam]
    
    Resul=os.getcwd()    
    out_data=Resul+'/graph_metrics.csv'
    out_mat=Resul+'/graph_metrics_global.mat'
    
    frame.to_csv(out_data)    
    sio.savemat(out_mat, {FuncName[0]: Dpc, FuncName[1]: cluster, FuncName[2]: estra, FuncName[3]: tnsity, FuncName[4]: conNo, FuncName[5]: ac})
    return out_data, out_mat

def template_MNI(temp=1):
    if temp==1:
        out_file='/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'
    if temp==2:
        out_file='/usr/local/fsl/data/standard/MNI152_T1_1mm.nii.gz'
    if temp==3:
        out_file='/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'    
    return out_file
    
    
    
    
    