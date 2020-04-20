#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:29:33 2020

@author: lxmera
"""

def DownloadAAL3(PATH):
    import os
    from nilearn import datasets
    
    if os.path.isfile(PATH+'/AAL3_for_SPM12.tar.gz'):
        print('Done')
    else:
        os.system('wget https://www.oxcns.org/AAL3_for_SPM12.tar.gz -P '+PATH)
        
    if os.path.exists(PATH+'/AAL3'):
        print('Done')
    else:
        os.system('tar -zxvf '+PATH+'/AAL3_for_SPM12.tar.gz -C '+PATH)
        
    if os.path.exists(PATH+'/AAL3/AAL3.mat'):
        print('Done')
    else:
        os.system('wget https://www.dropbox.com/s/eeullhxfv8tk6fg/AAL3.mat?dl=1 -P '+PATH+'/AAL3')
        os.system('mv '+PATH+'/AAL3/AAL3.mat?dl=1 '+PATH+'/AAL3/AAL3.mat')  
        
    ###################
    A_HOx='/home/lxmera/nilearn_data/fsl/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz'
    if os.path.isfile(A_HOx):
        print('Done')
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
        print('Done')
    else:
        os.system('wget https://www.dropbox.com/s/t0keqsapcbdl10b/labelsHof.mat?dl=1 -P '+A_HOx[:A_HOx.rfind('/')])
        os.system('mv '+A_HOx[:A_HOx.rfind('/')]+'/labelsHof.mat?dl=1 '+A_HOx[:A_HOx.rfind('/')]+'/labelsHof.mat')
    
    A_MSDL='/home/lxmera/nilearn_data/msdl_atlas/MSDL_rois/msdl_rois.nii'
    if os.path.isfile(A_MSDL):
        print('Done')
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
        print('Done')
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
    
    
    
if __name__=="__main__":
    
  
  from os.path import join as opj  
  from nipype import Workflow, Node, Function, MapNode
  from nipype.interfaces.io import DataSink
  import cv2
    
  experiment_dir = '/home/lxmera/neuro3/output'
  output_dir = 'datasink_Metrics'
  working_dir = 'workingdir'  
  ATLAS, mat, ATLAS2, mat2, ATLAS3, mat3=DownloadAAL3(opj(experiment_dir, working_dir))
  
  Subjec=Node(Function(input_names=[], output_names=['ANAT', 'FUNC'], function=sujetos), name='Sujetos')
    
  Text_In=Node(Function(input_names=['uno', 'dos', 'atlas'], output_names=[], function=texto), name='texto_recibido')
  Text_In.inputs.atlas=ATLAS
  
  series=Node(Function(input_names=['Maps', 'func', 'typeF'], output_names=['out_mat', 'n_plot'], function=series_times_ROI), name='series_time_AAL3')
  series.inputs.Maps=ATLAS
  series.inputs.typeF='Labels'
  
  series2=Node(Function(input_names=['Maps', 'func', 'typeF'], output_names=['out_mat', 'n_plot'], function=series_times_ROI), name='series_time_HarvardOxford')
  series2.inputs.Maps=ATLAS2
  series2.inputs.typeF='Labels'
  
  series3=Node(Function(input_names=['Maps', 'func', 'typeF'], output_names=['out_mat', 'n_plot'], function=series_times_ROI), name='series_time_MSDL')
  series3.inputs.Maps=ATLAS3
  series3.inputs.typeF='Maps'
  
  Integ=Node(Function(input_names=['t1', 't2', 't3'], output_names=['Time_files'], function=Integrate), name='Integrate_files')
  
  correlation=Node(Function(input_names=['Time_s', 'in_mat', 'typeF', 'kind'], output_names=['out_mat', 'n_plot', 'n_plot2'], function=Functional_Connectivity), name='Funcional_connectivity_ALL3')
  correlation.inputs.in_mat=mat
  correlation.inputs.typeF='Labels'
  correlation.inputs.kind='correlation'
  
  correlation2=Node(Function(input_names=['Time_s', 'in_mat', 'typeF', 'kind'], output_names=['out_mat', 'n_plot', 'n_plot2'], function=Functional_Connectivity), name='Funcional_connectivity_HOx')
  correlation2.inputs.in_mat=mat2
  correlation2.inputs.typeF='Labels'
  correlation2.inputs.kind='correlation'
  
  correlation3=Node(Function(input_names=['Time_s', 'in_mat', 'typeF', 'kind'], output_names=['out_mat', 'n_plot', 'n_plot2'], function=Functional_Connectivity), name='Funcional_connectivity_MSDL')
  correlation3.inputs.in_mat=mat3
  correlation3.inputs.typeF='Maps'
  correlation3.inputs.kind='correlation'
  
  Integ2=Node(Function(input_names=['t1', 't2', 't3'], output_names=['Corre_files'], function=Integrate), name='Correlation_files')
  
  Graph=MapNode(Function(input_names=['Mat_D', 'Threshold', 'percentageConnections', 'complet'], output_names=['out_data', 'out_mat'], function=get_graph), name='Graph_Metricts', iterfield=['Mat_D'])
  Graph.iterables = ("Threshold", [0.6])
  Graph.inputs.percentageConnections=False #Porcentaje de conexiones  utilizadas
  
  ALFF_fALFF=MapNode(Function(input_names=['slow', 'ASamplePeriod', 'Time_s', 'plots'], output_names=['out_mat'], function=Calculate_ALFF_fALFF), name='ALFF_and_fALFF', iterfield=['Time_s'])
  ALFF_fALFF.iterables = ("slow", [2, 3, 4, 5])
  ALFF_fALFF.inputs.ASamplePeriod=1.6   #Time repetition 
  
  ReHo=Node(Function(input_names=['func', 'nneigh', 'help_reho'], output_names=['out_ReHo'], function=Calculate_ReHo), name='Regional_homogeneity')
  ReHo.iterables = ("nneigh", [7, 19, 27])
 
  # Datasink - Crear una carpeta de salidad para almacenar las entradas
  datasink_metricas = Node(DataSink(base_directory=experiment_dir, container=output_dir), name="datasink_metricas")  
  substitutions = [('_subject_id_', 'sub-'),
                 ('_task_name_', '/task-'), 
                 ('_fwhm_', 'fwhm-'),       
                 ('_roi', ''),              
                 ('_mcf', ''),              
                 ('_st', ''),               
                 ('_flirt', ''),            
                 ('.nii_mean_reg', '_mean'),
                 ('.nii.par', '.par')]
  
  subjFolders = [('slow-%s/' % f, 'slow-%s_' % f) for f in [2, 3, 4, 5]]
  substitutions.extend(subjFolders)
  datasink_metricas.inputs.substitutions = substitutions
  
  ###################################################Crear el flujo de trabajo#
  metricas = Workflow(name='metricas')
  metricas.base_dir = opj(experiment_dir, working_dir)
  
  #Concatenar cada uno de lo nodos  
  metricas.connect([(Subjec, series, [('FUNC', 'func')]),
                    (Subjec, series2, [('FUNC', 'func')]),
                    (Subjec, series3, [('FUNC', 'func')]),                    
                    (series, Integ, [('out_mat', 't1')]),
                    (series2, Integ, [('out_mat', 't2')]),
                    (series3, Integ, [('out_mat', 't3')]),
                    (Integ, ALFF_fALFF, [('Time_files', 'Time_s')]),
                    
                    (Subjec, ReHo, [('FUNC', 'func')]),                    
                    
                    (series, correlation, [('out_mat', 'Time_s')]),
                    (series2, correlation2, [('out_mat', 'Time_s')]),
                    (series3, correlation3, [('out_mat', 'Time_s')]),
                    (correlation, Integ2, [('out_mat', 't1')]),
                    (correlation2, Integ2, [('out_mat', 't2')]),
                    (correlation3, Integ2, [('out_mat', 't3')]),
                    (Integ2, Graph, [('Corre_files', 'Mat_D')]),                    
                    
                    ###########################################################
                    (series, datasink_metricas, [('out_mat', 'metricas.@out_mats'), ('n_plot', 'metricas.@plot_atlas')]),
                    (series2, datasink_metricas, [('out_mat', 'metricas.@out_mats2'), ('n_plot', 'metricas.@plot_atlas2')]),
                    (series3, datasink_metricas, [('out_mat', 'metricas.@out_mats3')]),
                    
                    (correlation, datasink_metricas, [('out_mat', 'metricas.@out_mat_c'), ('n_plot', 'metricas.@plot_matrix')]),
                    (correlation2, datasink_metricas, [('out_mat', 'metricas.@out_mat_c2'), ('n_plot', 'metricas.@plot_matrix2')]),
                    (correlation3, datasink_metricas, [('out_mat', 'metricas.@out_mat_c3'), ('n_plot', 'metricas.@plot_matrix3'), ('n_plot2', 'metricas.@plot_connectome')]),
                    
                    (ReHo, datasink_metricas, [('out_ReHo', 'metricas.@out_ReHo')]),                                        
                    (ALFF_fALFF, datasink_metricas, [('out_mat', 'metricas.@out_mat')]),
                    (Graph, datasink_metricas, [('out_data', 'metricas.@out_data'), ('out_mat', 'metricas.@out_matGraph')]),
                    ])
  
  #Generar el gráfico y visualizarlo  
  metricas.write_graph(graph2use='flat')
  grafo=cv2.imread('/home/lxmera/neuro3/output/workingdir/metricas/graph_detailed.png')
  #plt.figure()
  #plt.imshow(grafo)
  
  #Ruuuuunn
  metricas.run('MultiProc', plugin_args={'n_procs': 4})
  
    
  