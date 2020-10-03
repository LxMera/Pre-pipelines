#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:49:42 2020

@author: lxmera
"""
#------------------------------------1 Librerias---------------------------------------------------------------

from nipype.interfaces.fsl import (BET, ExtractROI, FAST, FLIRT, SliceTimer, Threshold, MELODIC, FilterRegressor)
from nipype.algorithms.confounds import TCompCor
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.algorithms.rapidart import ArtifactDetect
from nipype import Workflow, Node, Function, MapNode
from os.path import join as opj
import os
import Pipeline_Functions as pif
import glob

#------------------------------------2 Dirección de trabajo---------------------------------------------------------------

Subjects_dir='/media/lxmera/Disco/Tesis/data/test3' #Direccion de los sujetos
experiment_dir = '/media/lxmera/Disco/Tesis/output' #Salidad del flujo
output_dir = 'datasink_prepro'                      #Salida resultados
output_dir2= 'datasink_metrics'                     #Salida de las métricas
working_dir = 'workingdir'                          #Direccion de trabajo

#Documentos adicionales  
CNN_H5=pif.downloadH5(opj(experiment_dir, working_dir))
Aropy=pif.downloadAROMA(opj(experiment_dir, working_dir))

#------------------------------------3 Estructura de los datos---------------------------------------------------------------

anat_file = opj('sub-{asubject_id}', 'ses-{session_num}', 'anat','sub-{asubject_id}_ses-{session_num}_T1w.nii.gz')
func_file = opj('sub-{asubject_id}', 'ses-{session_num}', 'func','sub-{asubject_id}_ses-{session_num}_task-rest_bold.nii.gz')
templates = {'anat': anat_file, 'func': func_file}

#------------------------------------4 Iteradores----------------------------------------------------------------------------

session=['1', '2', '3']           #Sesiones
sub=glob.glob(Subjects_dir+'/*')  #Sujetos (Numeros)
subject_list0=[]
for su in sub:
    subject_list0.append(su[-5:])    

#------------------------------------5 parametros de los procesos---------------------------------------------------------------------

componets_compcor=6         # Método de palo roto (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2214855/)
fwhm = [4]#, 8]             # Ancho del suavizado (mm) (3.4 china)
DOF=[12]#[3, 6, 7, 9, 12]   # Grados de libertad (HMC)
TR = 2.0                    # Tiempo de repeticion (s)
iso_size = 4                # Escalado isometrico de la ima fun - tamanho voxel (in mm)
dofx=12                     # Grados de libertad para el coregistro

#------------------------------------6 flujo de trabajo----------------------------------------------------------------------------

for kx in range(25):
    subject_list=[subject_list0[kx]]      
    
    #------------------------------------6.1 Nodos de seleccion de sujetos-------------------
    
    infosource = Node(IdentityInterface(fields=['asubject_id', 'session_num']), name="infosource")
    infosource.iterables = [('asubject_id', subject_list), ('session_num', session)]        
    selectfiles = Node(SelectFiles(templates, base_directory=Subjects_dir), name="selectfiles")
    
    #------------------------------------6.2 Nodos de Corregistro--------------------------------
    
    bet_anat = Node(BET(frac=0.5, robust=True, mask=True, output_type='NIFTI_GZ'),
                    name="bet_anat")
    
    segmentation = Node(FAST(output_type='NIFTI_GZ'),
                        name="segmentation")
    
    threshold = Node(Threshold(thresh=0.5, args='-bin', output_type='NIFTI_GZ'),
                     name="threshold")
    
    coreg_pre = Node(FLIRT(dof=dofx, output_type='NIFTI_GZ'),
                     name="coreg_pre")
    
    coreg_bbr = Node(FLIRT(dof=dofx, cost='bbr', schedule=opj(os.getenv('FSLDIR'),'etc/flirtsch/bbr.sch'), output_type='NIFTI_GZ'),
                     name="coreg_bbr")
                                         
    MNI=Node(Function(input_names=[], output_names=['out_file'], function=pif.template_MNI),
             name='Tamplated_MNI')
                           
    Normalization = Node(FLIRT(dof=dofx, output_type='NIFTI_GZ'),
                         name="Normalization")
    
    applywarp = Node(FLIRT(interp='spline', apply_isoxfm=iso_size, output_type='NIFTI'),
                     name="applywarp")
    
    applywarp_mean = Node(FLIRT(interp='spline', apply_isoxfm=iso_size, output_type='NIFTI_GZ'),
                          name="applywarp_mean")
                           
    
    #------------------------------------6.3 Flujo de corregistro----------------------
    
    coregwf = Workflow(name='coregwf')
    coregwf.base_dir = opj(experiment_dir, working_dir)
    
    coregwf.connect([(bet_anat, segmentation, [('out_file', 'in_files')]),
                    (segmentation, threshold, [(('partial_volume_files', pif.get_wm), 'in_file')]),
                     (bet_anat, coreg_pre, [('out_file', 'reference')]),                 
                     (MNI, Normalization, [('out_file', 'reference')]),                 
                     (threshold, coreg_bbr, [('out_file', 'wm_seg')]),
                     (coreg_pre, coreg_bbr, [('out_matrix_file', 'in_matrix_file')]),
                     (coreg_bbr, Normalization, [('out_matrix_file', 'in_matrix_file')]),
                     (Normalization, applywarp, [('out_matrix_file', 'in_matrix_file')]),
                     (MNI, applywarp, [('out_file', 'reference')]),
                     ])
    #coregwf.write_graph(graph2use='colored', format='svg', simple_form=True)
    
    #------------------------------------6.4 Nodos preprocesamiento (Funcional) ---------------------------'''
    
    extract = Node(ExtractROI(t_min=4, t_size=-1, output_type='NIFTI_GZ'),
                   name="extract")
    
    mcflirt2 = Node(Function(input_names=['in_file','dof'], output_names=['out_file', 'mean_img', 'par_file'], function=pif.MCflirt2), 
                    name='mcflirt2')
    mcflirt2.inputs.dof=12
    #mcflirt2.iterables = ("dof", DOF)
    
    slicetimer = Node(SliceTimer(index_dir=False, interleaved=True, output_type='NIFTI', time_repetition=TR),
                      name="slicetimer")
    
    smooth=Node(Function(input_names=['PATH_GZ', 'fwhm'], output_names=['out_file'], function=pif.smoothNi), 
                name='smooth')
    smooth.inputs.fwhm=4
    #smooth.iterables = ("fwhm", fwhm)
    
    Filter2=Node(Function(input_names=['files', 'lowpass_freq', 'highpass_freq', 'fs'], output_names=['out_file'], function=pif.bandpass_filter), 
                 name='filters_pass_high')
    Filter2.inputs.lowpass_freq=1./TR
    Filter2.inputs.highpass_freq=0
    Filter2.inputs.fs=1./TR
    
    #------------------------------------6.5 Nodos de almacenamiento-------------------------

    #Sustituciones de los nombres
    substitutions = [('_asubject_id_', 'sub-'), #sub-01     Carpeta por sujeto
                     ('_session_num', '/ses'),  #task-rest  Carpeta de tareas
                     ('_fwhm_', 'fwhm-'),       #Variacion en el fwhm
                     ('_roi', ''),              #segunto argumento vacio
                     ('_mcf', ''),              #''
                     ('_st', ''),               #''
                     ('_flirt', ''),            #''
                     ('_bp',''),
                     ('ssub-','sub-'),
                     ('_index_0/',''),
                     ('_index_1/',''),
                     ('_index_2/',''),
                     ('_Global_signal',''),
                     ('_components_file',''),
                     ('denoised_func_data','aroma'),
                     ('.nii_mean_reg', '_mean'),
                     ('.nii.par', '.par'),
                     ]
    subjFolders = [('fwhm-%s/' % f, 'fwhm-%s_' % f) for f in fwhm]
    substitutions.extend(subjFolders)
    
    datasink = Node(DataSink(base_directory=experiment_dir, container=output_dir),
                    name="datasink_prepro")
    datasink.inputs.substitutions = substitutions
    
    datasink2 = Node(DataSink(base_directory=experiment_dir, container=output_dir2),
                    name="datasink_metrics")
    datasink2.inputs.substitutions = substitutions
    
    
    #------------------------------------6.6 Nodos Elimininacion de ruido--------------------
    
    ICA = Node(MELODIC(report = True),
               name="Descomposition_ICA")
    
    Selec=Node(Function(input_names=['in_dir', 'datapy'], output_names=['melodic_mix', 'noise'], function=pif.SelecICA),
               name='Selection_files')
    
    Auto_CNN=Node(Function(input_names=['SUB', 'DirMod', 'DirPy'], output_names=['tex', 'npy', 'keyOut'], function=pif.autoCNN),
                  name='Classification')
    Auto_CNN.inputs.DirMod=CNN_H5
    Auto_CNN.inputs.DirPy=opj(experiment_dir, working_dir, 'automaticclassificationcnn.py')
    
    D_ICA=Node(FilterRegressor(),
                name='Denoising_ICA')
    
    DenoAR=Node(Function(input_names=['path_aroma', 'in_file', 'mat_file', 'par_file', 'tr'], output_names=['file_aggr', 'file_nonaggr'], function=pif.Ica_Aroma),
                name='Denoising_ICA-AROMA')
    DenoAR.inputs.path_aroma=Aropy
    DenoAR.inputs.tr=TR
    
    FilterAR=MapNode(Function(input_names=['files', 'lowpass_freq', 'highpass_freq', 'fs'], output_names=['out_file'], function=pif.bandpass_filter),
                     iterfield=['files'],
                     name='filters_pass_high_aroma')
    FilterAR.inputs.lowpass_freq=1./TR
    FilterAR.inputs.highpass_freq=0
    FilterAR.inputs.fs=1./TR
    
    Comp_cor = Node(TCompCor(num_components = componets_compcor, pre_filter = 'polynomial',  regress_poly_degree = 2, percentile_threshold = .03),  name="TCompCor")
    
    signG = Node(Function(input_names=['in_file'], output_names=['path_global'], function=pif.global_S),
                 name='Global_signal')
        
    RegComp2 = Node(Function(input_names=['in_file', 'regressor','ref_name'], output_names=['out_file', 'out_res'], function=pif.GLM2),
                    name='Regressor_CompCor')
    RegComp2.inputs.ref_name='_compcor'
        
    RegGlob2 = Node(Function(input_names=['in_file', 'regressor','ref_name'], output_names=['out_file', 'out_res'], function=pif.GLM2),
                    name='Regressor_Global')
    RegGlob2.inputs.ref_name='_GSR' 
    
    art = Node(ArtifactDetect(norm_threshold=2, zintensity_threshold=3, mask_type='spm_global', parameter_source='FSL', use_differences=[True, False], plot_type='svg'),
               name="art")
    
    jumps = Node(Function(input_names=['in_par'], output_names=['out_bin', 'out_peaks'], function=pif.jump_detecter),
                 name='signal_jumps')
    
    scrubbing=Node(Function(input_names=['in_file', 'outliers', 'peaks', 'method', 'thres'], output_names=['out_file'], function=pif.scrubbing_vol),
                 name='scrub')
    scrubbing.inputs.thres=3
    scrubbing.iterables = ("method", ['both','outliers','peaks']) 
    
    TEXTO=Node(Function(input_names=['uno', 'dos', 'tres'],
                        output_names=[],
                        function=pif.mostrar),
               name='Text')
    
    #------------------------------------6.7 Nodos Integracion e iteracion-------------------
        
    def f_iterador3(in_list, index):
        return in_list[index]
        
    def Integrate_Files(file_1,file_2):
        return list([file_1, file_2])
    
    def Integrate_three(file_1,file_2):
        file_1.append(file_2)
        return list(file_1)    
    
    Integ_Ar=Node(Function(input_names=['file_1', 'file_2'], output_names=['list'], function=Integrate_Files),
                name='Integrate_files')
    
    Integ_Base=Node(Function(input_names=['file_1', 'file_2'], output_names=['list'], function=Integrate_three),
                name='Base_files')
    
    iterador3=Node(Function(input_names=['in_list', 'index'],
                        output_names=['out_file'],
                        function=f_iterador3),
               name='iterador_3')
    iterador3.iterables = ("index", list(range(3)))
    
    integrado_cnn=Node(Function(input_names=['file_1', 'file_2'], output_names=['list'], function=Integrate_Files),
                name='Integrate_cnn')
    
    iterador4=Node(Function(input_names=['in_list', 'index'],
                        output_names=['out_file'],
                        function=f_iterador3),
               name='iterador_4')
    iterador4.iterables = ("index", list(range(2)))
    
    integrado_com=Node(Function(input_names=['file_1', 'file_2'], output_names=['list'], function=Integrate_Files),
                name='Integrate_compcor')
    
    iterador5=Node(Function(input_names=['in_list', 'index'],
                        output_names=['out_file'],
                        function=f_iterador3),
               name='iterador_5')
    iterador5.iterables = ("index", list(range(2)))
    
    integrado_gsr=Node(Function(input_names=['file_1', 'file_2'], output_names=['list'], function=Integrate_Files),
                name='Integrate_gsr')
    
    iterador6=Node(Function(input_names=['in_list', 'index'],
                        output_names=['out_file'],
                        function=f_iterador3),
               name='iterador_6')
    iterador6.iterables = ("index", list(range(2)))
    
    #------------------------------------6.8 Flujo Completo-----------------------
    
    # Create a preprocessing workflow
    preproc = Workflow(name='preproc')
    preproc.base_dir = opj(experiment_dir, working_dir) #Une los caracteres con un /
    
    # Connect all components of the preprocessing workflow
    preproc.connect([(infosource, selectfiles, [('asubject_id', 'asubject_id'),     ('session_num', 'session_num')]),
                     (selectfiles, extract, [('func', 'in_file')]),
                     (extract, mcflirt2, [('roi_file', 'in_file')]),
                     (mcflirt2, slicetimer, [('out_file', 'in_file')]),
                     (selectfiles, coregwf, [('anat', 'bet_anat.in_file'),
                                             ('anat', 'coreg_bbr.reference')]),
                     (mcflirt2, coregwf, [('mean_img', 'coreg_pre.in_file'),
                                          ('mean_img', 'Normalization.in_file'),
                                          ('mean_img', 'coreg_bbr.in_file')]),
                     (slicetimer, coregwf, [('slice_time_corrected_file', 'applywarp.in_file')]),
                     (coregwf, smooth, [('applywarp.out_file', 'PATH_GZ')]),
                     (smooth, Filter2, [('out_file', 'files')]), 
                     
                     #ICA-AROMA (genera aroma_aggr aroma_nonaggr)
                     (smooth, DenoAR, [('out_file','in_file')]),
                     (coregwf, DenoAR, [('Normalization.out_matrix_file','mat_file')]),
                     (mcflirt2 ,DenoAR, [('par_file','par_file')]),
                     (DenoAR, Integ_Ar, [('file_aggr','file_1'), ('file_nonaggr','file_2')]),             
                     (Integ_Ar, FilterAR, [('list','files')]),
                     
                     #Archivos Base (list) lista de archivos
                     (FilterAR, Integ_Base, [('out_file','file_1')]),
                     (Filter2,  Integ_Base, [('out_file','file_2')]),
                     (Integ_Base, iterador3, [('list','in_list')]),
                     
                     
                     #ICA-CNN (agrega regfilt)               
                     (iterador3, ICA, [("out_file", "in_files")]),                 
                     (ICA, Selec, [("out_dir", "in_dir")]),
                     (ICA, Auto_CNN,[("out_dir", "SUB")]),                
                     (Auto_CNN, Selec, [("npy", "datapy")]),
                     (iterador3, D_ICA, [("out_file", "in_file")]),
                     (Selec, D_ICA, [("melodic_mix", "design_file"),
                                     ("noise", "filter_columns")]),
                     
                     (iterador3, integrado_cnn, [("out_file", "file_1")]),
                     (D_ICA, integrado_cnn, [("out_file", "file_2")]),
                     (integrado_cnn, iterador4, [("list", "in_list")]),   
        
    
                     #TcompCor (agrega compcor_gml)
                     (iterador4, Comp_cor, [('out_file', 'realigned_file')]),
                     (iterador4, RegComp2, [('out_file', 'in_file')]),
                     (Comp_cor, RegComp2, [('components_file', 'regressor')]),
    
                     (iterador4, integrado_com, [("out_file", "file_1")]),
                     (RegComp2, integrado_com, [("out_res", "file_2")]),
                     (integrado_com, iterador5, [("list", "in_list")]),
                    
                     #Regresion de señal global (agrega GSR_GML)
                     (iterador5, signG, [('out_file', 'in_file')]),
                     (iterador5, RegGlob2, [('out_file', 'in_file')]),
                     (signG, RegGlob2,[('path_global', 'regressor')]),
    
                     (iterador5, integrado_gsr, [("out_file", "file_1")]),
                     (RegGlob2, integrado_gsr, [("out_res", "file_2")]),
                     (integrado_gsr, iterador6, [("list", "in_list")]),
        
                     #outliers and jumps
                     (coregwf, art, [('applywarp.out_file', 'realigned_files')]),
                     (mcflirt2, art, [('par_file', 'realignment_parameters')]),
                     (mcflirt2, jumps,[('par_file', 'in_par')]),
                     
                     #Scrubbing
                     (iterador6, scrubbing, [('out_file', 'in_file')]),
                     (art, scrubbing, [('outlier_files', 'outliers')]),
                     (jumps, scrubbing, [('out_bin', 'peaks')]),
                      
                     #Organizar los datos de salida
                     (Filter2, datasink, [('out_file', 'preproc.@Filtrado')]),
                     (DenoAR, datasink, [('file_aggr', 'preproc.@denoising_aroma_aggr'),
                                        ('file_nonaggr', 'preproc.@denoising_aroma_nonaggr')]),                 
                     (D_ICA, datasink, [('out_file', 'preproc.@denoising_cnn')]),                 
                     (RegComp2, datasink, [('out_res', 'preproc.@tcompcor_res')]),
                     (RegGlob2, datasink, [('out_res', 'preproc.@GSR_res')]),
                     (scrubbing, datasink,[('out_file', 'preproc.@scrub')]), 
                     ])
    
    
    
    #------------------------------------Run------------------------
    
    preproc.run('MultiProc', plugin_args={'n_procs': 4})





