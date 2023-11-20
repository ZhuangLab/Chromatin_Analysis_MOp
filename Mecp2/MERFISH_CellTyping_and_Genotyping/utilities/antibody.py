import sys
sys.path.append(r'C:\Users\cosmosyw\Documents\Softwares')

import pandas as pd
import numpy as np
import h5py

import ImageAnalysis3 as ia
from ImageAnalysis3.segmentation_tools.cell import Align_Segmentation
from scipy.ndimage import shift
from scipy.ndimage import binary_dilation
from collections import Counter

# load correction files
# folder contains information for bleedthrough correction, illumination correction, chromatic correction
correction_folder = r'\\10.245.74.158\Chromatin_NAS_0\Corrections\20210621-Corrections_lumencor_from_60_to_50'

# transpose with microscope
microscope_file = r'C:\Users\cosmosyw\Documents\Softwares\Merfish_Analysis_Scripts\merlin_parameters\microscope\storm6_microscope.json'
microscope_params = Align_Segmentation._read_microscope_json(microscope_file)


def calculate_mecp2_signal(dax_file, feature_file, offset_file, kept_uids):
    ### load MeCP2 image
    mecp2_cls = ia.classes.preprocess.DaxProcesser(dax_file, CorrectionFolder=correction_folder, DriftChannel=488, DapiChannel=405)
    mecp2_cls._load_image(sel_channels=[647,405])
    mecp2_cls._corr_illumination()

    dapi_im = Align_Segmentation._correct_image3D_by_microscope_param(mecp2_cls.im_405, microscope_params)
    mecp2_im = Align_Segmentation._correct_image3D_by_microscope_param(mecp2_cls.im_647, microscope_params)
    del mecp2_cls

    ### adjust the z stack
    if dapi_im.shape[0]==50:
        dapi_im = dapi_im[::4]
        mecp2_im = mecp2_im[::4]

    ### load transformation file
    transformation = np.load(offset_file, allow_pickle=True)[-1]
    drift = [0, transformation.translation[1], transformation.translation[0]]

    ### dictionary to contain all info
    dict_cell = {}
    dict_cell['uid'] = []
    dict_cell['fov'] = []
    dict_cell['cellID'] = []
    dict_cell['DAPI_mean'] = []
    dict_cell['MeCP2_mean'] = []
    dict_cell['MeCP2_internal_variation'] = []
    dict_cell['local_MeCP2_signal'] = []

    ### load feature file
    with h5py.File(feature_file) as f:

        ### load dapi segmentation and apply fiducial
        dapi_segment = np.copy(f['dapi_label']['label3D'])
        if dapi_segment.shape[0]==50:
            dapi_segment = dapi_segment[::4]
        dapi_segment = shift(dapi_segment, drift, order=0, mode='nearest')       


        for _cell in f['featuredata'].keys():
            if _cell not in kept_uids:
                continue
            ### load cell mask
            cell_mask = dapi_segment==f['featuredata'][_cell].attrs['label']
            z_coords = np.unique(np.where(cell_mask)[0])
            
            if len(z_coords)==0:
                continue
            
            z_for_analysis = z_coords[0]
            _temp_max = 0
            for _z in z_coords:
                _current_max = np.mean(dapi_im[_z][np.where(cell_mask[_z])])
                if _current_max>_temp_max:
                    z_for_analysis = _z
                    _temp_max = _current_max

            # load basic info
            dict_cell['uid'].append(str(_cell))
            dict_cell['fov'].append(f['featuredata'][_cell].attrs['fov'])
            dict_cell['cellID'].append(f['featuredata'][_cell].attrs['label'])

            cell_dapi = dapi_im[z_for_analysis]
            cell_mecp2 = mecp2_im[z_for_analysis]
            cell_mask = cell_mask[z_for_analysis]
            x_mecp2 = np.where(cell_mask)[0]
            y_mecp2 = np.where(cell_mask)[1]
            x_mid = (np.max(x_mecp2)+np.min(x_mecp2))/2
            y_mid = (np.max(y_mecp2)+np.min(y_mecp2))/2

            dapi_mean = np.mean(cell_dapi[x_mecp2, y_mecp2])
            mecp2_mean = np.mean(cell_mecp2[x_mecp2, y_mecp2])

            ### calculate the mecp2 intensity on the peripheral. v1
            cell_mask_dilate_large = binary_dilation(cell_mask, structure = np.ones((49,49)))
            cell_mask_dilate_small = binary_dilation(cell_mask, structure = np.ones((39,39)))
            outskirt_mask = (cell_mask_dilate_small!=cell_mask_dilate_large)&(dapi_segment[z_for_analysis]==0)
            outskirt_xy = np.where(outskirt_mask)
            background_mecp2_mean = np.mean(cell_mecp2[outskirt_xy])

            ### calculate the quandrant coordinates
            q1_x, q1_y, q2_x, q2_y, q3_x, q3_y, q4_x, q4_y = [], [], [], [], [], [], [], []
            for (_x, _y) in zip(x_mecp2, y_mecp2):
                if _x<=x_mid:
                    if _y<=y_mid:
                        q1_x.append(_x)
                        q1_y.append(_y)
                    else:
                        q2_x.append(_x)
                        q2_y.append(_y)
                else:
                    if _y<=y_mid:
                        q3_x.append(_x)
                        q3_y.append(_y)
                    else:
                        q4_x.append(_x)
                        q4_y.append(_y)
            q1 = np.mean(cell_mecp2[q1_x, q1_y])
            q2 = np.mean(cell_mecp2[q2_x, q2_y])
            q3 = np.mean(cell_mecp2[q3_x, q3_y])
            q4 = np.mean(cell_mecp2[q4_x, q4_y])
            mecp2_std = np.std(np.array([q1,q2,q3,q4]))

            dict_cell['DAPI_mean'].append(dapi_mean)
            dict_cell['MeCP2_mean'].append(mecp2_mean)
            dict_cell['MeCP2_internal_variation'].append(mecp2_std/mecp2_mean)
            dict_cell['local_MeCP2_signal'].append(np.log2(mecp2_mean/background_mecp2_mean))

    df_cell = pd.DataFrame(dict_cell)

    return df_cell