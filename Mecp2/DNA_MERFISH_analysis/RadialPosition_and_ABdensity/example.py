'''
This file contains the python function used to calculate radial position and AB density ratio, as used in the jupyter notebook.
'''

'''
Following is the code used to calculate radial position
'''
position_columns = ['z_um', 'x_um', 'y_um']

import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial import ConvexHull

def get_intersection(U,hull):
    eq=hull.equations.T
    V,b=eq[:-1],eq[-1]
    alpha=-b/np.dot(U,V)
    return U*np.full(np.array(U).shape, np.min(alpha[alpha>0]))

def Radial_positioning_from_center_for_loci (df_uid, pos_col=position_columns):
    
    # copy the input dataframe
    output_df = df_uid.copy()
    # generate allpoints
    points = output_df.loc[:, pos_col].values
    # generate convex hull
    hull = ConvexHull(points)
    # get hull centroid
    cz = np.mean(hull.points[hull.vertices,0])
    cx = np.mean(hull.points[hull.vertices,1])
    cy = np.mean(hull.points[hull.vertices,2])
    # attention: zxy in correct order!!
    _hull_ct_zxy = np.array([cz,cx,cy])
    # update point and hull
    points = points - _hull_ct_zxy
    hull = ConvexHull(points)
    # calculate distance to center and radial positions
    distances_to_center = []
    radial_positions = []
    for i, row in output_df.iterrows():
        _zxys = [row[pos_col[0]],row[pos_col[1]],row[pos_col[2]]]-_hull_ct_zxy
        _dist_to_center = euclidean(_zxys, [0,0,0])
        distances_to_center.append(_dist_to_center)
        _edge_pt = get_intersection(_zxys, hull)
        _loci_rp = (_dist_to_center/euclidean(_edge_pt, [0,0,0]))
        radial_positions.append(_loci_rp)
    output_df['distance_to_center'] = distances_to_center
    output_df['Norm_RP'] = radial_positions
    return output_df

'''
Following is the code used to calculate AB density ratio
'''
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd

position_columns = ['z_um', 'x_um', 'y_um']

# function to get subclass information
def get_AB_subclass(chrom, hyb, subclass, df_ab):
    
    if subclass=='Endo' and 'Endo' not in df_ab.columns:
        cell_type_key = 'Endo-PVM'
    elif subclass=='Endo-PVM' and 'Endo-PVM' not in df_ab.columns:
        cell_type_key = 'Endo'
    else:
        cell_type_key = subclass
    
    if len(df_ab.loc[(df_ab['chr']==chrom)&(df_ab['hyb']==hyb), cell_type_key].values)==0:
        print(cell_type_key)
        raise ValueError(f'No cell type key {cell_type_key}!')
    else:
        return df_ab.loc[(df_ab['chr']==chrom)&(df_ab['hyb']==hyb), cell_type_key].values[0]
    
# function to calculate AB density
def calculate_gaussian_density_from_distance(distances, sigma=0.5, 
                               intensity=1, background=0):
    sigma = np.array(sigma, dtype=float)
    g_pdf = np.exp(-0.5 * distances**2 / sigma**2)
    g_pdf = 1/sigma*np.sqrt(2*np.pi)* float(intensity) * g_pdf + float(background)
    return g_pdf

# function to multiprocessing the calculation of AB density ratio
def calculate_AB_score_original_Trans(df_uid, sigma, df_ab, pos_col=position_columns):
    # copy 
    output_df = df_uid.copy()
    # get AB
    if 'subclass' in df_uid.columns:
        celltype_key = 'subclass'
    elif 'CellType' in df_uid.columns:
        celltype_key = 'CellType'
    elif 'class' in df_uid.columns:
        celltype_key = 'class'
    else:
        celltype_key = 'majorType'
    output_df['AB'] = output_df.apply(lambda x: get_AB_subclass(x['chr'], x['hyb'], x[celltype_key], df_ab), axis=1)
    # reset_index
    output_df.reset_index(drop=True, inplace=True)
    # initialize
    local_AB_density = []
    # get distance
    all_pts = output_df.loc[:, pos_col].values.astype(float)
    pairwise_distance = cdist(all_pts, all_pts)
    for i, row in output_df.iterrows():
        # select loci that is not on the same chromosome fiber, aka trans loci
        drop_indices = output_df[(output_df['chr']==row['chr'])&(output_df['fiberidx']==row['fiberidx'])].index.values
        df_trans = output_df.drop(drop_indices)
        A_indices = df_trans[df_trans['AB']=='A'].index.values
        B_indices = df_trans[df_trans['AB']=='B'].index.values
        distance_to_A = pairwise_distance[i, A_indices]
        distance_to_B = pairwise_distance[i, B_indices]
        A_scores = calculate_gaussian_density_from_distance(distance_to_A, sigma=sigma).sum()
        B_scores = calculate_gaussian_density_from_distance(distance_to_B, sigma=sigma).sum()
        local_AB_density.append(np.log(A_scores)-np.log(B_scores))
    output_df['AB_density'] = local_AB_density
    return output_df