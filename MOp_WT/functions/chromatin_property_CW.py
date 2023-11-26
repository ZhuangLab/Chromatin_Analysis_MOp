import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, cdist
from itertools import combinations


pos_cols = ['z_um', 'x_um', 'y_um']

def calculate_rg(df, scale_factor=1):
    # this function computes the radii of gyration of a chromosome.
    # the scale_factor should be the square-root of the ratio of end-to-end distance of picked points over all potential points.
    # the square-root is taken because r^2 ~ 2/3*lp*L
    all_pts = df.loc[:, pos_cols].values
    center = np.mean(all_pts, axis=0)
    distance = np.sum(np.square(all_pts - center), axis=1)
    rg = np.mean(distance)
    
    return np.sqrt(rg)*scale_factor

def calculate_insulation(df_1, df_2):
    # this function computes insulation between two set of points stored in two dataframe
    # the insulation score is calculated similar as Su et al., Cell, 2020
    # pairwise distances for df_1 and df_2 are calculated and median is taken
    # pairwise distances between df_1 and df_2 are calcualted and median is taken
    # the inter over intra median distance is the insulation score
    intra_1 = pdist(df_1[pos_cols])
    intra_2 = pdist(df_2[pos_cols])
    intra = np.append(intra_1, intra_2)
    inter = cdist(df_1[pos_cols], df_2[pos_cols])
    intra_median = np.median(intra)
    inter_median = np.median(inter)
    return inter_median/intra_median

def calculate_volume_by_convex_hull(df, scale_factor=1, output_all_info=True):
    convex_hull = ConvexHull(df.loc[:,pos_cols].values)
    volume = convex_hull.volume

    if output_all_info:
        return np.array([df.uid.values[0], df.chr.values[0], df.fiberidx.values[0], volume*scale_factor])
    else:
        return volume*scale_factor

def calculate_rg_multiprocessing(df_fiber:pd.DataFrame, info_cols = ['uid', 'majorType', 'subclass', 'chr', 'fiberidx']):
    
    rg = calculate_rg(df_fiber.copy())

    dict_output = {'rg':rg}
    for col in info_cols:
        dict_output[col] = df_fiber[col].values[0]
    
    return pd.DataFrame(dict_output, index=[0])

def calculate_chr_insulation(df_cell:pd.DataFrame, info_cols = ['uid', 'majorType', 'subclass', 'chr', 'fiberidx']):

    # initialize dictionary to store necessary information
    dict_insulation = {}
    for col in info_cols:
        if col in ['chr', 'fiberidx']:
            col_1 = col+'_1'
            col_2 = col+'_2'
            dict_insulation[col_1]=[]
            dict_insulation[col_2]=[]
        else:
            dict_insulation[col]=[]
    dict_insulation['insulation_score'] = []
    
    df_fibers = []
    for (_,_), df_fiber in df_cell.groupby(['chr', 'fiberidx']):
        df_fibers.append(df_fiber)

    other_info_cols = [col for col in info_cols if col not in ['chr', 'fiberidx']]
    
    for (df_1, df_2) in combinations(df_fibers,2):
        dict_insulation['chr_1'].append(df_1['chr'].values[0])
        dict_insulation['fiberidx_1'].append(df_1['fiberidx'].values[0])
        dict_insulation['chr_2'].append(df_2['chr'].values[0])
        dict_insulation['fiberidx_2'].append(df_2['fiberidx'].values[0])
        
        for info_col in other_info_cols:
            dict_insulation[info_col].append(df_1[info_col].values[0])

        dict_insulation['insulation_score'].append(calculate_insulation(df_1,df_2))
    
    df_insulation = pd.DataFrame(dict_insulation)
    return df_insulation