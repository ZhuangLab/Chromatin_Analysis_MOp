import numpy as np
import multiprocessing as mp
import time
import pandas as pd
# functions
from itertools import combinations_with_replacement, permutations, combinations
from scipy.spatial.distance import pdist, squareform, cdist
from tqdm import tqdm




# Script adapated from distance.py from PZ's Im3 package;
# 1. For "Chr2ZxysList_2_summaryDist_by_key", Add a data_num_th size filtering so only pairs with enough data are reported
# 2. Add an adapted version as Chr2ZxysList_2_summaryDist_by_key_V2 to restric analysis to close chromosome pair only;
# 3. The adapted version can also analyze a subset of loci and can report raw dist

# shared parameters
default_num_threads = 8

# For a pair of chromosome, summarize
def Chr2ZxysList_2_summaryDist_by_key(chr_2_zxys_list, _c1, _c2, codebook_df,
                                 function='nanmedian', axis=0, data_num_th=100,
                                 _contact_th=600, contact_prob=False,
                                 verbose=False):
    _out_dist_dict = {}
    if _c1 != _c2:
        _out_dist_dict[(_c1,_c2)] = []
    else:
        _out_dist_dict[f"cis_{_c1}"] = []
        _out_dist_dict[f"trans_{_c1}"] = []
    for _chr_2_zxys in chr_2_zxys_list:
        # skip if not all info exists
        if _c1 not in _chr_2_zxys or _c2 not in _chr_2_zxys or _chr_2_zxys[_c1] is None or _chr_2_zxys[_c2] is None:
            continue
        else:
            # if not from the same chr label, calcluate trans-chr with cdist
            if _c1 != _c2:
                for _zxys1 in _chr_2_zxys[_c1]:
                    for _zxys2 in _chr_2_zxys[_c2]:
                        _out_dist_dict[(_c1,_c2)].append(cdist(_zxys1, _zxys2))
            # if from the same chr label, calculate both cis and trans
            else:
                # cis
                _out_dist_dict[f"cis_{_c1}"].extend([squareform(pdist(_zxys)) for _zxys in _chr_2_zxys[_c1]])
                # trans
                if len(_chr_2_zxys[_c1]) > 1:
                    # loop through permutations
                    for _i1, _i2 in permutations(np.arange(len(_chr_2_zxys[_c1])), 2):
                        _out_dist_dict[f"trans_{_c1}"].append(
                            cdist(_chr_2_zxys[_c1][_i1], _chr_2_zxys[_c1][_i2])
                        )
    for _key in _out_dist_dict:
        print(_key, len(_out_dist_dict[_key]))
    #return _out_dist_dict
    # summarize
    _summary_dict = {}
    all_chrs = [str(_chr) for _chr in np.unique(codebook_df['chr'])]
    all_chr_sizes = {_chr:np.sum(codebook_df['chr']==_chr) for _chr in all_chrs}
    for _key, _dists_list in _out_dist_dict.items():
        
        if len(_dists_list) >= 50:
            if contact_prob:
                # get nan positions
                _nan_index = np.isnan(np.array(_dists_list))
                # th value comparision and restore nan
                _dists_contact_list=np.array(_dists_list)<=_contact_th
                # convert array as float type to enable nan re-assignment through indexing; 
                # otherwise all nan will be treated as True for bool
                _dists_contact_list=_dists_contact_list*1.0
                _dists_contact_list[_nan_index==1]=np.nan
                _summary_result = getattr(np, 'nanmean')(_dists_contact_list, axis=axis)
                # keep result if non-nan N > data number th
                _valid_index = np.sum(~np.isnan(_dists_contact_list),axis=0)>=data_num_th
                _summary_result[_valid_index==0]=np.nan
                _summary_dict[_key] = _summary_result
            else:
                # summarize but only keep result if non-nan N > data number th
                #_summary_dict[_key] = getattr(np, function)(_dists_list, axis=axis)
                _summary_result =getattr(np, function)(_dists_list, axis=axis)
                # keep result if non-nan N > data number th
                _valid_index = np.sum(~np.isnan(_dists_list),axis=0)>=data_num_th
                _summary_result[_valid_index==0]=np.nan
                _summary_dict[_key] = _summary_result
        else:
            if isinstance(_key, str): # cis or trans
                _chr = _key.split('_')[-1] 
                _summary_dict[_key]= np.nan * np.ones([all_chr_sizes[_chr], all_chr_sizes[_chr]])
            else:
                _chr1, _chr2 = _key
                _summary_dict[_key]= np.nan * np.ones([all_chr_sizes[_chr1], all_chr_sizes[_chr2]])
    
    return _summary_dict



# call previous function to calculate all pair-wise chromosomal distance
def Chr2ZxysList_2_summaryDict(
    chr_2_zxys_list, total_codebook,
    function='nanmedian', axis=0, data_num_th=100,
    _contact_th=500, contact_prob=False,
    parallel=True, num_threads=default_num_threads,
    verbose=False):
    """Function to batch process chr_2_zxys_list into summary_dictionary"""
    if verbose:
        print(f"-- preparing chr_2_zxys from {len(chr_2_zxys_list)} cells", end=' ')
        _start_prepare = time.time()
    _summary_args = []
    # prepare args
    _all_chrs = np.unique(total_codebook['chr'].values)
    #sorted(_all_chrs, key=lambda _c:sort_mouse_chr(_c))
    for _chr1, _chr2 in combinations_with_replacement(_all_chrs, 2):
        if _chr1 != _chr2:
            _sel_chr_2_zxys = [
                {_chr1: _d.get(_chr1, None),
                 _chr2: _d.get(_chr2, None)} for _d in chr_2_zxys_list
            ]
        else:
            _sel_chr_2_zxys = [
                {_chr1: _d.get(_chr1, None)} for _d in chr_2_zxys_list
            ]
        _summary_args.append(
            (_sel_chr_2_zxys, _chr1, _chr2, total_codebook, function, axis, data_num_th, _contact_th, contact_prob,verbose)
        )
    if verbose:
        print(f"in {time.time()-_start_prepare:.3f}s.")
    # process
    _start_time = time.time()
    if parallel:
        if verbose:
            print(f"-- summarize {len(_summary_args)} inter-chr distances with {num_threads} threads", end=' ')
        with mp.Pool(num_threads) as _summary_pool:
            all_summary_dicts = _summary_pool.starmap(
                Chr2ZxysList_2_summaryDist_by_key, 
                _summary_args, chunksize=1)
            _summary_pool.close()
            _summary_pool.join()
            _summary_pool.terminate()
        if verbose:
            print(f"in {time.time()-_start_time:.3f}s.")
    else:
        if verbose:
            print(f"-- summarize {len(_summary_args)} inter-chr distances sequentially", end=' ')
        all_summary_dicts = [Chr2ZxysList_2_summaryDist_by_key(*_args) for _args in _summary_args]
        if verbose:
            print(f"in {time.time()-_start_time:.3f}s.")
    # summarize into one dict
    _summary_dict = {}
    for _dict in all_summary_dicts:
        _summary_dict.update(_dict)
    return _summary_dict

# sort chromosome order
def sort_mouse_chr(_chr):
    try:
        out_key = int(_chr)
    except:
        if _chr == 'X':
            out_key = 23
        elif _chr == 'Y':
            out_key = 24
    return out_key



# Generate a chromosome plot order by either region id in codebook, or by chromosome
def Generate_PlotOrder(total_codebook, sel_codebook, sort_by_region=True):
    """Function to cleanup plot_order given total codebook and selected codebook"""
    chr_2_plot_indices = {}
    chr_2_chr_orders = {}
    _sel_Nreg = 0
    for _chr in sorted(np.unique(total_codebook['chr']), key=lambda _c:sort_mouse_chr(_c)):
        _chr_codebook = total_codebook[total_codebook['chr']==_chr]

        _reg_ids = _chr_codebook['id'].values 
        _orders = _chr_codebook['chr_order'].values
        _chr_sel_inds, _chr_sel_orders = [], []
        for _id, _ord in zip(_reg_ids, _orders):
            if _id in sel_codebook['id'].values:
                # this ensures returning of iloc when the index col is not by number
                #_chr_sel_inds.append(sel_codebook[sel_codebook['id']==_id].index[0])
                _iloc_ind = sel_codebook.index.get_loc(sel_codebook[sel_codebook['id']==_id].index[0])
                _chr_sel_inds.append(_iloc_ind)
                _chr_sel_orders.append(_ord)
        if len(_chr_sel_inds) == 0:
            continue
        # append
        if sort_by_region:
            chr_2_plot_indices[_chr] = np.array(_chr_sel_inds)
            chr_2_chr_orders[_chr] = np.array(_chr_sel_orders)
        else: # sort by chr
            chr_2_plot_indices[_chr] = np.arange(_sel_Nreg, _sel_Nreg+len(_chr_sel_inds))
            chr_2_chr_orders[_chr] = np.arange(len(_chr_sel_inds))
        # update num of selected regions
        _sel_Nreg += len(_chr_sel_inds)
    return chr_2_plot_indices, chr_2_chr_orders


# Summarize summary_dict into a matrix, using plot order generated by previous function
def assemble_ChrDistDict_2_Matrix(dist_dict, 
                                  total_codebook, sel_codebook=None,
                                  use_cis=True, sort_by_region=True):
    """Assemble a dist_dict into distance matrix shape"""
    if sel_codebook is None:
        sel_codebook = total_codebook
    # get indices
    chr_2_plot_inds, chr_2_chr_orders = Generate_PlotOrder(
        total_codebook, sel_codebook, sort_by_region=sort_by_region)
    # init plot matrix
    _matrix = np.ones([len(sel_codebook),len(sel_codebook)]) * np.nan
    # loop through chr
    for _chr1 in sorted(np.unique(total_codebook['chr']), key=lambda _c:sort_mouse_chr(_c)):
        for _chr2 in sorted(np.unique(total_codebook['chr']), key=lambda _c:sort_mouse_chr(_c)):
            if _chr1 not in chr_2_plot_inds or _chr2 not in chr_2_plot_inds:
                continue
            # get plot_inds
            _ind1, _ind2 = chr_2_plot_inds[_chr1], chr_2_plot_inds[_chr2]
            _ords1, _ords2 = chr_2_chr_orders[_chr1], chr_2_chr_orders[_chr2]
            # if the same chr, decide using cis/trans
            if _chr1 == _chr2:
                if use_cis and f"cis_{_chr1}" in dist_dict:
                    _matrix[_ind1[:, np.newaxis], _ind2] = dist_dict[f"cis_{_chr1}"][_ords1[:, np.newaxis], _ords2]
                elif not use_cis and f"trans_{_chr1}" in dist_dict:
                    _matrix[_ind1[:, np.newaxis], _ind2] = dist_dict[f"trans_{_chr1}"][_ords1[:, np.newaxis], _ords2]
            # if not the same chr, get trans_chr_mat
            else:
                if (_chr1, _chr2) in dist_dict:
                    _matrix[_ind1[:, np.newaxis], _ind2] = dist_dict[(_chr1, _chr2)][_ords1[:, np.newaxis], _ords2]
                    _matrix[_ind2[:, np.newaxis], _ind1] = dist_dict[(_chr1, _chr2)][_ords1[:, np.newaxis], _ords2].transpose()
                elif (_chr2, _chr1) in dist_dict:
                    _matrix[_ind1[:, np.newaxis], _ind2] = dist_dict[(_chr2, _chr1)][_ords2[:, np.newaxis], _ords1].transpose()
                    _matrix[_ind2[:, np.newaxis], _ind1] = dist_dict[(_chr2, _chr1)][_ords2[:, np.newaxis], _ords1]
    # assemble references
    _chr_edges, _chr_names = generate_plot_chr_edges(sel_codebook, chr_2_plot_inds, sort_by_region)
    return _matrix, _chr_edges, _chr_names



# sort chr_2_chr_orders generated from sel_codebook [12,50, 67] to a pseduo order by their argsort ind [0,1,2]
# use with the function below
def convert_chr_2_chr_orders (chr_2_chr_orders):
    new_chr_2_chr_orders = {}
    for _chr_key, _chr_order_arr in  chr_2_chr_orders.items():
        _new_chr_order_arr = np.argsort(_chr_order_arr)
        new_chr_2_chr_orders[_chr_key] = _new_chr_order_arr
    
    return new_chr_2_chr_orders


# Summarize summary_dict into a matrix, using plot order generated by previous function
def assemble_ChrDistDict_2_Matrix_V2(dist_dict, 
                                  total_codebook, sel_codebook=None,
                                  use_cis=True, sort_by_region=True):
    """Assemble a dist_dict into distance matrix shape"""
    if sel_codebook is None:
        sel_codebook = total_codebook
    # get indices
    chr_2_plot_inds, chr_2_chr_orders = Generate_PlotOrder(
        total_codebook, sel_codebook, sort_by_region=sort_by_region)
    
    # convert selected chr_order to pseduo order by argsort
    chr_2_chr_orders = convert_chr_2_chr_orders(chr_2_chr_orders)

    # init plot matrix
    _matrix = np.ones([len(sel_codebook),len(sel_codebook)]) * np.nan
    # loop through chr
    for _chr1 in sorted(np.unique(total_codebook['chr']), key=lambda _c:sort_mouse_chr(_c)):
        for _chr2 in sorted(np.unique(total_codebook['chr']), key=lambda _c:sort_mouse_chr(_c)):
            if _chr1 not in chr_2_plot_inds or _chr2 not in chr_2_plot_inds:
                continue
            # get plot_inds
            _ind1, _ind2 = chr_2_plot_inds[_chr1], chr_2_plot_inds[_chr2]
            _ords1, _ords2 = chr_2_chr_orders[_chr1], chr_2_chr_orders[_chr2]
            # if the same chr, decide using cis/trans
            if _chr1 == _chr2:
                if use_cis and f"cis_{_chr1}" in dist_dict:
                    _matrix[_ind1[:, np.newaxis], _ind2] = dist_dict[f"cis_{_chr1}"][_ords1[:, np.newaxis], _ords2]
                elif not use_cis and f"trans_{_chr1}" in dist_dict:
                    _matrix[_ind1[:, np.newaxis], _ind2] = dist_dict[f"trans_{_chr1}"][_ords1[:, np.newaxis], _ords2]
            # if not the same chr, get trans_chr_mat
            else:
                if (_chr1, _chr2) in dist_dict:
                    _matrix[_ind1[:, np.newaxis], _ind2] = dist_dict[(_chr1, _chr2)][_ords1[:, np.newaxis], _ords2]
                    _matrix[_ind2[:, np.newaxis], _ind1] = dist_dict[(_chr1, _chr2)][_ords1[:, np.newaxis], _ords2].transpose()
                elif (_chr2, _chr1) in dist_dict:
                    _matrix[_ind1[:, np.newaxis], _ind2] = dist_dict[(_chr2, _chr1)][_ords2[:, np.newaxis], _ords1].transpose()
                    _matrix[_ind2[:, np.newaxis], _ind1] = dist_dict[(_chr2, _chr1)][_ords2[:, np.newaxis], _ords1]
    # assemble references
    _chr_edges, _chr_names = generate_plot_chr_edges(sel_codebook, chr_2_plot_inds, sort_by_region)
    return _matrix, _chr_edges, _chr_names



def generate_plot_chr_edges(sel_codebook, chr_2_plot_inds=None, sort_by_region=True):
    if chr_2_plot_inds is None or not isinstance(chr_2_plot_inds, dict):
        chr_2_plot_inds,_ = Generate_PlotOrder(sel_codebook, sel_codebook, sort_by_region=sort_by_region)
    # assemble references
    _chr_edges, _chr_names = [], []
    if sort_by_region:
        # loop through regions
        prev_chr = None
        for _ind, _chr in zip(sel_codebook.index, sel_codebook['chr']):
            if _chr != prev_chr:
                _chr_edges.append(_ind)
                _chr_names.append(_chr)
            prev_chr = _chr
        _chr_edges.append(len(sel_codebook))
    else:
        # loop through chr
        for _chr, _inds in chr_2_plot_inds.items():
            _chr_edges.append(_inds[0])
            _chr_names.append(_chr)
        _chr_edges.append(len(sel_codebook))
    _chr_edges = np.array(_chr_edges)
    return _chr_edges, _chr_names



# New functiions 1 below
############################################################################
# Function to calculate chromosome center
def get_chroZxys_mean_center (_ichr_zxys):
    x, y, z= np.nanmean(_ichr_zxys[:,1]),np.nanmean(_ichr_zxys[:,2]),np.nanmean(_ichr_zxys[:,0])
    _ichr_ct_zxy = np.array([x,y,z])
    return _ichr_ct_zxy


#############################################################################
# For a pair of chromosomes   
# Function to summarize Dist for chr pair meets following critera: 
# (1) trans-non-homolog_chr (2) chr centers within certain dist 
# optional: use dist between center to normalize 
# optional: analyze a subset of loci and can report raw dist

def Chr2ZxysList_2_summaryDist_by_key_V2 (chr_2_zxys_list, 
                                        _c1, 
                                        _c2, 
                                        sel_chr_loci_info_df,  # any codebook-like df that has 'chr' and 'chr_order' cols
                                        function='nanmedian', 
                                        axis=0, 
                                        data_num_th=50,
                                        _center_dist_th = 5000000,
                                        normalize_by_center=False,
                                        _contact_th=500, 
                                        contact_prob=False,
                                        interaction_type='trans',
                                        return_raw = False,  
                                        verbose=False):
    
    from scipy.spatial.distance import euclidean

    # Step 1. Adding dist_list by chr key accordingly (chr_order_ind, chr_center_dist, etc) 
    ########################################################################################
    print(f'-- Start analyzing neighboring chromosome pair within {_center_dist_th}nm.')

    all_chrs = [str(_chr) for _chr in np.unique(sel_chr_loci_info_df['chr'])]
    all_chr_sizes = {_chr:np.sum(sel_chr_loci_info_df['chr']==_chr) for _chr in all_chrs}

    sel_chr_order_ind_c1 = sel_chr_loci_info_df[sel_chr_loci_info_df['chr']==_c1]['chr_order'].tolist()
    sel_chr_order_ind_c2 = sel_chr_loci_info_df[sel_chr_loci_info_df['chr']==_c2]['chr_order'].tolist()
    
    _out_dist_dict = {}
    if _c1 != _c2:
        _out_dist_dict[(_c1,_c2)] = []
    else:
        _out_dist_dict[f"cis_{_c1}"] = []
        # skip for trans homolog analysis for now
        #_out_dist_dict[f"trans_{_c1}"] = []  
        
    for _chr_2_zxys in chr_2_zxys_list:
        # skip if not all info exists
        if _c1 not in _chr_2_zxys or _c2 not in _chr_2_zxys or _chr_2_zxys[_c1] is None or _chr_2_zxys[_c2] is None:
            continue
        else:
            # if not from the same chr label, calcluate trans-chr with cdist
            if _c1 != _c2 and interaction_type in ['trans','both']:
                # dual loop for each allel
                for _zxys1 in _chr_2_zxys[_c1]:
                    _center_zxys1 = get_chroZxys_mean_center(_zxys1)
                    for _zxys2 in _chr_2_zxys[_c2]:
                        _center_zxys2 = get_chroZxys_mean_center(_zxys2)
                        # only analyze neighboring chromosome pair within given distance
                        if euclidean(_center_zxys1,_center_zxys2)<= _center_dist_th:
                            # slice a subset of loci on the chromosome to analyze
                            sel_zxys1 = _zxys1[sel_chr_order_ind_c1,:]
                            sel_zxys2 = _zxys2[sel_chr_order_ind_c2,:]
                            _chr_pair_cdists = cdist(sel_zxys1, sel_zxys2)
                            if normalize_by_center and not contact_prob:
                                _chr_pair_cdists=_chr_pair_cdists/euclidean(_center_zxys1,_center_zxys2)
                            _out_dist_dict[(_c1,_c2)].append(_chr_pair_cdists)
                        else:
                            continue
            # if from the same chr label, calculate both cis and trans
            elif _c1 == _c2 and interaction_type in ['cis','both']:
                # cis (and slice subset)
                _out_dist_dict[f"cis_{_c1}"].extend([squareform(pdist(_zxys[sel_chr_order_ind_c1,:])) for _zxys in _chr_2_zxys[_c1]])
                # skip for trans-homolog analysis for now
                if len(_chr_2_zxys[_c1]) > 1:
                    continue
                    ## loop through permutations
                    #for _i1, _i2 in permutations(np.arange(len(_chr_2_zxys[_c1])), 2):
                        #_out_dist_dict[f"trans_{_c1}"].append(
                            #cdist(_chr_2_zxys[_c1][_i1], _chr_2_zxys[_c1][_i2]))
            else:
                continue

    # Step 2.1 Summarzie dist_list as raw values by chr and chr_order key 
    ########################################################################################                    
    # summarize result or not
    # format raw using chr_chr_order as key
    if return_raw:
        dist_by_cell_pair_dict = {}
        for _key, _dists_list in _out_dist_dict.items():
            if isinstance(_key, str): # cis or trans
                _chr1 = _key
                _chr2 = _key
            else:
                _chr1, _chr2 = _key
            dist_by_cell = np.array(_dists_list)
            for _ind_c1 in range(len(sel_chr_order_ind_c1)):
                for _ind_c2 in range(len(sel_chr_order_ind_c2)):
                    chr_order_c1 = sel_chr_order_ind_c1[_ind_c1]
                    chr_order_c2 = sel_chr_order_ind_c2[_ind_c2]
                    pair_key = ((_chr1, chr_order_c1),(_chr2, chr_order_c2))
                    # append depending on if the raw is empty (e.g., interaction not caluclated) or not
                    if len(dist_by_cell)>0:
                        dist_by_cell_pair  = dist_by_cell [:, _ind_c1, _ind_c2] 
                    else:
                        dist_by_cell_pair=np.array([np.nan])
                    dist_by_cell_pair_dict[pair_key]=dist_by_cell_pair

        print ('Return raw results by pair key.')
        return dist_by_cell_pair_dict

    # Step 2.2 Summarzie dist_list as mean/median value by chr key
    ########################################################################################       
    else:
        print ('Return summarized results.')
        for _key in _out_dist_dict:
            print(_key, len(_out_dist_dict[_key]))
        #return _out_dist_dict
        # summarize
        _summary_dict = {}
        # add nan for non-relevant or empty chr pairs
        for _key, _dists_list in _out_dist_dict.items():
            
            if len(_dists_list) >= 100: # discard if not enough cell statistics
                # summarize
                # no prior normalization if calculate contact prob
                if contact_prob:
                    # get nan positions
                    _nan_index = np.isnan(np.array(_dists_list))
                    # th value comparision and restore nan
                    _dists_contact_list=np.array(_dists_list)<=_contact_th
                    # convert array as float type to enable nan re-assignment through indexing; 
                    # otherwise all nan will be treated as True for bool
                    _dists_contact_list=_dists_contact_list*1.0
                    _dists_contact_list[_nan_index==1]=np.nan
                    _summary_result = getattr(np, 'nanmean')(_dists_contact_list, axis=axis)
                    # keep result if non-nan N > data number th
                    _valid_index = np.sum(~np.isnan(_dists_contact_list),axis=0)>=data_num_th
                    _summary_result[_valid_index==0]=np.nan
                    _summary_dict[_key] = _summary_result

                else:
                    _summary_result =getattr(np, function)(_dists_list, axis=axis)
                    # keep result if non-nan N > data number th
                    _valid_index = np.sum(~np.isnan(_dists_list),axis=0)>=data_num_th
                    _summary_result[_valid_index==0]=np.nan
                    _summary_dict[_key] = _summary_result
            else:
                if isinstance(_key, str): # cis or trans
                    _chr = _key.split('_')[-1] 
                    _summary_dict[_key]= np.nan * np.ones([all_chr_sizes[_chr], all_chr_sizes[_chr]])
                else:
                    _chr1, _chr2 = _key
                    _summary_dict[_key]= np.nan * np.ones([all_chr_sizes[_chr1], all_chr_sizes[_chr2]])
    
    return _summary_dict



# call previous function to calculate all pair-wise chromosomal distance
# calculate for a subset of loci or all loci depending on if the total_codebook has all or a subset of loci
def Chr2ZxysList_2_summaryDict_V2 ( chr_2_zxys_list, total_codebook,
                                    function='nanmedian', axis=0,data_num_th=50,
                                    _center_dist_th = 3000,normalize_by_center=False,
                                    _contact_th=500, contact_prob=False,
                                    parallel=True, num_threads=default_num_threads,
                                    interaction_type = 'trans',
                                    return_raw = False,  
                                    verbose=False):
    """Function to batch process chr_2_zxys_list into summary_dictionary"""


    Chr2ZxysList_2_summaryDist_trans_function = Chr2ZxysList_2_summaryDist_by_key_V2

    if verbose:
        print(f"-- preparing chr_2_zxys from {len(chr_2_zxys_list)} cells", end=' ')
        _start_prepare = time.time()
    _summary_args = []
    # prepare args
    _all_chrs = np.unique(total_codebook['chr'].values)
    #sorted(_all_chrs, key=lambda _c:sort_mouse_chr(_c))
    for _chr1, _chr2 in combinations_with_replacement(_all_chrs, 2):
        if _chr1 != _chr2:
            _sel_chr_2_zxys = [
                {_chr1: _d.get(_chr1, None),
                 _chr2: _d.get(_chr2, None)} for _d in chr_2_zxys_list
            ]
        else:
            _sel_chr_2_zxys = [
                {_chr1: _d.get(_chr1, None)} for _d in chr_2_zxys_list
            ]
        _summary_args.append(
            (_sel_chr_2_zxys, _chr1, _chr2, total_codebook, function, axis, data_num_th,
            _center_dist_th, normalize_by_center, _contact_th, contact_prob,interaction_type, return_raw, verbose)
        )
    if verbose:
        print(f"in {time.time()-_start_prepare:.3f}s.")
    # process
    _start_time = time.time()
    if parallel:
        if verbose:
            print(f"-- summarize {len(_summary_args)} inter-chr distances with {num_threads} threads", end=' ')
        with mp.Pool(num_threads) as _summary_pool:
            all_summary_dicts = _summary_pool.starmap(
                Chr2ZxysList_2_summaryDist_trans_function, 
                _summary_args, chunksize=1)
            _summary_pool.close()
            _summary_pool.join()
            _summary_pool.terminate()
        if verbose:
            print(f"in {time.time()-_start_time:.3f}s.")
    else:
        if verbose:
            print(f"-- summarize {len(_summary_args)} inter-chr distances sequentially", end=' ')
        all_summary_dicts = [Chr2ZxysList_2_summaryDist_trans_function(*_args) for _args in _summary_args]
        if verbose:
            print(f"in {time.time()-_start_time:.3f}s.")
    # summarize into one dict
    _summary_dict = {}
    for _dict in all_summary_dicts:
        _summary_dict.update(_dict)
    return _summary_dict



# A simple function to get dist for a specific pari
# Use this only for one or a few pairs
# Alternatively 
def simple_dist_between_loci_pair_from_chrZxys(chr_2_zxys_list, 
                                        _chr_r1 = ['1',0], 
                                        _chr_r2 =['1',0],  # chr as str and chr_order as int
                                        _center_dist_th=5000000,
                                        report_center=True,
                                        #verbose=False
                                        ):
    
    from scipy.spatial.distance import euclidean
    import numpy as np
    
    if not ( isinstance (_chr_r1, list) and isinstance (_chr_r2, list)):
        print ('Error in loci. Check args')
        return None
    else:
        if not (len(_chr_r1)==2 and len(_chr_r2)==2):
            print ('Error in loci. Check args')
            return None
        else:
            pass
        
    print (f'Start analysis with _center_dist_th as {_center_dist_th} nm.')
            
    # init dict and list to store results
    _out_dist_dict = {}
    _out_center_dists = []
    _out_loci_dists = []
    
    _c1, _chr_order1 = _chr_r1
    _c2, _chr_order2 = _chr_r2

    for _chr_2_zxys in chr_2_zxys_list:
        # skip if not all info exists
        if _c1 not in _chr_2_zxys or _c2 not in _chr_2_zxys or _chr_2_zxys[_c1] is None or _chr_2_zxys[_c2] is None:
            continue
        else:
            # if from the same chr label, calcluate cis-chr dist
            if _c1 == _c2:
                for _zxys1 in _chr_2_zxys[_c1]:
                    _loci_zxy1 = _zxys1[_chr_order1]
                    _loci_zxy2 = _zxys1[_chr_order2]
                    # append dist for both center (as 0.0) and loci pair resepctively
                    _out_center_dists.append(float(0))
                    if np.sum(np.isnan(_loci_zxy1))==0 and np.sum(np.isnan(_loci_zxy2))==0:
                        _loci_dist= euclidean(_loci_zxy1,_loci_zxy2)
                    else:
                        _loci_dist=np.nan
                    _out_loci_dists.append(_loci_dist)
                      
            # if not from the same chr label, calcluate trans-chr dist for both chr copies
            else:
                for _zxys1 in _chr_2_zxys[_c1]:
                    _center_zxys1 = get_chroZxys_mean_center(_zxys1)
                    _loci_zxy1 = _zxys1[_chr_order1]
                    for _zxys2 in _chr_2_zxys[_c2]:
                        _center_zxys2 = get_chroZxys_mean_center(_zxys2)
                        _loci_zxy2 = _zxys2[_chr_order2]
                        # append dist for both center and loci pair resepctively
                        _center_dist = euclidean(_center_zxys1,_center_zxys2)
                        # this restrict analysis to neighboring chromosome pair within given distance
                        if _center_dist<= _center_dist_th:
                            _out_center_dists.append(_center_dist)
                            if np.sum(np.isnan(_loci_zxy1))==0 and np.sum(np.isnan(_loci_zxy2))==0:
                                _loci_dist= euclidean(_loci_zxy1,_loci_zxy2)
                            else:
                                _loci_dist=np.nan
                            _out_loci_dists.append(_loci_dist)
                        else:
                            continue
                        
    _out_dist_dict[f'({_c1}_{_chr_order1}-{_c2}_{_chr_order2})_dists']={}
    _out_dist_dict[f'({_c1}_{_chr_order1}-{_c2}_{_chr_order2})_dists']['loci'] = np.array(_out_loci_dists)
    
    if report_center:
        _out_dist_dict[f'({_c1}_{_chr_order1}-{_c2}_{_chr_order2})_dists']['chr_center']=np.array(_out_center_dists)
        return _out_dist_dict
    else:
        return  _out_dist_dict







