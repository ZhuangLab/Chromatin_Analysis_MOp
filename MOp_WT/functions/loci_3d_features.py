import numpy as np
import multiprocessing as mp
import time
import pandas as pd
# functions
from itertools import combinations_with_replacement, permutations
from scipy.spatial.distance import pdist, squareform, cdist
from tqdm import tqdm
# shared parameters

default_num_threads = 12





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
def Chr2ZxysList_2_summaryDist_by_key_for_trans_neigh_chr(chr_2_zxys_list, _c1, _c2, codebook_df,
                                 function='nanmedian', axis=0, data_num_th=50, _center_dist_th = 5,normalize_by_center=False,
                                 _contact_th=0.6, contact_prob=False,
                                 verbose=False):
    
    from scipy.spatial.distance import euclidean

    print(f'-- Start analyzing neighboring chromosome pair within {_center_dist_th}nm.')
    
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
                # dual loop for each allel
                for _zxys1 in _chr_2_zxys[_c1]:
                    _center_zxys1 = get_chroZxys_mean_center(_zxys1)
                    for _zxys2 in _chr_2_zxys[_c2]:
                        _center_zxys2 = get_chroZxys_mean_center(_zxys2)
                        # only analyze neighboring chromosome pair within given distance
                        if euclidean(_center_zxys1,_center_zxys2)<= _center_dist_th:
                            _chr_pair_cdists = cdist(_zxys1, _zxys2)
                            if normalize_by_center and not contact_prob:
                                _chr_pair_cdists=_chr_pair_cdists/euclidean(_center_zxys1,_center_zxys2)
                            _out_dist_dict[(_c1,_c2)].append(_chr_pair_cdists)
                        else:
                            continue
            # if from the same chr label, calculate both cis and trans
            else:
                # cis
                _out_dist_dict[f"cis_{_c1}"].extend([squareform(pdist(_zxys)) for _zxys in _chr_2_zxys[_c1]])
                # trans
                if len(_chr_2_zxys[_c1]) > 1:
                    # loop through permutations
                    for _i1, _i2 in permutations(np.arange(len(_chr_2_zxys[_c1])), 2):
                        _center_zxys1 = get_chroZxys_mean_center(_chr_2_zxys[_c1][_i1])
                        _center_zxys2 = get_chroZxys_mean_center(_chr_2_zxys[_c1][_i2])
                        if euclidean(_center_zxys1,_center_zxys2)<= _center_dist_th:
                            _chr_pair_cdists =  cdist(_chr_2_zxys[_c1][_i1], _chr_2_zxys[_c1][_i2])
                            if normalize_by_center and not contact_prob:
                                _chr_pair_cdists=_chr_pair_cdists/euclidean(_center_zxys1,_center_zxys2)
                            _out_dist_dict[f"trans_{_c1}"].append(_chr_pair_cdists)
                        else:
                            continue

    for _key in _out_dist_dict:
        print(_key, len(_out_dist_dict[_key]))
    #return _out_dist_dict
    # summarize
    _summary_dict = {}
    all_chrs = [str(_chr) for _chr in np.unique(codebook_df['chr'])]
    all_chr_sizes = {_chr:np.sum(codebook_df['chr']==_chr) for _chr in all_chrs}
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
def Chr2ZxysList_2_summaryDict_for_trans_neigh_chr(
    chr_2_zxys_list, total_codebook,
    function='nanmedian', axis=0,data_num_th=50,
    _center_dist_th = 5,normalize_by_center=True,
     _contact_th=0.6, contact_prob=False,
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
            (_sel_chr_2_zxys, _chr1, _chr2, total_codebook, function, axis, data_num_th,
            _center_dist_th, normalize_by_center, _contact_th, contact_prob,verbose)
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
                Chr2ZxysList_2_summaryDist_by_key_for_trans_neigh_chr, 
                _summary_args, chunksize=1)
            _summary_pool.close()
            _summary_pool.join()
            _summary_pool.terminate()
        if verbose:
            print(f"in {time.time()-_start_time:.3f}s.")
    else:
        if verbose:
            print(f"-- summarize {len(_summary_args)} inter-chr distances sequentially", end=' ')
        all_summary_dicts = [Chr2ZxysList_2_summaryDist_by_key_for_trans_neigh_chr(*_args) for _args in _summary_args]
        if verbose:
            print(f"in {time.time()-_start_time:.3f}s.")
    # summarize into one dict
    _summary_dict = {}
    for _dict in all_summary_dicts:
        _summary_dict.update(_dict)
    return _summary_dict




# code from ImageAnalysis3.structure_tools.distance import Chr2ZxysList_2_summaryDist_by_key
def Chr2ZxysList_2_Dist_by_key(chr_2_zxys_list, _c1, _c2,):
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
    return _out_dist_dict



# modified slighly from Pu's code to require more stringent data number to analyzed
def Chr2ZxysList_2_summaryDist_by_key_v2(chr_2_zxys_list, _c1, _c2, codebook_df,
                                 function='nanmedian', axis=0, data_num_th=50,contact_prob=False,_contact_th=500,
                                 verbose=False, summarize_result=True):
    
    _out_dist_dict = Chr2ZxysList_2_Dist_by_key(chr_2_zxys_list, _c1, _c2,)
    # summarize
    _summary_dict = {}
    all_chrs = [str(_chr) for _chr in np.unique(codebook_df['chr'])]
    all_chr_sizes = {_chr:np.sum(codebook_df['chr']==_chr) for _chr in all_chrs}
    for _key, _dists_list in _out_dist_dict.items():
        
        if len(_dists_list) >= 100:
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
def Chr2ZxysList_2_summaryDict_v2(
    chr_2_zxys_list, total_codebook,
    function='nanmedian', axis=0, 
     data_num_th=50,contact_prob=False,_contact_th=500,
    parallel=True, num_threads=default_num_threads,
    verbose=False):
    """Function to batch process chr_2_zxys_list into summary_dictionary"""
    if verbose:
        print(f"-- preparing chr_2_zxys from {len(chr_2_zxys_list)} cells", end=' ')
        _start_prepare = time.time()
    _summary_args = []
    # prepare args
    _all_chrs = np.unique(total_codebook['chr'].values)
    #sorted(_all_chrs, key=lambda _c:sort_chr(_c))
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
            (_sel_chr_2_zxys, _chr1, _chr2, total_codebook, function, axis, data_num_th,contact_prob,_contact_th,verbose,True) # always True to summarize result
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
                Chr2ZxysList_2_summaryDist_by_key_v2, 
                _summary_args, chunksize=1)
            _summary_pool.close()
            _summary_pool.join()
            _summary_pool.terminate()
        if verbose:
            print(f"in {time.time()-_start_time:.3f}s.")
    else:
        from tqdm import tqdm
        if verbose:
            print(f"-- summarize {len(_summary_args)} inter-chr distances sequentially", end=' ')
        all_summary_dicts = [Chr2ZxysList_2_summaryDist_by_key_v2(*_args) for _args in tqdm(_summary_args)]
        if verbose:
            print(f"in {time.time()-_start_time:.3f}s.")
    # summarize into one dict
    _summary_dict = {}
    for _dict in all_summary_dicts:
        _summary_dict.update(_dict)
    return _summary_dict




# functions to get contact count summary
def get_genomic_distance_mtx (df_refgen, chosen_chrom):
    import sklearn
    
    df_refgen_chr = df_refgen[df_refgen['chr']==chosen_chrom]
    df_refgen_chr = df_refgen_chr.sort_values(by='chr_order')
    chr_starts = np.array(df_refgen_chr['start'].tolist())

    def gene_dist_calc(x,y):
        return (y-x)
    gene_dist_mtx = sklearn.metrics.pairwise_distances(chr_starts.reshape(-1,1), metric=gene_dist_calc)
            
    return gene_dist_mtx
    

def generate_gene_dist_bins (step_size = 100, # step size by 2**
                             max_dist = 200000000,
                            ):
    # refer to 3mC paper method
    #gene_dist_bins = [basic_res * (2*(step_size*bin_idx)) for bin_idx in range(0, )]
    gene_dist_list = np.arange(0,max_dist, max_dist/step_size)
    gene_dist_bins = [(gene_dist_list[i], gene_dist_list[i+1]) for i in range(len(gene_dist_list)-1)]
    
    return gene_dist_bins
     
# summarize the count grouped by gene dists
def Chr2Zxys_list_2_cis_contact_summary(chr_2_zxys_list, df_refgen, gene_dist_bins, gene_dist_mtx_dict=None, contact_th =0.6):
    

    all_chrs = np.unique(df_refgen['chr'].values)
    all_chr_sizes = {_chr:np.sum(df_refgen['chr']==_chr) for _chr in all_chrs}
    # get ref gene dist mtx if not given as input
    if isinstance(gene_dist_mtx_dict, type(None)):
        gene_dist_mtx_dict = {}
        for _chr in all_chrs:
            gene_dist_mtx_dict[_chr] = get_genomic_distance_mtx (df_refgen, _chr)
    
    # loop for each cell in the list
    output_dict_list = []
    for chr_2_zxys in chr_2_zxys_list:
        # summarize for each chromosome
        _contact_count_dict = {}
        for _chr in all_chrs:
            if _chr in chr_2_zxys.keys():
                _contact_count_dict[_chr]={}
                # calcuate zxy mtx
                _chr_distmap = np.array([squareform(pdist(_zxys)) for _zxys in chr_2_zxys[_chr]]) <= contact_th
                # retrieve mtx for genomic dists
                gene_dist_mtx_chr = gene_dist_mtx_dict[_chr]
                # get counts for each specified bins; sum for all haploid copy
                for gene_dist_bin in gene_dist_bins:
                    _bin_start = gene_dist_bin[0]
                    _bin_end = gene_dist_bin[1]
                    contact_count = 0
                    for _ichr_distmap in _chr_distmap:
                        contact_count += np.sum(_ichr_distmap & 
                                                (gene_dist_mtx_chr >_bin_start) & # >start to remove dist 0
                                                (gene_dist_mtx_chr <= _bin_end))
                    _contact_count_dict[_chr][gene_dist_bin] = contact_count

                # get total counts
                total_counts = 0
                for _ichr_distmap in _chr_distmap:
                    total_counts += np.sum(_ichr_distmap)
                _contact_count_dict[_chr]['total'] = total_counts
            else:
                # skip adding info for the missing chrom
                continue
        
        output_dict_list.append(_contact_count_dict)

    return output_dict_list



def Chr2Zxys_list_2_cis_contact_summary_batch_byclass (chr_2_zxys_list_dict, df_refgen, gene_dist_bins, gene_dist_mtx_dict=None, contact_th =0.6, num_threads = 16):

    #prepare mp args
    mp_chr_2_zxys_list = []
    for _class, chr_2_zxys_list in chr_2_zxys_list_dict.items():
        mp_chr_2_zxys_list.append(chr_2_zxys_list)

    mp_df_regen_list = [df_refgen] * len(mp_chr_2_zxys_list)
    mp_gene_dist_bins_list = [gene_dist_bins] * len(mp_chr_2_zxys_list)
    mp_gene_dist_mtx_dict_list = [gene_dist_mtx_dict] * len(mp_chr_2_zxys_list)
    mp_contact_th_list = [contact_th] * len(mp_chr_2_zxys_list)

    # process
    _start_time = time.time()
    print(f"-- summarize chromosomal contacts for {len(mp_chr_2_zxys_list)} cell types with {num_threads} threads", end=' ')
    with mp.Pool(num_threads) as _summary_pool:
        all_summary_lists = _summary_pool.starmap(Chr2Zxys_list_2_cis_contact_summary, zip(mp_chr_2_zxys_list, 
                                                                                            mp_df_regen_list, 
                                                                                            mp_gene_dist_bins_list, 
                                                                                            mp_gene_dist_mtx_dict_list, 
                                                                                            mp_contact_th_list),
                                                                                            chunksize=1)
        _summary_pool.close()
        _summary_pool.join()
        _summary_pool.terminate()
    print(f"in {time.time()-_start_time:.3f}s.")

    _summary_dict = {}
    if len(list(chr_2_zxys_list_dict.keys())) == len(all_summary_lists):
        for _class, summary_list in zip(list(chr_2_zxys_list_dict.keys()),all_summary_lists):
            _summary_dict[_class] = summary_list
    else:
        print ('Number of input cell types do not match.')
        _summary_dict = None

    return _summary_dict



