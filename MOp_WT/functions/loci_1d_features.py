# Import shared required packages
import numpy as np
import pandas as pd
import scanpy as sc
import anndata




def im_loci_dataframe_from_gene_dataframe (marker_genes_df:pd.core.frame.DataFrame, sel_cols = None):
    
    """Convert a processed gene dataframe to a loci dataframe"""
    
    # check if gene dataframe has been cleaned (with only keeping genes with a found imaged loci)
    marker_genes_df_new = marker_genes_df[marker_genes_df['Imaged_loci'].str.contains('chr')]
    if len(marker_genes_df) != len(marker_genes_df_new):
        print ('Gene DataFrame has elements which do not have adjacent imaged loci.')
        print ('Remove these element for generating the Loci DataFrame.')
    
    marker_genes_df_new['Marker_gene'] = marker_genes_df_new.index.tolist()
    im_loci_df = marker_genes_df_new.set_index(['Imaged_loci']) 
    if isinstance(sel_cols, type(None)):
        sel_cols = im_loci_df.columns
        print ('Retrieve all columns from Gene DataFrame')
    
    im_loci_df = im_loci_df[sel_cols]

    
    return im_loci_df





# Use for sorting chromosome by number (adapted from PZ)
def mouse_chr_as_num (_chr):
    try:
        out_key = int(_chr)
    except:
        if _chr == 'X':
            out_key = 20
        elif _chr == 'Y':
            out_key = 21
    return out_key


def human_chr_as_num (_chr):
    try:
        out_key = int(_chr)
    except:
        if _chr == 'X':
            out_key = 23
        elif _chr == 'Y':
            out_key = 24
    return out_key



def sort_loci_df_by_chr_order (im_loci_df: pd.core.frame.DataFrame, genome_type='mouse'):
    """simply sort loci df by their chromosome and chr order"""
    im_loci_df_new = im_loci_df.copy(deep=True)
    # sort im_loci_df for matrix generation if not sorted
    if genome_type == 'mouse':
        im_loci_df_new['chr_as_num'] = im_loci_df['chr'].map(mouse_chr_as_num)
    elif genome_type == 'human':
        im_loci_df_new['chr_as_num'] = im_loci_df['chr'].map(human_chr_as_num)
    else:
        print ('Unknown genome type, Exit with no sort')
        return im_loci_df
    im_loci_df_new = im_loci_df_new.sort_values(by = ['chr_as_num', 'chr_order'], ignore_index=False)
    im_loci_df_new = im_loci_df_new.drop(columns=['chr_as_num'])

    return im_loci_df_new



# Use for sorting chromosome by number (from PZ)
def sort_mouse_chr(_chr):
    try:
        out_key = int(_chr)
    except:
        if _chr == 'X':
            out_key = 20
        elif _chr == 'Y':
            out_key = 21
    return out_key




# Get chr and chr order info, etc from the codebook
def codebook_chr_order_for_loci_dataframe (im_loci_df: pd.core.frame.DataFrame, 
                                           codebook_df:pd.core.frame.DataFrame, 
                                           sel_cols =['chr','chr_order','id'], 
                                           sort_df = True,
                                           sort_by_chr=True):
    
    """Add chr and chr-order to the im loci dataframe"""
    
    im_loci_df_new = im_loci_df.copy()

    # slice codebook with the index from loci df
    im_loci_list = im_loci_df.index.tolist()
    sel_codebook_df = codebook_df.loc[im_loci_list]
    
    # append new cols accordingly
    for _col in sel_cols:
        #_new_col = _col.capitalize()
        _new_col = _col
        im_loci_df_new[_new_col] = sel_codebook_df[_col]
    
    # if sort dataframe, whether to return a sorted df or not 
    if sort_df:
        # sort by chr and chr order
        if sort_by_chr:
            im_loci_df_new = sort_loci_df_by_chr_order (im_loci_df_new)
        # sort by library id
        else:
            im_loci_df_new = im_loci_df_new.sort_values(by = ['id'], ignore_index=False)
        
    return im_loci_df_new



# Function to calculate pairwise distance for the loci from the processed DataFrame
def pairwise_distance_loci_dataframe (im_loci_df, 
                                      class_2_chrZxysList,
                                      add_control = False,
                                      function='nanmedian',
                                      axis=0,
                                      data_num_th=50,
                                      _center_dist_th = 5000000,
                                      normalize_by_center=False,
                                      _contact_th=500, 
                                      contact_prob=False,
                                      parallel=True, 
                                      num_threads=12,
                                      interaction_type='trans',
                                      return_raw=True,
                                      plot_cis = True,
                                      verbose=False,
                                      ):
    
    """Function to get pairwise distance for loci pair of interest"""
  
    result_dict_by_dir_by_group = {}
    print ('Summarizing the pairwise distance for the Imaged Loci Dataframe.')


    # sort im_loci_df for matrix generation if not sorted
    im_loci_df = sort_loci_df_by_chr_order (im_loci_df)

    
    cellgroups = im_loci_df['Compared_groups'][0].split('; ')
    directions=np.flip(np.unique(im_loci_df['Expression_change'].tolist()))
    if not add_control:
        directions = [_dir for _dir in directions if _dir != 'control']
        
    from distance_atc import Chr2ZxysList_2_summaryDict_V2


    for _group in cellgroups:
        print (f'Summarizing for loci in {_group}.')
        chr_2_zxys_list = class_2_chrZxysList[_group]

        result_dict_by_dir_by_group[_group]={}
        for _dir in directions:
            print (f'Summarizing the pairwise distance for {_dir}-related loci.')
            sel_im_loci_df = im_loci_df[im_loci_df['Expression_change']==_dir]
            sel_chr_loci_info_df = sel_im_loci_df[['chr','chr_order','id']]

            result = Chr2ZxysList_2_summaryDict_V2 (chr_2_zxys_list, 
                                                    sel_chr_loci_info_df,
                                                    function=function, 
                                                    axis=axis,
                                                    data_num_th=data_num_th,
                                                    _center_dist_th = _center_dist_th,
                                                    normalize_by_center=normalize_by_center,
                                                    _contact_th=_contact_th, 
                                                    contact_prob=contact_prob,
                                                    parallel=parallel, 
                                                    num_threads=num_threads,
                                                    interaction_type=interaction_type,
                                                    return_raw=return_raw,
                                                    verbose=verbose)
        
            result_dict_by_dir_by_group[_group][_dir]= result
        
    if return_raw:
        print ('Return raw pairwise distances by (chr,chr-order) keys.')
        return result_dict_by_dir_by_group
    
    else:
        from distance_atc import assemble_ChrDistDict_2_Matrix_V2
        print ('Return reduced pairwise distances by (chr) keys.')
        print('Calculating reduced matrix.')

        if interaction_type=='cis':
            plot_cis=True
            print("Set plot_cis as True as the distance calculation was done with cis-only.")

        dist_map_dict_by_dir_by_group = {}
        for _group in cellgroups:
            dist_map_dict_by_dir_by_group[_group]={}
            for _dir in directions:
                # use sel_chr_loci_info_df as codebook; the chr order will be renamed by argsort convert_chr_2_chr_orders so match the matrix order
                sel_im_loci_df = im_loci_df[im_loci_df['Expression_change']==_dir]
                sel_chr_loci_info_df = sel_im_loci_df[['chr','chr_order','id']]
                distmap, chr_edges, chr_names = assemble_ChrDistDict_2_Matrix_V2(
                                                                            result_dict_by_dir_by_group[_group][_dir], 
                                                                            sel_chr_loci_info_df, 
                                                                            sel_codebook=None, 
                                                                            use_cis=plot_cis, sort_by_region=False, # here sort_by_region=False means sort by chr?
                                                                        )

                dist_map_dict_by_dir_by_group[_group][_dir] = {'matrix':distmap, 'chr_edges': chr_edges, 'chr_names':chr_names}
                
        
        return dist_map_dict_by_dir_by_group
        
        
##############################################################################################################################
# Compartment analysis 
def sc_compartment_ratio_by_loci_key_old (sc_compartment_dict_by_group, cell_group_list, loci_key, report_type = 'median'):
    
    """Get the corresponding single-cell AB comparment density ratio by chr, chr_order for a list of cell groups of interest"""
    _chr_key = loci_key[0]
    _chr_order = loci_key[1]
    
    # append result from single cell for each cell group as column
    _loci_group_dict = {_k: [] for _k in  cell_group_list}
     # get the AB ratio for all cell group and append together
    for _group in cell_group_list:
        if _group in sc_compartment_dict_by_group.keys():
            # store ratio for each sc
            loci_ratio_list = []
            group_ABs = sc_compartment_dict_by_group[_group]
            for _cell_ABs in group_ABs:
                # if chr_key exists and there is a valid nd.array 
                if _chr_key in _cell_ABs.keys():
                    if isinstance(_cell_ABs[_chr_key],np.ndarray):
                        for _ichr_ABs in _cell_ABs[_chr_key]:
                            #print(_ichr_ABs[_chr_order])
                            loci_ratio_list.append(_ichr_ABs[_chr_order])
                    else:
                        loci_ratio_list.append(np.nan)
                        loci_ratio_list.append(np.nan)  
                # if not chr_key exists, append nan for both copies
                else:
                    loci_ratio_list.append(np.nan)
                    loci_ratio_list.append(np.nan)
            # get nanmedian of the sc into a cell-type bulk average for each loci    
            if report_type == 'raw':   
                _loci_group_dict [_group]=loci_ratio_list
            elif report_type == 'mean':   
                _loci_group_dict [_group]=[np.nanmean(loci_ratio_list)]
            elif report_type == 'median':   
                _loci_group_dict [_group]=[np.nanmedian(loci_ratio_list)]
                
        else:
            print ('The query cell group not present. Skip.')
            pass
    
    _loci_group_df=pd.DataFrame.from_dict(_loci_group_dict,orient='index')
    _loci_group_df=_loci_group_df.transpose()
    
    return  _loci_group_df




def check_cell_quality_by_spot_number (_cell_ABs, spot_num_th=100):
    """Check the number of decoded spots for each cell zxys dict"""

    loci_nonna_list = []
    for _chr_key in _cell_ABs.keys():
        if isinstance(_cell_ABs[_chr_key],np.ndarray):
            for _ichr_ABs in _cell_ABs[_chr_key]:
                loci_nonna_list.append(np.count_nonzero(~np.isnan(_ichr_ABs)))

    if np.sum(loci_nonna_list)>=spot_num_th:
        return True
    else:
        return False



# Compartment analysis with the compatibility to average ratio for all copies of the same loci
def sc_compartment_ratio_by_loci_key (sc_compartment_dict_by_group, cell_group_list, loci_key, report_type = 'median', average_ratios_in_cell=True, spot_num_th=100):
    
    """Get the corresponding single-cell AB comparment density ratio by chr, chr_order for a list of cell groups of interest"""
    _chr_key = loci_key[0]
    _chr_order = loci_key[1]
    
    # append result from single cell for each cell group as column
    _loci_group_dict = {_k: [] for _k in  cell_group_list}
     # get the AB ratio for all cell group and append together
    for _group in cell_group_list:
        if _group in sc_compartment_dict_by_group.keys():
            # store ratio for each sc
            loci_ratio_list = []
            group_ABs = sc_compartment_dict_by_group[_group]
            for _cell_ABs in group_ABs:
                # check quality first
                quality_pass = check_cell_quality_by_spot_number (_cell_ABs, spot_num_th=spot_num_th)
                
                # sub-list to get all ratios for a cell first
                loci_ratio_cell = []
                # if chr_key exists and there is a valid nd.array 
                if _chr_key in _cell_ABs.keys() and quality_pass:
                    if isinstance(_cell_ABs[_chr_key],np.ndarray):
                        for _ichr_ABs in _cell_ABs[_chr_key]:
                            #print(_ichr_ABs[_chr_order])
                            loci_ratio_cell.append(_ichr_ABs[_chr_order])
                    else: # wrong format
                        loci_ratio_cell.append(np.nan)
                        loci_ratio_cell.append(np.nan)  
                # if not chr_key exists or cell not good quality, append nan for both copies
                else:
                    loci_ratio_cell.append(np.nan)
                    loci_ratio_cell.append(np.nan)
                
                # average the ratios for each cell
                if average_ratios_in_cell:
                    mean_ratio_cell = np.nanmean(np.array(loci_ratio_cell)) # warning raised here: nanmean of nans return nan
                    loci_ratio_list.append(mean_ratio_cell)
                else:
                    loci_ratio_list.extend(loci_ratio_cell)

            # get nanmedian of the sc into a cell-type bulk average for each loci    
            if report_type == 'raw':   
                _loci_group_dict [_group]=loci_ratio_list
            elif report_type == 'mean':   
                _loci_group_dict [_group]=[np.nanmean(loci_ratio_list)]
            elif report_type == 'median':   
                _loci_group_dict [_group]=[np.nanmedian(loci_ratio_list)]
                
        else:
            print ('The query cell group not present. Skip.')
            pass
    
    _loci_group_df=pd.DataFrame.from_dict(_loci_group_dict,orient='index')
    _loci_group_df=_loci_group_df.transpose()
    
    return  _loci_group_df




def check_cell_quality_by_spot_number_vScore (_cell_ABs, spot_num_th=100):
    """Check the number of decoded spots for each cell zxys dict"""

    loci_nonna_list = []
    for _chr_key in _cell_ABs.keys():
        if isinstance(_cell_ABs[_chr_key]['A'],np.ndarray):
            for _ichr_ABs in _cell_ABs[_chr_key]['A']:
                loci_nonna_list.append(np.count_nonzero(~np.isnan(_ichr_ABs)))

    if np.sum(loci_nonna_list)>=spot_num_th:
        return True
    else:
        return False



# Compartment analysis with the compatibility to average ratio for all copies of the same loci
def sc_compartment_score_by_loci_key (sc_compartment_score_dict_by_group, cell_group_list, loci_key, report_type = 'median', average_ratios_in_cell=True, spot_num_th=100):
    
    """Get the corresponding single-cell AB comparment density ratio by chr, chr_order for a list of cell groups of interest"""
    _chr_key = loci_key[0]
    _chr_order = loci_key[1]
    
    # append result from single cell for each cell group as column
    _loci_group_dict_keys = [f"{c}_A" for c in cell_group_list]
    _loci_group_dict_keys.extend([f"{c}_B" for c in cell_group_list])
    _loci_group_dict = {_k: [] for _k in  _loci_group_dict_keys}
     # get the AB ratio for all cell group and append together
    for _group in cell_group_list:
        if _group in sc_compartment_score_dict_by_group.keys():
            # store ratio for each sc
            loci_Ascore_list = []
            loci_Bscore_list = []
            group_ABs = sc_compartment_score_dict_by_group[_group]
            for _cell_ABs in group_ABs:
                # check quality first
                quality_pass = check_cell_quality_by_spot_number_vScore (_cell_ABs, spot_num_th=spot_num_th)
                
                # sub-list to get all ratios for a cell first
                loci_Ascore_cell = []
                loci_Bscore_cell = []
                # if chr_key exists and there is a valid nd.array 
                if _chr_key in _cell_ABs.keys() and quality_pass:
                    if isinstance(_cell_ABs[_chr_key]['A'],np.ndarray):
                        for _ichr_ABs in _cell_ABs[_chr_key]['A']:
                            loci_Ascore_cell.append(_ichr_ABs[_chr_order])
                    else:
                        loci_Ascore_cell.append(np.nan)
                    if isinstance(_cell_ABs[_chr_key]['B'],np.ndarray):
                        for _ichr_ABs in _cell_ABs[_chr_key]['B']:
                            loci_Bscore_cell.append(_ichr_ABs[_chr_order])
                    else:
                        loci_Bscore_cell.append(np.nan)  
                # if not chr_key exists or cell not good quality, append nan for both copies
                else:
                    loci_Ascore_cell.append(np.nan)
                    loci_Bscore_cell.append(np.nan) 
                
                # average the ratios for each cell
                if average_ratios_in_cell:
                    mean_Ascore_cell = np.nanmean(np.array(loci_Ascore_cell)) # warning raised here: nanmean of nans return nan
                    mean_Bscore_cell = np.nanmean(np.array(loci_Bscore_cell))
                    loci_Ascore_list.append(mean_Ascore_cell)
                    loci_Bscore_list.append(mean_Bscore_cell)

                else:
                    loci_Ascore_list.extend(loci_Ascore_cell)
                    loci_Bscore_list.extend(loci_Bscore_cell)

            # get nanmedian of the sc into a cell-type bulk average for each loci    
            if report_type == 'raw':   
                _loci_group_dict [f'{_group}_A']=loci_Ascore_list
                _loci_group_dict [f'{_group}_B']=loci_Bscore_list
            elif report_type == 'mean':   
                _loci_group_dict [f'{_group}_A']=[np.nanmean(loci_Ascore_list)]
                _loci_group_dict [f'{_group}_B']=[np.nanmean(loci_Bscore_list)]
            elif report_type == 'median':   
                _loci_group_dict [f'{_group}_A']=[np.nanmedian(loci_Ascore_list)]
                _loci_group_dict [f'{_group}_B']=[np.nanmedian(loci_Bscore_list)]
                
        else:
            print ('The query cell group not present. Skip.')
            pass
    
    _loci_group_df=pd.DataFrame.from_dict(_loci_group_dict,orient='index')
    _loci_group_df=_loci_group_df.transpose()
    
    return  _loci_group_df


##############################################################################################################################

def sorted_loci_keys_for_loci_dataframe(im_loci_df:pd.core.frame.DataFrame):

    """return the chr-chr_order as tuple"""

    chr_list = im_loci_df['chr'].tolist()
    chr_order_list =im_loci_df['chr_order'].tolist()

    chr_loci_keys = [(_chr, _chr_order) for _chr, _chr_order in zip(chr_list,chr_order_list)]
    
    return chr_loci_keys
   
   
def find_chr_loci_key_from_loci_iloc (im_loci_df:pd.core.frame.DataFrame, loci_iloc_list:list):

    """Return unique chr key (chr, chr_order) using loci ind"""

    chr_loci_keys =sorted_loci_keys_for_loci_dataframe(im_loci_df)
    chr_loci_inds = range(len(im_loci_df))

    output_loci_keys = []
    for _query_iloc in loci_iloc_list:
        _query_key = chr_loci_keys[chr_loci_inds.index(_query_iloc)]
        output_loci_keys.append(_query_key)
    
    return output_loci_keys



def find_chr_loci_iloc_from_loci_keys (im_loci_df:pd.core.frame.DataFrame, loci_key_list:list):

    """Return unique loci ind (for given codebook) using chr key (chr, chr_order)"""

    chr_loci_keys =sorted_loci_keys_for_loci_dataframe(im_loci_df)
    chr_loci_inds = range(len(im_loci_df))

    output_loci_iloc = []
    for _query_key in loci_key_list:
        _query_iloc = chr_loci_inds[chr_loci_keys.index(_query_key)]
        output_loci_iloc.append(_query_iloc)
    
    return output_loci_iloc



def sorted_pair_keys_for_loci_dataframe(im_loci_df:pd.core.frame.DataFrame):

    """Return pairwise chr pair key (chr, chr_order)"""

    chr_list = im_loci_df['chr'].tolist()
    chr_order_list =im_loci_df['chr_order'].tolist()

    chr_keys = [(_chr, _chr_order) for _chr, _chr_order in zip(chr_list,chr_order_list)]

    from itertools import combinations
    print ('Prepare the iloc-to-chr, chr order correspondance info.')
    # the pair key would i, j combinations of all loci
    comb_inds = list(combinations(range(len(im_loci_df)),2))

    chr_pair_keys = [(chr_keys[_comb[0]],chr_keys[_comb[1]]) for _comb in comb_inds]
    
    return chr_pair_keys
   


def find_chr_pair_key_from_pair_iloc (im_loci_df:pd.core.frame.DataFrame, pair_iloc_list:list):

    """Return pairwise chr pair key (chr, chr_order)"""

    chr_pair_keys =sorted_pair_keys_for_loci_dataframe(im_loci_df)
    from itertools import combinations
    chr_pair_inds = list(combinations(range(len(im_loci_df)),2))
    print('Get pair list as chr, chr_order.')
    output_pair_keys = []
    for _query_iloc in pair_iloc_list:
        _query_key = chr_pair_keys[chr_pair_inds.index(_query_iloc)]
        output_pair_keys.append(_query_key)
    
    return output_pair_keys


def find_chr_pair_iloc_from_pair_keys (im_loci_df:pd.core.frame.DataFrame, pair_key_list:list):

    """Return pairwise chr pair key (chr, chr_order)"""

    chr_pair_keys =sorted_pair_keys_for_loci_dataframe(im_loci_df)
    from itertools import combinations
    chr_pair_inds = list(combinations(range(len(im_loci_df)),2))
    print('Get pair list as dataframe iloc.')
    output_pair_iloc = []
    for _query_key in pair_key_list:
        _query_iloc = chr_pair_inds[chr_pair_keys.index(_query_key)]
        output_pair_iloc.append(_query_iloc)
    
    return output_pair_iloc


def gene_activity_by_pair_iloc(loci_pair_iloc:tuple, 
                              im_loci_df:pd.core.frame.DataFrame, 
                              pair_score_type = 'mean',
                              activity_col='Activity_score_mean_genes_100kb_tss'):
    
    """Return a merged score from the corresponding score column for a pair of loci specified by their iloc in dataframe"""
    
    loci_pair_iloc= list(loci_pair_iloc)
    #sel_loci_pair_df = im_loci_df.iloc[list(loci_pair_iloc)]
    activity_arr = np.array(im_loci_df[activity_col].tolist())

    if pair_score_type == 'mean':
        #_pair_score = np.nanmean(sel_loci_pair_df[activity_col])
        _pair_score = np.nanmean(activity_arr[loci_pair_iloc])
    elif pair_score_type == 'sum':
        #_pair_score = np.nansum(sel_loci_pair_df[activity_col])
        _pair_score = np.nansum(activity_arr[loci_pair_iloc])
    print(loci_pair_iloc, _pair_score)
        
    return _pair_score
    

def gene_activity_by_pair_iloc_fast(loci_pair_iloc:tuple, 
                              activity_arr:np.ndarray, 
                              pair_score_type = 'mean',
                              activity_col='Activity_score_mean_genes_100kb_tss'):
    
    """Return a merged score from the corresponding score column for a pair of loci specified by their iloc in dataframe"""
    
    loci_pair_iloc= list(loci_pair_iloc)

    if pair_score_type == 'mean':
        #_pair_score = np.nanmean(sel_loci_pair_df[activity_col])
        _pair_score = np.nanmean(activity_arr[loci_pair_iloc])
    elif pair_score_type == 'sum':
        #_pair_score = np.nansum(sel_loci_pair_df[activity_col])
        _pair_score = np.nansum(activity_arr[loci_pair_iloc])
    print(loci_pair_iloc, _pair_score)
        
    return _pair_score


def gene_activity_by_pair_iloc_list (loci_pair_iloc_list:list, 
                              im_loci_df:pd.core.frame.DataFrame, 
                              pair_score_type = 'mean',
                              activity_col='Activity_score_mean_genes_100kb_tss',
                              fast_mode=True,
                              num_threads =12):
    
    """For a list, call function above"""
    from tqdm import tqdm

    print("Caculate the gene activity for the input loci pair list.")
    
    score_dict_by_pair_iloc = {}
    output_dict_keys = []

    print("Extract all pair loci.")
    
    # use one list as input for faster computation
    if fast_mode:
        activity_arr = np.array(im_loci_df[activity_col].tolist())
        _mp_args = []
        for loci_pair_iloc in tqdm(loci_pair_iloc_list):
            _mp_args.append((loci_pair_iloc, activity_arr,pair_score_type,activity_col))
            output_dict_keys.append(loci_pair_iloc)

    else:
        _mp_args = []
        for loci_pair_iloc in tqdm(loci_pair_iloc_list):
            _mp_args.append((loci_pair_iloc, im_loci_df,pair_score_type,activity_col))
            output_dict_keys.append(loci_pair_iloc)

    import time
    _start_time = time.time()
    import multiprocessing as mp
    print ('Multiprocessing for combining gene activity for all pairs:')
    with mp.Pool(num_threads) as _mp_pool:
        if fast_mode:
            _mp_results = _mp_pool.starmap(gene_activity_by_pair_iloc_fast, _mp_args, chunksize=1)
        else:
            _mp_results = _mp_pool.starmap(gene_activity_by_pair_iloc, _mp_args, chunksize=1)
        _mp_pool.close()
        _mp_pool.join()
        _mp_pool.terminate()
    print (f"Complete in {time.time()-_start_time:.3f}s.")

    for _key, _result in zip(output_dict_keys, _mp_results):
        score_dict_by_pair_iloc[_key] = _result

    return score_dict_by_pair_iloc

