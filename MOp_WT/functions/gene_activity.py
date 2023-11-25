# Import shared required packages
import numpy as np
import pandas as pd
import scanpy as sc
import anndata



from gene_selection import check_adata_rank_genes_groups


def gene_activity_score (sel_genes:list, 
                            ref_score_list: list, 
                            ref_gene_list:list, 
                            ref_norm_list = [],
                            report_type ='mean'):

    """Calculate the zscore activity (from AnnData) of a list of genes (for annotated groups)"""

    if len(ref_gene_list) != len(ref_score_list):
        print ('Errors in reference, check reference or adata.')
        return None

    sel_genes = np.intersect1d(sel_genes, ref_gene_list)
    #print ('Ignore missing genes.')
    sel_genes_scores = [ref_score_list[ref_gene_list.index(_gene)] for _gene in sel_genes]

    # set norm_factors as 1 for now; which may be implemented with diff gene length, etc later
    if len(ref_norm_list) == 0:
        norm_factors = np.ones(len(sel_genes_scores))

    norm_sel_genes_scores = np.array(sel_genes_scores) * norm_factors

    if report_type == 'mean':
        result = np.nanmean(norm_sel_genes_scores)
    elif report_type == 'sum':
        result = np.nansum(norm_sel_genes_scores)
        
    return result
    

    
def gene_activity_score_gene_dataframe (marker_genes_df:pd.core.frame.DataFrame, 
                                        adata: anndata._core.anndata.AnnData, 
                                        adjacent_gene_col = 'Adjacent_genes_500kb_tss',
                                        stat_method = 'wilcoxon',
                                        report_type = 'mean',
                                        parallel=False, 
                                        num_threads = 8):
    
    """Call gene_activity_score for batch calculation for gene dataframe"""
    
    marker_genes_df_new = marker_genes_df.copy()
    
    # get gene names and scores reference from adata
    check_adata_rank_genes_groups (marker_genes_df, adata,stat_method = stat_method)
    ref_score_list = [_s[0] for _s in adata.uns['rank_genes_groups']['scores']]
    ref_gene_list = [_g[0] for _g in adata.uns['rank_genes_groups']['names']]
    
    sel_genes_list = marker_genes_df[adjacent_gene_col].tolist()
    
    # prepare arg list for processing
    _mp_args = []
    
    # empty normalization list as placeholder for the gene_activity_score function
    ref_norm_list = []

    for _sel_genes in sel_genes_list:
        _sel_genes = _sel_genes.split('; ')
        _mp_args.append((_sel_genes, ref_score_list, ref_gene_list, ref_norm_list, report_type))
    
    import time
    _start_time = time.time()
    if parallel:
        import multiprocessing as mp
        print ('Multiprocessing for gene acitivity calculation:')
        with mp.Pool(num_threads) as _mp_pool:
            _mp_results = _mp_pool.starmap(gene_activity_score, _mp_args, chunksize=1)
            _mp_pool.close()
            _mp_pool.join()
            _mp_pool.terminate()
        print (f"Complete in {time.time()-_start_time:.3f}s.")
    # looping might be faster for shorter list
    else:
        print ('Looping for gene acitivity calculation:')
        _mp_results = [gene_activity_score (*_args) for _args in _mp_args]
        print (f"Complete in {time.time()-_start_time:.3f}s.")
        
        
    activity_results = _mp_results
    key_added = adjacent_gene_col.replace('Adjacent', f'Activity_score_{report_type}')
    marker_genes_df_new[key_added] = activity_results
    
    return marker_genes_df_new



# cells are not reduced 
def gene_activity_raw_groups (sel_genes:list, 
                        sel_adata: anndata._core.anndata.AnnData, # adata.var only containing the gene of interest
                        cell_groups,
                        groupby,
                        ref_norm_list = [],
                        report_type ='sum'):

    """Get the raw of the gene counts (from AnnData) of a list of genes (for annotated groups);
    Summarize by cell instead of gene compared to the function below [gene_activity_average_groups]"""
    
    ref_gene_list = sel_adata.var.index.tolist()
    sel_genes = np.intersect1d(sel_genes, ref_gene_list)

    # set norm_factors as 1 for now; which may be implemented with diff gene length, etc later
    if len(ref_norm_list) == 0:
        norm_factors = np.ones(len(sel_genes))

    result_groups = {}
    for _ind, _group in enumerate(cell_groups):
        sel_adata_group = sel_adata[sel_adata.obs[groupby]==_group]
        raw_counts = sel_adata_group.X
        # convert sparse mtx if needed
        if isinstance(type(raw_counts), type(anndata._core.views.SparseCSCView)):
            raw_counts = raw_counts.toarray()
        if isinstance(type(raw_counts), type(np.ndarray)):
            # reduce accordingly for genes
            if report_type == 'mean':
                result = np.nanmean(raw_counts,axis=1)
            elif report_type == 'sum':
                result = np.nansum(raw_counts,axis=1)
            elif report_type == 'raw':
                result = raw_counts
            result_groups[_group] = result # cells are not reduced 
        else:
            print ('Wrong anndata X.')
            return None

    return result_groups




# cells are reduced 
def gene_activity_average_groups (sel_genes:list, 
                        sel_adata: anndata._core.anndata.AnnData, 
                        cell_groups,
                        groupby,
                        add_one=False,
                        ref_norm_list = [],
                        report_type ='sum'):

    """Calculate the average of the gene mean counts (from AnnData) of a list of genes (for annotated groups)"""
    ref_gene_list = sel_adata.var.index.tolist()
    sel_genes = np.intersect1d(sel_genes, ref_gene_list)

    # set norm_factors as 1 for now; which may be implemented with diff gene length, etc later
    if len(ref_norm_list) == 0:
        norm_factors = np.ones(len(sel_genes))

    result_groups = {}
    for _ind, _group in enumerate(cell_groups):
        sel_adata_group = sel_adata[sel_adata.obs[groupby]==_group]
        raw_counts = sel_adata_group.X
        # convert sparse mtx if needed
        if isinstance(type(raw_counts), type(anndata._core.views.SparseCSCView)):
            raw_counts = raw_counts.toarray()
            norm_check = False
        else:
            norm_check = True
        if isinstance(type(raw_counts), type(np.ndarray)):
            raw_means = raw_counts.mean(axis=0) # cells are averaged  
            def add_1_to_mean(raw_mean):
                if raw_mean<1:   # do not do this for (normalized) counts which are always lower than 1
                    adj_mean=1
                else:
                    adj_mean=raw_mean
                return adj_mean
            # reduce accordingly for genes
            if norm_check and not add_one:
                adj_means = list(map(add_1_to_mean, raw_means))
                
            else:
                adj_means =raw_means

            norm_adj_means = np.array(adj_means) * norm_factors

            if report_type == 'mean':
                result = np.nanmean(norm_adj_means)
            elif report_type == 'sum':
                result = np.nansum(norm_adj_means)
            elif report_type == 'raw': # mean for each gene
                result = norm_adj_means
            result_groups[_group] = result

        else:
            print ('Wrong anndata X.')
            return None

    return result_groups




def gene_activity_average_gene_dataframe (marker_genes_df:pd.core.frame.DataFrame, 
                                    adata: anndata._core.anndata.AnnData, 
                                    adjacent_gene_col = 'Adjacent_genes_500kb_tss',
                                    #stat_method = 'wilcoxon',
                                    report_type = 'sum',
                                    parallel=False, 
                                    num_threads = 8):

    """Call gene_activity_average for batch calculation for gene dataframe"""
    if report_type not in ['sum', 'mean']:
        print ('Report type not supported for the gene dataframe.')
        return None

    else:
        marker_genes_df_new = marker_genes_df.copy()
        # get cell group info
        _groups =  marker_genes_df['Compared_groups'][0].split('; ')
        groupby =  marker_genes_df['Groupby'][0]
        # get adata_raw for mtx of all genes
        adata_ori = adata.raw.to_adata()

        sel_genes_list = marker_genes_df[adjacent_gene_col].tolist()

        # prepare arg list for processing
        _mp_args = []

        # empty normalization list as placeholder for the gene_activity_score function
        ref_norm_list = []

        for _sel_genes in sel_genes_list:
            _sel_genes = _sel_genes.split('; ')
            # get subset adata for selected genes
            sel_adata =  adata_ori[:,adata_ori.var.index.isin(_sel_genes)]
            _mp_args.append((_sel_genes, sel_adata, _groups, groupby, ref_norm_list, report_type))
            
        import time
        _start_time = time.time()
        if parallel:
            import multiprocessing as mp
            print ('Multiprocessing for gene acitivity calculation:')
            with mp.Pool(num_threads) as _mp_pool:
                _mp_results = _mp_pool.starmap(gene_activity_average_groups, _mp_args, chunksize=1)
                _mp_pool.close()
                _mp_pool.join()
                _mp_pool.terminate()
            print (f"Complete in {time.time()-_start_time:.3f}s.")
        # looping will be much faster for shorter list since computing multiple sel_adata is slow
        else:
            print ('Looping for gene acitivity calculation:')
            _mp_results = [gene_activity_average_groups (*_args) for _args in _mp_args]
            print (f"Complete in {time.time()-_start_time:.3f}s.")

        # unravel the dict result from list and rearrange 
        summary_dict = {_g:[] for _g in _groups}
        for _result_dict in _mp_results:
            for _g, _result in _result_dict.items():
                summary_dict[_g].append(_result_dict[_g])
            
        for _g in _groups:
            key_added = adjacent_gene_col.replace('Adjacent', f'Activity_average_{report_type}')
            key_added = key_added + f": {_g}"
            marker_genes_df_new[key_added] = summary_dict[_g]


        return marker_genes_df_new




# 
def gene_activity_pts (sel_genes:list, 
                       ref_pts_df: pd.core.frame.DataFrame, 
                        ref_norm_list = [],
                        report_type ='mean'):

    """Calculate the gene pts for a list of genes (for annotated groups) """
    sel_genes = np.intersect1d(sel_genes, ref_pts_df.index.tolist())
    #print ('Ignore missing genes.')
    sel_genes_pts_df = ref_pts_df.loc[sel_genes]

    # set norm_factors as 1 for now; which may be implemented with diff gene length, etc later
    if len(ref_norm_list) == 0:
        norm_factors = np.ones(len(sel_genes))
    
    # append pts for each cellgroup
    result_groups = {}
    for _group in sel_genes_pts_df.columns:
        sel_genes_pts = sel_genes_pts_df[_group].tolist()
        norm_sel_genes_pts = np.array(sel_genes_pts) * norm_factors

        if report_type == 'mean':
            result = np.nanmean(norm_sel_genes_pts)
        elif report_type == 'sum':
            result = np.nansum(norm_sel_genes_pts)
    
        result_groups[_group] = result
        
    return result_groups




def gene_activity_pts_gene_dataframe (marker_genes_df:pd.core.frame.DataFrame, 
                                        adata: anndata._core.anndata.AnnData, 
                                        adjacent_gene_col = 'Adjacent_genes_100kb_tss',
                                        stat_method = 'wilcoxon',
                                        report_type = 'mean',
                                        parallel=False, 
                                        num_threads = 8):
    
    """Call gene_activity_pts for batch calculation"""
    
    marker_genes_df_new = marker_genes_df.copy()
     
    
    # get gene pts reference from adata
    check_adata_rank_genes_groups (marker_genes_df, adata,stat_method = stat_method)
    gene_pts_df = adata.uns['rank_genes_groups']['pts']
    
    _groups = gene_pts_df.columns

    sel_genes_list = marker_genes_df[adjacent_gene_col].tolist()
    
    # prepare arg list for processing
    _mp_args = []
    
    # empty normalization list as placeholder for the gene_activity_score function
    ref_norm_list = []

    for _sel_genes in sel_genes_list:
        _sel_genes = _sel_genes.split('; ')      
        _mp_args.append((_sel_genes, gene_pts_df, ref_norm_list, report_type))
    
    import time
    _start_time = time.time()
    if parallel:
        import multiprocessing as mp
        print ('Multiprocessing for gene acitivity calculation:')
        with mp.Pool(num_threads) as _mp_pool:
            _mp_results = _mp_pool.starmap(gene_activity_pts, _mp_args, chunksize=1)
            _mp_pool.close()
            _mp_pool.join()
            _mp_pool.terminate()
        print (f"Complete in {time.time()-_start_time:.3f}s.")
    # looping might be faster for shorter list
    else:
        print ('Looping for gene acitivity calculation:')
        _mp_results = [gene_activity_pts (*_args) for _args in _mp_args]
        print (f"Complete in {time.time()-_start_time:.3f}s.")
        
        
    # unravel the dict result from list and rearrange 
    summary_dict = {_g:[] for _g in _groups}
    for _result_dict in _mp_results:
        for _g, _result in _result_dict.items():
            summary_dict[_g].append(_result_dict[_g])
        
    for _g in _groups:
        key_added = adjacent_gene_col.replace('Adjacent', f'Activity_pts_{report_type}')
        key_added = key_added + f": {_g}"
        marker_genes_df_new[key_added] = summary_dict[_g]
    
    return marker_genes_df_new






def log_gene_activity (marker_genes_df:pd.core.frame.DataFrame, 
                             gene_activity_col = 'Activity_average_sum_genes_0kb_tss',
                             log_method = np.log10, 
                            log1p=True,
                            ):
    
    """Log to reduce the effect of the extremly high gene_activity_average to a similar scale for gene dataframe"""
    marker_genes_df_new = marker_genes_df.copy()
    # get cell group info
    _groups =  marker_genes_df['Compared_groups'][0].split('; ')
    #groupby =  marker_genes_df['Groupby'][0]

    for _group in _groups:
        gene_activity_col_group = f'{gene_activity_col}: {_group}'
        gene_activity_arr = np.array(marker_genes_df[gene_activity_col_group].tolist())
        # to resolve division by zero
        if log1p:
            log_gene_activity_arr = log_method(gene_activity_arr+1)
        else:
            log_gene_activity_arr = log_method(gene_activity_arr)
            
        if log_method == np.log10:
            log_col = 'Log10_' + gene_activity_col_group
        elif log_method == np.log2:
            log_col = 'Log2_' + gene_activity_col_group
            
        marker_genes_df_new[log_col]=log_gene_activity_arr
        
    return marker_genes_df_new