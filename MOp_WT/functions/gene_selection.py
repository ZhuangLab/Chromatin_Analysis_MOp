# Import shared required packages
import numpy as np
import pandas as pd
import scanpy as sc
import anndata


# Function to get ranked marker genes and relatd information between two cell types with using scanpy's function
def rank_genes_groups_into_dataframe (adata: anndata._core.anndata.AnnData, 
                                      groupby:str, 
                                      cellgroup_1:str,
                                      cellgroup_2:str, 
                                      n_genes=300, 
                                      stat_method='wilcoxon', 
                                      add_control_genes = True,
                                      pool_size = 1, 
                                      use_raw = True,
                                      **kwargs):
    
    '''Get ranked marker genes and relatd information between two cell groups.'''
    
    ################################
    # add checking for args later if desired
    ################################
    # use tie-correct for wilcoxon rank sum test
    if stat_method=='wilcoxon':
        tie_correct=True
    else:
        tie_correct=False
        
    print (f'Start Gene Selection:')
    import time
    _start_time = time.time()
    
    # 1. Perform rank genes analysis between the two cell group.
    print (f'1. Ranking marker genes with Scanpy using {stat_method}.')
    sc.tl.rank_genes_groups(adata,
                            groupby=groupby, 
                            groups=[cellgroup_1],
                            method=stat_method,
                            reference=cellgroup_2,
                            tie_correct=tie_correct,
                            use_raw=use_raw,
                            pts=True, 
                            pts_rest=True,
                            **kwargs)
    
    # 2. Slice and process results to get marker genes based on the gene expression change direction.
    print (f'2. Selecting marker genes for expression change direction.')
    marker_genes_dict={}  
    # make sure the stat_method is symmetric so markers for each compared cell group are on each side of the ranked list
    # upregulated for marker in the cellgroup1 and downregulated for marker in the cellgroup2
    if add_control_genes:
        directions = ['upregulated','downregulated','control']
    else:
        directions = ['upregulated','downregulated']
    
    # for each direction of the ranked list: subset, unravel rec (,and reverse order if needed)
    for _dir in directions:
        if _dir == 'upregulated':
            marker_genes_cellgroup = adata.uns['rank_genes_groups']['names'][:n_genes]
            marker_genes_cellgroup = [_g[0] for _g in marker_genes_cellgroup]
            marker_genes_dict[_dir] = marker_genes_cellgroup
        elif _dir == 'downregulated':
            marker_genes_cellgroup = adata.uns['rank_genes_groups']['names'][-n_genes:]
            marker_genes_cellgroup = [_g[0] for _g in marker_genes_cellgroup]
            marker_genes_cellgroup.reverse()
            marker_genes_dict[_dir] = marker_genes_cellgroup          
        elif _dir == 'control':
            mid_num = int(len(adata.uns['rank_genes_groups']['names'])/2)
            marker_genes_cellgroup = adata.uns['rank_genes_groups']['names'][int(mid_num-n_genes*pool_size):int(mid_num+n_genes*pool_size)]
            marker_genes_cellgroup = [_g[0] for _g in marker_genes_cellgroup]
            import random
            # random sample for unique sampling
            marker_genes_cellgroup = random.sample(marker_genes_cellgroup, k=n_genes)
            marker_genes_dict[_dir] = marker_genes_cellgroup    
    
    # 3. Store result into a DataFrame for subsequent filtering/annotating
    print (f'3. Generating a DataFrame for the selected marker genes.')
    marker_genes_df =pd.DataFrame(columns =['Gene_name','Expression_change'])
    for _dir, _genes in marker_genes_dict.items():
        gene_df = pd.DataFrame(_genes,columns =['Gene_name'])
        print (f'There are {len(gene_df)} genes selected for {_dir} expression.')
        gene_df['Expression_change']=_dir
        marker_genes_df =pd.concat([marker_genes_df,gene_df])

    marker_genes_df.index = marker_genes_df['Gene_name']
    marker_genes_df['Groupby'] = groupby
    marker_genes_df['Compared_groups'] = cellgroup_1 + '; ' + cellgroup_2
    marker_genes_df = marker_genes_df.drop(columns=['Gene_name'])

    print (f'Complete Gene Selection in {time.time()-_start_time:.3f}s.')
    return marker_genes_df




# Function to check if rank genes groups has been performed for the input anndata;
# If not, re-do the analysis using the given parameters extracted from the gene dataframe
def check_adata_rank_genes_groups (marker_genes_df:pd.core.frame.DataFrame, 
                                   adata: anndata._core.anndata.AnnData, stat_method = 'wilcoxon'):

    """"""
    # Check if corresponding analysis is kept in the given adata
    groups = marker_genes_df['Compared_groups'][0].split('; ')
    if 'rank_genes_groups' in adata.uns.keys():
        if groups == adata.uns['rank_genes_groups']['pts'].columns.tolist():
            print ('Use existing result from the input adata.')
            return adata

        elif 'rest' in groups:
            if groups[0] == adata.uns['rank_genes_groups']['pts'].columns.tolist()[0]:
                print ('Use existing result from the input adata.')
                return adata
        
        else:
            pass
    else:
        pass
    
    # Proceed to re-analysis
    print ('Re-perform the Rank Genes Groups analysis with the given params.')
    use_raw = True
    if stat_method=='wilcoxon':
        tie_correct=True
    else:
        tie_correct=False

    cellgroup_1, cellgroup_2 = groups
    groupby = marker_genes_df['Groupby'][0]
    print (f'Use {stat_method} to repeat the Rank Genes Groups analysis.')

    sc.tl.rank_genes_groups(adata,
                        groupby=groupby, 
                        groups=[cellgroup_1],
                        method=stat_method,
                        reference=cellgroup_2,
                        tie_correct=tie_correct,
                        use_raw=use_raw,
                        pts=True, 
                        pts_rest=True,
                        )
    return adata



# Function to add zscore for marker genes;
# requires the above rank gene function to be completed first; 
def add_scores_into_gene_dataframe (adata: anndata._core.anndata.AnnData, 
                                    marker_genes_df:pd.core.frame.DataFrame,
                                    stat_method = 'wilcoxon'):


    check_adata_rank_genes_groups (marker_genes_df, adata,stat_method = stat_method)

    print (f'Add Gene Scores from Rank Genes Groups analysis.')
    marker_genes_df_new = marker_genes_df.copy()

    marker_genes_rank_names = adata.uns['rank_genes_groups']['names']
    marker_genes_rank_names = [_g[0] for _g in marker_genes_rank_names]

    marker_genes_rank_scores = adata.uns['rank_genes_groups']['scores']
    marker_genes_rank_scores = [_s[0] for _s in marker_genes_rank_scores]

    sel_scores = []
    for _g in marker_genes_df.index.tolist():
        _ind = marker_genes_rank_names.index(_g)
        _score = marker_genes_rank_scores[_ind]
        sel_scores.append(_score)

    marker_genes_df_new['Scores'] = sel_scores

    return marker_genes_df_new




# Function to add gene expression percentage to the marker_genes_df;
# requires the above rank gene function to be completed first; 
def add_pts_into_gene_dataframe (adata: anndata._core.anndata.AnnData, 
                                marker_genes_df:pd.core.frame.DataFrame,
                                stat_method = 'wilcoxon'):


    '''Add gene expression percentage for marker genes.'''

    check_adata_rank_genes_groups (marker_genes_df, adata,stat_method = stat_method)

    ################################
    # add checking for args later if desired
    ################################
    print (f'Add Gene Expression Percentage.')
    marker_genes_df_new = marker_genes_df.copy()
    gene_pts_df = adata.uns['rank_genes_groups']['pts']

    # extract df for marker genes only
    sel_genes_pts_df = gene_pts_df.loc[marker_genes_df.index]
    # append df col, which keeps index order
    for _col_name in sel_genes_pts_df.columns:
        _frac_col = sel_genes_pts_df[_col_name]
        marker_genes_df_new[f'Pts: {_col_name}'] = _frac_col

    # rest-pts was saved in a different df
    if 'rest' in marker_genes_df['Compared_groups'][0].split('; '):
        gene_pts_df = adata.uns['rank_genes_groups']['pts_rest']
        sel_genes_pts_df = gene_pts_df.loc[marker_genes_df.index]
        _frac_col = sel_genes_pts_df
        marker_genes_df_new[f'Pts: rest'] = _frac_col

    return marker_genes_df_new


# Function to add gene expression mean and the log2foldchange to the marker_genes_df;
# Re-calculate the log2foldchange to avoid inf and nan from scanpy's function after normalization
# requires the above rank gene function to be completed first; 
def add_mean_into_gene_dataframe (adata: anndata._core.anndata.AnnData, 
                                    marker_genes_df:pd.core.frame.DataFrame,
                                    add_log2fold =True):


    '''Add gene expression mean and the log2foldchange for marker genes.'''

    ################################
    # add checking for args later if desired
    ################################
    print (f'Add Gene Expression Means.')

    marker_genes_df_new=marker_genes_df.copy()

    # use adata.raw to get all genes
    adata_ori = adata.raw.to_adata()
    # get cell group info
    _groups =  marker_genes_df['Compared_groups'][0].split('; ')
    groupby =  marker_genes_df['Groupby'][0]

    # get subset adata for marker genes
    sel_adata =  adata_ori[:,adata_ori.var.index.isin(marker_genes_df.index)]

    # note that the gene index order from the adata is different from the input pf
    # here we get the index order to sort the index 
    sort_inds = []
    for _g in marker_genes_df.index.tolist():
        _ind = sel_adata.var.index.tolist().index(_g)
        sort_inds.append(_ind)
    # sort
    sel_adata = sel_adata[:,sort_inds]

    # calculate mean for each cell group; adjust mean to 1 if lower than 1 to accomendate issues in foldchange
    for _group in _groups:
        if _group == 'rest': # mean of the rest groups combined
            sel_adata_group = sel_adata[sel_adata.obs[groupby]!=_groups[0]]
        else:
            sel_adata_group = sel_adata[sel_adata.obs[groupby]==_group]

        raw_counts = raw_counts = sel_adata_group.X
        # for sparse matrix
        if isinstance(type(raw_counts), type(anndata._core.views.SparseCSCView)):
            raw_counts = raw_counts.toarray()
        if isinstance(type(raw_counts), type(np.ndarray)):
            raw_means = raw_counts.mean(axis=0) # cells are averaged  

            def add_1_to_mean(raw_mean):
                if raw_mean<1:
                    adj_mean=1
                else:
                    adj_mean=raw_mean
                return adj_mean
            adj_means = list(map(add_1_to_mean, raw_means))
            marker_genes_df_new[f'Mean: {_group}'] = adj_means
        else:
            print ('Wrong anndata X.')
            return None

    if add_log2fold:
        marker_genes_df_new[f'Log2foldchange'] = np.log2(marker_genes_df_new[f'Mean: {_groups[0]}']/
                                                        marker_genes_df_new[f'Mean: {_groups[1]}'])


    return marker_genes_df_new



# Function to filter marker genes by pts
# conceptually, the markers should also be widely expressed for one group but not in the compared group
# by default the controls are "inactive" genes for each compared groups
def filter_pts_for_gene_dataframe (marker_genes_df:pd.core.frame.DataFrame,
                                   high_th_in_group_1 = 0.75,
                                   low_th_in_group_2 = 0.25, 
                                   high_th_in_group_2 = None,
                                   low_th_in_group_1 = None, 
                                   control_as_inactive=True):


    '''Filter genes using expression percentage from the compared groups.'''
    print (f'Filter Genes by Expression Percentage.')
    # get expression change types and start from upregulated
    directions = np.unique(marker_genes_df['Expression_change'].tolist())
    directions = np.flip(directions) 

    # get column name for filtering
    _filter_col = 'Pts: '
    _groups =  marker_genes_df['Compared_groups'][0].split('; ')
    _filter_cols = [(_filter_col + _group) for _group in _groups]

    # if use same set of th for both groups
    # note: a group_1-specific marker's pts in group_1 shall be higher than high_th_in_group_1; 
    # and its pts in group_2 shall be lower than low_th_in_group_2; 
    # filter similarly for a group_2-specific marker
    if isinstance(high_th_in_group_2, type(None)):
        high_th_in_group_2=high_th_in_group_1
    if isinstance(low_th_in_group_1, type(None)):
        low_th_in_group_1=low_th_in_group_2

    # filter by the given threshold 
    marker_genes_df_new = pd.DataFrame(columns=marker_genes_df.columns)
    for _dir in directions:
        sel_marker_genes_df = marker_genes_df[marker_genes_df['Expression_change']==_dir]
        # process group_1 and group_2 sequentially for each direction
        if _dir == 'upregulated':
            th_list=[high_th_in_group_1,low_th_in_group_2]
            sel_marker_genes_df_new = sel_marker_genes_df[sel_marker_genes_df[_filter_cols[0]]>=th_list[0]]
            sel_marker_genes_df_new = sel_marker_genes_df_new[sel_marker_genes_df_new[_filter_cols[1]]<=th_list[1]]

        elif _dir == 'downregulated':
            th_list=[high_th_in_group_2,low_th_in_group_1]
            sel_marker_genes_df_new = sel_marker_genes_df[sel_marker_genes_df[_filter_cols[1]]>=th_list[0]]
            sel_marker_genes_df_new = sel_marker_genes_df_new[sel_marker_genes_df_new[_filter_cols[0]]<=th_list[1]]

        elif _dir == 'control':
            if control_as_inactive:
                th_list=[low_th_in_group_1,low_th_in_group_2]
                sel_marker_genes_df_new = sel_marker_genes_df[sel_marker_genes_df[_filter_cols[0]]<=th_list[0]]
                sel_marker_genes_df_new = sel_marker_genes_df_new[sel_marker_genes_df_new[_filter_cols[1]]<=th_list[1]]
            else:
                th_list=[high_th_in_group_1,high_th_in_group_2]
                sel_marker_genes_df_new = sel_marker_genes_df[sel_marker_genes_df[_filter_cols[0]]>=th_list[0]]
                sel_marker_genes_df_new = sel_marker_genes_df_new[sel_marker_genes_df_new[_filter_cols[1]]>=th_list[1]]   
        
        print (f'There are {len(sel_marker_genes_df_new)} genes kept for {_dir} expression.')
        marker_genes_df_new = pd.concat([marker_genes_df_new, sel_marker_genes_df_new])
            
    return  marker_genes_df_new




# Function to filter marker genes by mean count
# conceptually, the markers' mean count should be higher than a value
def filter_mean_for_gene_dataframe (marker_genes_df:pd.core.frame.DataFrame,
                                   high_th_in_group_1 = None, 
                                   high_th_in_group_2 = None):


    '''Filter genes for so that the mean for each is higher than a given threshold.'''

    print (f'Filter Genes by Minimal Mean Counts.')
    # get expression change types and start from upregulated
    directions = np.unique(marker_genes_df['Expression_change'].tolist())
    directions = np.flip(directions) 

    # get column name for filtering
    _filter_col = 'Mean: '
    _groups =  marker_genes_df['Compared_groups'][0].split('; ')
    _filter_cols = [(_filter_col + _group) for _group in _groups]

    if isinstance(high_th_in_group_1, type(None)):
        high_th_in_group_1=np.percentile(marker_genes_df[_filter_cols[0]],25)
    # if use same set of th for both groups
    if isinstance(high_th_in_group_2, type(None)):
        high_th_in_group_2=high_th_in_group_1
    print(f'Use {high_th_in_group_1} and {high_th_in_group_2} as mean threshold for each group filtering.')

    # filter by the given threshold 
    marker_genes_df_new = pd.DataFrame(columns=marker_genes_df.columns)
    for _dir in directions:
        sel_marker_genes_df = marker_genes_df[marker_genes_df['Expression_change']==_dir]
        th_list=[high_th_in_group_1,high_th_in_group_2]
        if _dir == 'upregulated':
            sel_marker_genes_df_new = sel_marker_genes_df[sel_marker_genes_df[_filter_cols[0]]>=th_list[0]]
        elif _dir == 'downregulated':
            sel_marker_genes_df_new = sel_marker_genes_df[sel_marker_genes_df[_filter_cols[1]]>=th_list[1]] 
        elif _dir == 'control':
            sel_marker_genes_df_new = sel_marker_genes_df
            
        print (f'There are {len(sel_marker_genes_df_new)} genes kept for {_dir} expression.')
        marker_genes_df_new = pd.concat([marker_genes_df_new, sel_marker_genes_df_new])

    
    return  marker_genes_df_new




# Function to filter marker genes by log2foldchange
# conceptually, the higher the foldchange, the difference is bigger
def filter_foldchange_for_gene_dataframe (marker_genes_df:pd.core.frame.DataFrame,
                                   change_th_in_group_1 = 1, 
                                   change_th_in_group_2 = None):


    '''Filter genes with a given threshold of log2foldchange.'''
    print (f'Filter Genes by Minimal Fold-Change.')
    # get expression change types and start from upregulated
    directions = np.unique(marker_genes_df['Expression_change'].tolist())
    directions = np.flip(directions) 

    # get column name for filtering
    _filter_col = 'Log2foldchange'

    # if use same set of th for both groups
    if isinstance(change_th_in_group_2, type(None)):
        change_th_in_group_2=change_th_in_group_1
    print(f'Use {change_th_in_group_1} and {change_th_in_group_2} as foldchange threshold for each group filtering.')

    # filter by the given threshold 
    marker_genes_df_new = pd.DataFrame(columns=marker_genes_df.columns)
    for _dir in directions:
        sel_marker_genes_df = marker_genes_df[marker_genes_df['Expression_change']==_dir]
        th_list=[change_th_in_group_1,change_th_in_group_2]
        if _dir == 'upregulated':
            sel_marker_genes_df_new = sel_marker_genes_df[abs(sel_marker_genes_df[_filter_col])>=th_list[0]]
        elif _dir == 'downregulated':
            sel_marker_genes_df_new = sel_marker_genes_df[abs(sel_marker_genes_df[_filter_col])>=th_list[1]] 
        elif _dir == 'control':
            sel_marker_genes_df_new = sel_marker_genes_df
            
        print (f'There are {len(sel_marker_genes_df_new)} genes kept for {_dir} expression.')
        marker_genes_df_new = pd.concat([marker_genes_df_new, sel_marker_genes_df_new])

    return  marker_genes_df_new



# Function to save the processed dataframe
def save_gene_dataframe (marker_genes_df:pd.core.frame.DataFrame,
                        save_folder, 
                        save_preifx="",
                        save_subfolder = None, 
                        save_name = None, 
                        #check_cols = True  #not implemented yet
                        ):

    '''Save marker_genes_df.'''    
    import os
    if isinstance(save_name, type(None)):
        # get info from columns as savename 
        _groups =  marker_genes_df['Compared_groups'][0].split('; ')
        groups = '-'.join(_groups)
        groupby =  marker_genes_df['Groupby'][0]
        save_name = groupby+'_'+groups+'.csv'
    
    # add prefix if desired
    save_name = save_preifx + save_name

    if not isinstance(save_subfolder, type(None)):
        if isinstance (save_subfolder,str):
            save_folder=os.path.join(save_folder,save_subfolder)
    
    if not os.path.exists(save_folder):
        print ('Making savefolder:', save_folder)
        os.mkdir(save_folder)
    
    save_name_full = os.path.join(save_folder,save_name)
    if save_name_full.split('.')[-1]=='csv':
        marker_genes_df.to_csv(save_name_full,index=True)
        print ('DataFrame saved.')
    elif save_name_full.split('.')[-1]=='xlsx':
        marker_genes_df.to_excel(save_name_full,index=True)
        print ('DataFrame saved.')
    else:
        print ('The specified save format is not compatible.')

    return marker_genes_df


