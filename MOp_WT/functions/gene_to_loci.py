# Import shared required packages
import numpy as np
import pandas as pd
import scanpy as sc
import anndata

# Most functions below use a loci_name with the format_version == 1;
# that is name like chr1_3740000_3760000
# Conver the format using the function below

# Function to standardzie and change the loci name format
def loci_pos_format (loci_name: str, read_loci_pos= True, format_version =1):
    """Change loci name format based on the input; can also return the chr, start and end"""
    
    # Change to name like chr1_3740000_3760000
    if format_version ==1:
        # for name like: 1:3740000-3760000 (e.g, codebook)
        if 'chr' not in loci_name:
            new_loci_name = 'chr' + loci_name.replace(':','_').replace('-','_')
            if read_loci_pos:
                _chr, _start, _end = new_loci_name.split('_')
                _chr = _chr.split('chr')[1]
                return new_loci_name, _chr, _start, _end
            else:
                return new_loci_name
                    
        # for name like: chr1_3740000_3760000 (e.g, atac or processed marker_genes_df_with_genomic info)
        elif 'chr' in loci_name:
            _chr, _start, _end = loci_name.split('_')
            _chr = _chr.split('chr')[1]
            #new_loci_name = loci_name
            if read_loci_pos:
                return loci_name, _chr, _start, _end
            else:
                return loci_name

     # Change to name like 1:3740000-3760000
    elif format_version ==2:
        # for name like: 1:3740000-3760000 (e.g, codebook)
        if 'chr' not in loci_name:
            new_loci_name = 'chr' + loci_name.replace(':','_').replace('-','_')
            if read_loci_pos:
                _chr, _start, _end = new_loci_name.split('_')
                _chr = _chr.split('chr')[1]
                return loci_name, _chr, _start, _end
            else:
                return loci_name
                    
        # for name like: chr1_3740000_3760000 (e.g, atac or processed marker_genes_df_with_genomic info)
        elif 'chr' in loci_name:
            _chr, _start, _end = loci_name.split('_')
            _chr = _chr.split('chr')[1]
            new_loci_name = _chr + ':' + _start + '-' + _end
            if read_loci_pos:
                return new_loci_name, _chr, _start, _end
            else:
                return new_loci_name
        



# Function to invert the 'chrX_start_end' to 'chrX_end_start' based on coding strand for genes/transcripts
def invert_loci_orientation (loci_name, coding_strand):
    
    """
    Invert loci name if not inverted when gene is on the minus strand
    """
    
    _chr, _start, _end  = loci_pos_format (loci_name, read_loci_pos= True, format_version =1) [1:]
    if coding_strand == -1:
        if int(_start) <= int (_end):
            new_loci_name = 'chr' + _chr + '_' + _end + '_' + _start
        # if already 'inverted' (accidently)
        else:
            new_loci_name = loci_name
    else:
        new_loci_name = loci_name
        
    return new_loci_name




# Function to get the genomic information for marker genes
def get_genomic_for_gene_dataframe (marker_genes_df:pd.core.frame.DataFrame, 
                                    gene_annotation_df:pd.core.frame.DataFrame,):
    
    """Add genomic pos and related information"""
    
    marker_genes_df_new = marker_genes_df.copy()
    # cols to be added
    new_cols = ['Genomic_position','Length','Gene_biotype','Coding_strand']
    marker_genes_df_new[new_cols] = ""

    # add info for genes exist in the annotation; 
    # note the index order will be reset after np.intersect1d
    sel_gene_list = marker_genes_df.index.tolist()
    ref_gene_list = gene_annotation_df.index.tolist()
    shared_gene_list = np.intersect1d(sel_gene_list,ref_gene_list)
    
    # subset both df
    sel_gene_annotation_df=gene_annotation_df.loc[shared_gene_list]
    sel_marker_genes_df_new =marker_genes_df_new.loc[shared_gene_list]

    # append cols
    for _col in new_cols:
        if _col.lower() in sel_gene_annotation_df.columns:
            sel_marker_genes_df_new[_col] = sel_gene_annotation_df[_col.lower()]

    # replace the modified subset; the original index should be kept this way
    marker_genes_df_new.loc[shared_gene_list] = sel_marker_genes_df_new  

    return marker_genes_df_new



# Function to find a adjacent loci for the query loci from a ref loci list
def match_adjcent_loci_from_ref (loci_name, ref_loci_list, nearby_type, nearby_dist, coding_strand):
    
    """Use a loci name to find a close proxy from a ref list of loci name"""

    # call loci_pos_format to standardize the loci name format and extract info

    # if input has a loci
    if len(loci_name)>0 and 'PATCH' not in loci_name: # temp solution for removing incompatible loci name
        _chr, _start, _end = loci_pos_format(loci_name, read_loci_pos= True, format_version =1)[1:]
        if nearby_type == 'center':
            _mid = int((int(_start) + int(_end))/2)
            query_pos = _mid
        elif nearby_type == 'tss' and coding_strand == 1:
            query_pos = int(_start)
        elif nearby_type == 'tss' and coding_strand == -1:
            query_pos = int(_end)
        
        # get chr_name, chr, start and end info
        # make sure its using the default version1 format
        ref_loci_list_v1 = list(map(loci_pos_format, ref_loci_list))

        # conver to array with cols: chr_name, chr, start, end
        ref_loci_list_v1 = [list(_l) for _l in ref_loci_list_v1]
        ref_loci_list_v1 = np.array(ref_loci_list_v1)
        
        # slice for chromosome
        ref_loci_list_v1 = ref_loci_list_v1[ref_loci_list_v1[:,1]==_chr]
        # get a new mid_pos col for dist calculation
        _starts = np.array(list(map(lambda x: int(x), ref_loci_list_v1[:,2])))
        _ends = np.array(list(map(lambda x: int(x), ref_loci_list_v1[:,3])))
        _mids = _starts + (_ends - _starts)/2
        _dists = abs(_mids-query_pos)
        # get the smalles dists
        _sel_dist = np.min(_dists)
        # add if it < dist_th
        if _sel_dist <= nearby_dist:
            _sel_ind = np.argmin(_dists)
            sel_loci = ref_loci_list_v1[_sel_ind,0]
        else:
            sel_loci = ""
        
    # if input is empty
    else:
        sel_loci = ""
    
    return sel_loci


# Batch function to process a list of query loci
def batch_match_adjcent_loci_from_ref (query_loci_list,
                                       ref_loci_list, 
                                       coding_strand_list=None, 
                                       nearby_type='tss', 
                                       nearby_dist=100*1000, 
                                       num_threads = 8, 
                                       parallel=True):

    """Batch function for above to match a list of loci"""
    # set default conding strand as forward strand
    if isinstance(coding_strand_list, type(None)):
        coding_strand_list = [1 for _i in range(len(query_loci_list))]
    # proceed if list length matches
    if len(coding_strand_list)==len(query_loci_list):
        pass
    else:
        return None

    # prepare mp args
    _mp_args = []
    for _loci, _strand in zip(query_loci_list,coding_strand_list):
        _mp_args.append((_loci, ref_loci_list, nearby_type, nearby_dist, _strand))
    
    import time
    _start_time = time.time()
    # mp or sequentiall compute
    if parallel:
        import multiprocessing as mp
        print ('Multiprocessing for loci matching:')
        with mp.Pool(num_threads) as _mp_pool:
            _loci_results = _mp_pool.starmap(match_adjcent_loci_from_ref, _mp_args, chunksize=1)
            _mp_pool.close()
            _mp_pool.join()
            _mp_pool.terminate()
        print (f"Complete in {time.time()-_start_time:.3f}s.")
    else:
        print ('Looping for loci matching:')
        _loci_results = [match_adjcent_loci_from_ref(*_args) for _args in _mp_args]
        print (f"Complete in {time.time()-_start_time:.3f}s.")

    sel_loci_list = _loci_results

    return sel_loci_list



# Function to get adjacent imaged loci for marker genes if exists
def get_imaged_loci_near_gene_dataframe (marker_genes_df:pd.core.frame.DataFrame, 
                                         codebook_df:pd.core.frame.DataFrame, 
                                         nearby_type='tss', 
                                         nearby_dist=100*1000, 
                                         num_threads = 8, 
                                         parallel=True, 
                                         clean_df=True):
    """Find imaged loci near the gene's genomic position accordingly"""
    marker_genes_df_new= marker_genes_df.copy()
    
    genomic_pos_list = marker_genes_df['Genomic_position'].tolist()
    coding_strand_list = marker_genes_df['Coding_strand'].tolist()
    
    imaged_loci_list = codebook_df['name'].tolist()
    
    sel_loci_list = batch_match_adjcent_loci_from_ref (genomic_pos_list,
                                       imaged_loci_list, 
                                       coding_strand_list=coding_strand_list, 
                                       nearby_type=nearby_type, 
                                       nearby_dist=nearby_dist, 
                                       num_threads =num_threads, 
                                       parallel=parallel)
    
    marker_genes_df_new['Imaged_loci'] = sel_loci_list


    if clean_df:
        print ('Remove loci whose match were not found.')
        marker_genes_df_new = marker_genes_df_new[marker_genes_df_new['Imaged_loci'].str.contains('chr')]
    else:
        print ('Keep all loci.')
    
    return marker_genes_df_new



# Function to find genes near specific loci using the pre-prepared gene_annotation_df
def find_genes_near_loci (loci_name: str, 
                          gene_annotation_df: pd.core.frame.DataFrame,
                          extend_dist = 50*1000, gene_coverage_type = 'center'):
    
    """
    Find genes near loci using the gene_annotation_df:
    
    the genomic coord could be merged for some genes with multiple transcripts by taking the two farthest ends
    
    annotation: ENSEMBL 
    
    gene_coverage_type = 'center' or 'tss' or 'both' 
    
    """
    
    # read _chr, _start, _end from query loci; then extend
    _chr, _start, _end  = loci_pos_format (loci_name, read_loci_pos= True, format_version =1) [1:]
    _start = int(_start)
    _end = int(_end)
    _start = int(max(0, (_start - extend_dist)))
    _end = int(_end + extend_dist) # it'd be fine if the end is larger than the chr len

    # slice annotation df by chr_key
    sel_gene_annotation_df = gene_annotation_df[gene_annotation_df['chr']==_chr]
    
    if gene_coverage_type == 'tss':
        # 'invert' the start and end for genes on the minus strand
        strand_gene_pos_list = list(map(invert_loci_orientation, 
                                        sel_gene_annotation_df['genomic_position'].tolist(),
                                        sel_gene_annotation_df['coding_strand'].tolist(),
                                       ))
    else:
        # for 'both' or 'center' type, the orientation doesn't matter
        strand_gene_pos_list = sel_gene_annotation_df['genomic_position'].tolist()
        
    
    # get the pos list for all relevant genes
    gene_pos_list_v1 = list(map(loci_pos_format, strand_gene_pos_list))
    # conver to array with cols: chr_name, chr, start, end
    gene_pos_list_v1 = [list(_l) for _l in gene_pos_list_v1]
    gene_pos_list_v1 = np.array(gene_pos_list_v1)
    
    _gene_starts = np.array(list(map(lambda x: int(x), gene_pos_list_v1[:,2])))
    _gene_ends = np.array(list(map(lambda x: int(x), gene_pos_list_v1[:,3])))
    
    # only the tss is required to be inside the extended query loci
    if gene_coverage_type == 'tss':
        _inds_start =  _gene_starts>=_start
        _inds_end =  _gene_starts<=_end
    # both ends of a gene are required to be inside the extended query loci
    elif gene_coverage_type == 'both':
        _inds_start =  _gene_starts>=_start
        _inds_end =  _gene_ends<=_end
    # only the center of a gene is required to be inside the extended query loci
    elif gene_coverage_type == 'center':
        _gene_mids = _gene_starts + (_gene_ends - _gene_starts)/2
        _inds_start =  _gene_mids>=_start
        _inds_end =  _gene_mids<=_end    
    
    # slice genes with inds
    _inds_good = _inds_start*_inds_end
    sel_genes = sel_gene_annotation_df.index[_inds_good]
    
    if len(sel_genes)==0:
        sel_genes = 'intergenic'
        
    else:
        sel_genes = '; '.join(sel_genes)

    return sel_genes



# Function to add nearby genes for genes or the imaged loci by directly loading the pre-prepared loci_adjcent_genes_df
# Can also use this for a loci df if 'Imaged_loci' is the index col
def direct_get_genes_near_gene_dataframe (marker_genes_df:pd.core.frame.DataFrame,
                                   loci_adjcent_genes_df:pd.core.frame.DataFrame, 
                                   adjacent_gene_col = None):
    """ """
    # Direct get by loading a loci_adjcent_genes_df, which prepared in advance
    # if need annotation for more/other loci, either prepare them first;
    # OR alternatively, call find_genes_near_loci/find_genes_near_gene_dataframe for any specifc loci_name
    
    # below, the imaged loci coord from marker_genes_df should match the loci_adjcent_genes_df
    marker_genes_df_new=marker_genes_df.copy()
    
    if 'Imaged_loci' in marker_genes_df.columns:
        loci_marker_genes = marker_genes_df['Imaged_loci'].tolist()
    elif 'Imaged_loci' == marker_genes_df.index.name:
        loci_marker_genes = marker_genes_df.index.tolist()
    else:
        print ('No candidate loci list available. Exit.')
        return None

    # since all query loci exists in the loci_adjcent_genes_df, index order will be kept
    sel_loci_adjcent_genes_df= loci_adjcent_genes_df.loc[loci_marker_genes]
    # add the annotation column accordingly
    if isinstance(adjacent_gene_col, type(None)):
        print ('Get all existing adjacent gene columns.')
        adjacent_gene_cols = [_col for _col in loci_adjcent_genes_df.columns if 'adjacent_genes' in _col]
    
        for _col in adjacent_gene_cols:
            _new_col = _col.capitalize()
            marker_genes_df_new[_new_col]=sel_loci_adjcent_genes_df[_col].tolist()
    
    else:
        _col = adjacent_gene_col
        _new_col = _col.capitalize()
        marker_genes_df_new[_new_col]=sel_loci_adjcent_genes_df[_col].tolist()

    return marker_genes_df_new


# NOT TESTED YET
# Alternative Function to add nearby genes for genes or the imaged loci by de novo calculating
# Can also use this to make loci_adjcent_genes_df for a loci df
def find_genes_near_gene_dataframe (marker_genes_df:pd.core.frame.DataFrame, 
                                    gene_annotation_df:pd.core.frame.DataFrame, 
                                    sel_loci_col = 'Imaged_loci',
                                    extend_dist = 50*1000,
                                    gene_coverage_type = 'tss',
                                    key_added = None):


    """
    Find genes near loci for a gene dataframe
    use a selected loci column to find;
    default is 'Imaged_loci; can also use 'Genomic_position'
    the loci column needs to be all filled; thus use the cleaned df after imaged loci search
    """
    marker_genes_df_new = marker_genes_df.copy()

    loci_list = marker_genes_df[sel_loci_col].tolist()

    sel_genes_list = []
    for loci_name in loci_list:
        sel_genes =  find_genes_near_loci (loci_name, 
                                           gene_annotation_df,
                                           extend_dist = extend_dist, 
                                           gene_coverage_type = gene_coverage_type
                                          )
        sel_genes_list.append(sel_genes)
    
    if isinstance (key_added, type(None)):
        extend_dist_name = str(int(extend_dist/1000)) + 'kb'
        key_added = f'adjacent_genes_{extend_dist_name}_{gene_coverage_type}'
    
    marker_genes_df_new[key_added]=sel_genes_list
    
    return marker_genes_df_new