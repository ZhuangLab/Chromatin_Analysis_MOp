# Import shared required packages
import numpy as np
import pandas as pd
import scanpy as sc
import anndata

from gene_to_loci import loci_pos_format


# Function to find peaks near specific loci 
def find_peaks_near_loci (loci_name: str, 
                          peak_names_list: list,
                          extend_dist = 50*1000, peak_coverage_type = 'center'):
    
    """
    Find peaks near loci using the peak_names_df:

    peak_coverage_type = 'center' or 'both' (ends) to be covered 
    
    """
    # read _chr, _start, _end from query loci; then extend
    _chr, _start, _end  = loci_pos_format (loci_name, read_loci_pos= True, format_version =1) [1:]
    _start = int(_start)
    _end = int(_end)
    _start = int(max(0, (_start - extend_dist)))
    _end = int(_end + extend_dist) # it'd be fine if the end is larger than the chr len


    # standardize name for the peak list
    peak_names_list_v1 = list(map(loci_pos_format, peak_names_list))
    # conver to array with cols: chr_name, chr, start, end
    peak_names_list_v1 = [list(_l) for _l in peak_names_list_v1]
    peak_names_list_v1 = np.array(peak_names_list_v1)
    # slice arr for chromosome
    peak_names_list_v1_chr = peak_names_list_v1[peak_names_list_v1[:, 1]==_chr]
    
    _peak_starts = np.array(list(map(lambda x: int(x), peak_names_list_v1_chr[:,2])))
    _peak_ends = np.array(list(map(lambda x: int(x), peak_names_list_v1_chr[:,3])))
    
    # both ends of a gene are required to be inside the extended query loci
    if peak_coverage_type == 'both':
        _inds_start =  _peak_starts>=_start
        _inds_end =  _peak_ends<=_end
    # only the center of a gene is required to be inside the extended query loci
    elif peak_coverage_type == 'center':
        _peak_mids = _peak_starts + (_peak_ends - _peak_starts)/2
        _inds_start =  _peak_mids>=_start
        _inds_end =  _peak_mids<=_end    
    
    # slice genes with inds
    _inds_good = _inds_start * _inds_end
    sel_peaks = peak_names_list_v1_chr[:, 0][_inds_good]
    
    if len(sel_peaks)==0:
        sel_peaks = 'None'
        
    else:
        sel_peaks = '; '.join(sel_peaks)

    return sel_peaks





def find_peaks_near_gene_dataframe (im_loci_df:pd.core.frame.DataFrame, 
                                peak_names_list:list, 
                                sel_loci_col = None,
                                extend_dist = 50*1000,
                                peak_coverage_type = 'center',
                                key_added = None):

    """
    Find peaks near loci for a gene dataframe
    use loci index names or a selected loci column to find;
    """
    im_loci_df_new = im_loci_df.copy()
    
    if isinstance(sel_loci_col, str) and sel_loci_col in im_loci_df_new.columns:
        loci_list = im_loci_df[sel_loci_col].tolist()
    else:
        loci_list = im_loci_df.index.tolist()

    sel_peaks_list = []
    from tqdm import tqdm
    for loci_name in tqdm(loci_list):
        sel_peaks =  find_peaks_near_loci (loci_name, 
                                        peak_names_list,
                                        extend_dist = extend_dist, 
                                        peak_coverage_type = peak_coverage_type
                                        )
        sel_peaks_list.append(sel_peaks)
    
    if isinstance (key_added, type(None)):
        extend_dist_name = str(int(extend_dist/1000)) + 'kb'
        key_added = f'adjacent_peaks_{extend_dist_name}_{peak_coverage_type}'
    
    im_loci_df_new[key_added]=sel_peaks_list
    
    return im_loci_df_new





