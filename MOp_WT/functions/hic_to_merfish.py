import tqdm
import numpy as np
import pandas as pd

# function to convert m3C count extracted tsv to half-matrix for one chromosome
def convert_count_to_merfish_matrix_byChr (_chr_count_df, codebook_df_chr, _ext=500000):
    
    _chr_contact_dict = {}
    _query_loci_list_1 = np.array(_chr_count_df[2]) # hic loci pos 1
    _query_loci_list_2 = np.array(_chr_count_df[6]) # hic loci pos 2
    
    codebook_df_chr_loci = codebook_df_chr.index # codebook loci list
    for _loci in codebook_df_chr_loci:
        _start =int(_loci.split('_')[1])-_ext # bp to ext for both upstream and downstream
        _stop = int(_loci.split('_')[2])+_ext # bp to ext for both upstream and downstream
        # hic loci 1 in the targeted imaged codebook loci
        sel_query_loci_list_1 = (_query_loci_list_1>_start)&(_query_loci_list_1<_stop)

        _chr_contact_dict[_loci] = {}
        # hic loci 2 in the targeted imaged codebook loci_2
        for _loci_2 in codebook_df_chr_loci:
            _start_2 =int(_loci_2.split('_')[1])-_ext
            _stop_2 = int(_loci_2.split('_')[2])+_ext
            sel_query_loci_list_2 = (_query_loci_list_2>_start_2)&(_query_loci_list_2<_stop_2)
            sel_query_loci_list_pair = sel_query_loci_list_1 & sel_query_loci_list_2\
            # number of pairs that qualify 
            _chr_contact_dict[_loci][_loci_2] = np.sum(sel_query_loci_list_pair) 

    _chr_count_matrix_df = pd.DataFrame(_chr_contact_dict)
    _chr_count_matrix  = _chr_count_matrix_df.values
    
    return _chr_count_matrix
    

# batch function to convert for all chromosome selected for each single cell
def batch_convert_count_to_merfish_matrix_dict (hic_extract_df, 
                                                codebook_df, 
                                                all_sorted_chr, 
                                                _ext = 500000,
                                                num_threads = 16,
                                                parallel=False):

    # prepare mp args
    _mp_args = []
    for _chr in all_sorted_chr[:]:
        _chr_count_df = hic_extract_df[(hic_extract_df[1]==_chr) & (hic_extract_df[5]==_chr)] # subset cis-contact
        _chr_count_df = _chr_count_df[(_chr_count_df[0]==1)&(_chr_count_df[4]==1)] # subset contact?
        codebook_df_chr = codebook_df[codebook_df['chr']==_chr.split('chr')[1]]
        
        _mp_args.append((_chr_count_df, codebook_df_chr, _ext))
        
    import time
    _start_time = time.time()
    # init dict to save
    all_chr_contact_dict = {}
    
    # mp or sequentiall compute
    if parallel:
        import multiprocessing as mp
        print ('Multiprocessing for converting single cell hic matrix:')
        with mp.Pool(num_threads) as _mp_pool:
            _mp_results = _mp_pool.starmap(convert_count_to_merfish_matrix_byChr, 
                                           _mp_args, chunksize=1)
            _mp_pool.close()
            _mp_pool.join()
            _mp_pool.terminate()     
    else:
        print ('Looping for converting single cell hic matrix:')
        _mp_results = [convert_count_to_merfish_matrix_byChr(*_args) for _args in _mp_args]
    
    _mp_results_dict = {_chr: _chr_mtx for _chr, _chr_mtx in zip(all_sorted_chr, _mp_results)}

    print (f"Complete in {time.time()-_start_time:.3f}s.")
    return _mp_results_dict