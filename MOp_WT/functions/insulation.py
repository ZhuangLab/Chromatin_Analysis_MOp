
# adpated script from PZ's IA3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import normaltest, ks_2samp, ttest_ind
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree, is_valid_linkage

# Adpated from PZ sliding_window_dist; Calculate distance given distance_matrix, window_size and metric type
# but clip the edge where w exceeds the matrix and use the method from the orginal Crane 2015
def sliding_window_insulation (_mat, _wd=20, _dist_metric='normed_insulation'):
    """Function to calculate sliding-window distance across one distance-map of chromosome"""
    dists = np.zeros(len(_mat))
    for _i in range(len(_mat)):
        if _i - int(_wd/2) < 0 or _i + int(_wd/2) >= len(_mat):
            dists[_i] = np.nan
        else:
            # get slices
            _left_slice = slice(max(0, _i-_wd), _i)
            _right_slice = slice(_i, min(_i+_wd, len(_mat)))
            # slice matrix
            _intra1 = np.triu(_mat[_left_slice,_left_slice], 1)
            _intra1 = _intra1[np.isnan(_intra1)==False]
            _intra2 = np.triu(_mat[_right_slice,_right_slice], 1)
            _intra2 = _intra2[np.isnan(_intra2)==False]
            _intra_dist = np.concatenate([_intra1[_intra1 > 0],
                                          _intra2[_intra2 > 0]])
            _inter_dist = _mat[_left_slice,_right_slice]
            _inter_dist = _inter_dist[np.isnan(_inter_dist) == False]
            _inter_dist = _inter_dist[_inter_dist>0]
            if len(_intra_dist) == 0 or len(_inter_dist) == 0:
                # return zero distance if one dist list is empty
                dists[_i] = 0
                continue
            # add dist info
            if _dist_metric == 'normed_insulation':
                dists[_i] = (np.nanmean(_intra_dist) - np.nanmean(_inter_dist)) / (np.nanmean(_intra_dist) + np.nanmean(_inter_dist))
            elif _dist_metric == 'insulation':
                m_inter, m_intra = np.nanmean(_inter_dist), np.nanmean(_intra_dist)
                dists[_i] = m_intra / m_inter 
            elif _dist_metric == 'hic_insulation':
                _inter_mat = _mat[_i - int(_wd/2): _i + int(_wd/2), _i - int(_wd/2): _i + int(_wd/2)]
                _inter_mat = np.triu(_inter_mat, 1)
                _inter_mat[_inter_mat == 0] = np.nan
                dists[_i] = np.nanmean(_inter_mat)
            else:
                raise ValueError(f"Wrong input _dist_metric")

    mean_dist = np.nanmean(dists)
    norm_dists = np.log2(dists/mean_dist)

    return norm_dists



# use a gene dist wd to slide rather than loci bin wd
def sliding_window_insulation_by_gene_dist (_mat, _gene_dists, _wd_gene_dist=30000000, _dist_metric='normed_insulation'):
    """Function to calculate sliding-window distance across one distance-map of chromosome"""
    dists = np.zeros(len(_mat))
    for _i in range(len(_mat)):
        slice_start = _gene_dists[_i] - int(_wd_gene_dist/2)
        slice_end = _gene_dists[_i] + int(_wd_gene_dist/2)
        # assign nan to ignore the edges
        if slice_start < _gene_dists[0] or slice_end >= _gene_dists[-1]:
            dists[_i] = np.nan
        else:
            mat_dist_inds = np.where((_gene_dists>slice_start)&(_gene_dists<slice_end))[0]
            # get slices
            _left_slice = slice(max(0, min(mat_dist_inds)), _i)
            _right_slice = slice(_i, min(max(mat_dist_inds), len(_mat)))
            # slice matrix
            _intra1 = np.triu(_mat[_left_slice,_left_slice], 1)
            _intra1 = _intra1[np.isnan(_intra1)==False]
            _intra2 = np.triu(_mat[_right_slice,_right_slice], 1)
            _intra2 = _intra2[np.isnan(_intra2)==False]
            _intra_dist = np.concatenate([_intra1[_intra1 > 0],
                                          _intra2[_intra2 > 0]])
            _inter_dist = _mat[_left_slice,_right_slice]
            _inter_dist = _inter_dist[np.isnan(_inter_dist) == False]
            _inter_dist = _inter_dist[_inter_dist>0]
            if len(_intra_dist) == 0 or len(_inter_dist) == 0:
                # return zero distance if one dist list is empty
                dists[_i] = 0
                continue
            # add dist info
            if _dist_metric == 'normed_insulation':
                dists[_i] = (np.nanmean(_intra_dist) - np.nanmean(_inter_dist)) / (np.nanmean(_intra_dist) + np.nanmean(_inter_dist))
            elif _dist_metric == 'insulation':
                m_inter, m_intra = np.nanmean(_inter_dist), np.nanmean(_intra_dist)
                dists[_i] = m_intra / m_inter 
            elif _dist_metric == 'hic_insulation':
                _inter_mat = _mat[mat_dist_inds,:][:,mat_dist_inds]
                _inter_mat = np.triu(_inter_mat, 1)
                _inter_mat[_inter_mat == 0] = np.nan
                dists[_i] = np.nanmean(_inter_mat)
            else:
                raise ValueError(f"Wrong input _dist_metric")

    mean_dist = np.nanmean(dists)
    norm_dists = np.log2(dists/mean_dist)

    return norm_dists