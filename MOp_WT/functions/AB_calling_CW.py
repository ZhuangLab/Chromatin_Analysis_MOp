import os
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.integrate import quad
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import copy

from matrix_plot_CW import plot_cross_correlation_map_and_AB_compartments, plot_cross_correlation_map_and_AB_compartments_v2


# define function for gaussian distribution
def _polymer_chain_model(r, s_2):
    return np.power(np.pi*s_2, -1/2)*np.power(np.e, -r**2/s_2)

def calculate_expected_contact_matrix(gene_dist_matrix, contact_threshold, lp=40*1e-3, tau=0.3*1e-3):
    # lp: persistence length of DNA in base pairs
    # tau: length scale factor: um per bp.
    s_2 = 2/3*lp*gene_dist_matrix*tau
    _integrate = np.ones(s_2.shape)
    for i in range(_integrate.shape[0]):
        for j in range(_integrate.shape[1]):
            if i!=j:
                _integrate[i,j], _ = quad(_polymer_chain_model, -contact_threshold, contact_threshold, args=s_2[i,j])

def generate_cross_correlation_map(dist_mat):
    dimension = dist_mat.shape[0]
    correlation_map = np.ones(dist_mat.shape)
    for i in range(dimension):
        for j in range(dimension):
            if i!=j:
                x1 = dist_mat[i,:]
                x2 = dist_mat[:,j].transpose()
                mask = (~np.isnan(x1))&(~np.isnan(x2))
                p, r = pearsonr(x1[mask], x2[mask])
                correlation_map[i,j] = p

    return correlation_map


def generate_cross_correlation_map_v2(dist_mat, min_points = 50):
    dimension = dist_mat.shape[0]
    correlation_map = np.ones(dist_mat.shape)
    for i in range(dimension):
        for j in range(dimension):
            if i!=j:
                x1 = dist_mat[i,:]
                x2 = dist_mat[:,j].transpose()
                mask = (~np.isnan(x1))&(~np.isnan(x2))
                if np.sum(mask)>min_points:
                    p, r = pearsonr(x1[mask], x2[mask])
                    correlation_map[i,j] = p
                else:
                    correlation_map[i,j] = np.nan
    return correlation_map

# deprecated
# this function aimed at only calculating cross correlation for distant loci
def generate_cross_correlation_map_by_long_range(dist_mat, gene_dist, gene_dist_threshold=1e7):
    dimension = dist_mat.shape[0]
    correlation_map = np.zeros(dist_mat.shape)
    for i in range(dimension):
        for j in range(dimension):
            x = dist_mat[i,:]
            y = dist_mat[:,j]
            long_for_x = np.where(gene_dist[i,:]>=gene_dist_threshold)[0]
            long_for_y = np.where(gene_dist[:,j]>=gene_dist_threshold)[0]
            mask = np.intersect1d(long_for_x, long_for_y)
            x = x[mask]
            y = y[mask]
            p, r = spearmanr(x, y)
            correlation_map[i,j] = p
    return correlation_map

def call_AB_compartments(observed_contact_matrix, expected_contact_matrix, mCG_levels,
               cell_type=None, chrom=None, save_fig=False, figure_folder=None, output_all_info=True):
    '''    
    # expected contact matrix is calculated by calculating the CDF of the Gaussian distribution for the DNA polymer model
    # mCG_levels: 1D np array sorted by hyb values based on each chromosomes
    '''
    # normalize matrix for normalization
    matrix_for_normalization = np.copy(expected_contact_matrix)
    for i in range(matrix_for_normalization.shape[0]):
        for j in range(matrix_for_normalization.shape[1]):
            prob_real = np.nansum(observed_contact_matrix[i,:])+np.nansum(observed_contact_matrix[:,j])
            prob_norm = np.nansum(expected_contact_matrix[i,:])+np.nansum(expected_contact_matrix[:,j])
            matrix_for_normalization[i,j] = matrix_for_normalization[i,j]*prob_real/prob_norm
    
    contact_matrix = observed_contact_matrix / matrix_for_normalization
    
    # for trouble shooting
    #return contact_matrix, observed_contact_matrix, matrix_for_normalization
    
    contact_matrix[contact_matrix>=2] = 2
    correlation_map = generate_cross_correlation_map(contact_matrix)
    pca = PCA(3)
    pca.fit(correlation_map)
    pc1 = pca.components_[0]
    norm_pc1 = pc1/np.std(pc1)
    
    # calculate the sign based on bulk mCG levels
    positive_pc1_mCG = np.average(mCG_levels[norm_pc1>0])
    negative_pc1_mCG = np.average(mCG_levels[norm_pc1<0])
    if negative_pc1_mCG<positive_pc1_mCG:
        norm_pc1 *= -1

    if save_fig:
        figure_folder_for_chrom = os.path.join(figure_folder, chrom)
        if not os.path.exists(figure_folder_for_chrom):
            os.mkdir(figure_folder_for_chrom)
        figure_file = os.path.join(figure_folder_for_chrom, f'{cell_type}_{chrom}_AB_map.pdf')
        plot_cross_correlation_map_and_AB_compartments(correlation_map, norm_pc1, cell_type, chrom, save_fig=True, figure_file=figure_file)

    if output_all_info:
        return (cell_type, chrom, norm_pc1)
    else:
        return norm_pc1
    




def call_AB_compartments_v2(observed_contact_matrix, expected_contact_matrix, mCG_levels,
               cell_type=None, chrom=None, save_fig=False, figure_folder=None, output_all_info=True, return_corr_map = True):
    '''    
    # expected contact matrix is calculated by calculating the CDF of the Gaussian distribution for the DNA polymer model
    # mCG_levels: 1D np array sorted by hyb values based on each chromosomes
    '''
    # normalize matrix for normalization
    matrix_for_normalization = np.copy(expected_contact_matrix)
    for i in range(matrix_for_normalization.shape[0]):
        for j in range(matrix_for_normalization.shape[1]):
            prob_real = np.nansum(observed_contact_matrix[i,:])+np.nansum(observed_contact_matrix[:,j])
            prob_norm = np.nansum(expected_contact_matrix[i,:])+np.nansum(expected_contact_matrix[:,j])
            matrix_for_normalization[i,j] = matrix_for_normalization[i,j]*prob_real/prob_norm
    
    contact_matrix = observed_contact_matrix / matrix_for_normalization
    
    # for trouble shooting
    #return contact_matrix, observed_contact_matrix, matrix_for_normalization
    
    contact_matrix[contact_matrix>=2] = 2
    correlation_map = generate_cross_correlation_map(contact_matrix)
    pca = PCA(3)
    pca.fit(correlation_map)
    pc1 = pca.components_[0]
    norm_pc1 = pc1/np.std(pc1)


    pc2 = pca.components_[1]
    norm_pc2 = pc2/np.std(pc2)
    
    # calculate the sign based on bulk (cell-type) mCG levels (or other modality)
    from scipy import stats
    res_pc1 = stats.spearmanr(norm_pc1,mCG_levels)
    res_pc2 = stats.spearmanr(norm_pc2,mCG_levels)
    for res_corr, _norm_pc in zip([res_pc1[0], res_pc2[0]], [norm_pc1,norm_pc2]):
        if res_corr <0:
            _norm_pc *= -1
        
    # use the pc with highest correlation to plot figure
    #if abs(res_pc1[0])>=abs(res_pc2[0]):
        #pc_flag = f'PC1 p={round(res_pc1[0],3)}'
        #sel_norm_pc = norm_pc1
    #else:
        #pc_flag = f'PC2 p={round(res_pc2[0],3)}'
        #sel_norm_pc = norm_pc2
    pc1_corr = f'PC1 p={round(res_pc1[0],2)}'
    pc2_corr = f'PC2 p={round(res_pc2[0],2)}'
    

    if save_fig:
        figure_folder_for_chrom = os.path.join(figure_folder, chrom)
        if not os.path.exists(figure_folder_for_chrom):
            os.mkdir(figure_folder_for_chrom)
        for _idx, (_norm_pc, _pc_corr) in enumerate(zip([norm_pc1,norm_pc2],[pc1_corr,pc2_corr])):
            figure_file = os.path.join(figure_folder_for_chrom, f'{cell_type}_{chrom}_AB_map_PC{_idx+1}.pdf')
            plot_cross_correlation_map_and_AB_compartments_v2(correlation_map, _norm_pc, cell_type, chrom, _pc_corr, save_fig=True, figure_file=figure_file)

    if return_corr_map:
        if output_all_info:
            return (cell_type, chrom, norm_pc1, norm_pc2, correlation_map)
        else:
            return norm_pc1, norm_pc2, correlation_map
    else:
        if output_all_info:
            return (cell_type, chrom, norm_pc1, norm_pc2, )
        else:
            return norm_pc1, norm_pc2
        






def call_AB_compartments_v3(observed_contact_matrix, expected_contact_matrix, mCG_levels,
               cell_type=None, chrom=None, save_fig=False, figure_folder=None,):
    '''    
    # expected contact matrix is calculated by calculating the CDF of the Gaussian distribution for the DNA polymer model
    # mCG_levels: 1D np array sorted by hyb values based on each chromosomes
    '''
    # normalize matrix for normalization
    matrix_for_normalization = np.copy(expected_contact_matrix)
    for i in range(matrix_for_normalization.shape[0]):
        for j in range(matrix_for_normalization.shape[1]):
            prob_real = np.nansum(observed_contact_matrix[i,:])+np.nansum(observed_contact_matrix[:,j])
            prob_norm = np.nansum(expected_contact_matrix[i,:])+np.nansum(expected_contact_matrix[:,j])
            matrix_for_normalization[i,j] = matrix_for_normalization[i,j]*prob_real/prob_norm
    
    contact_matrix = observed_contact_matrix / matrix_for_normalization
    
    # for trouble shooting
    #return contact_matrix, observed_contact_matrix, matrix_for_normalization
    
    contact_matrix[contact_matrix>=2] = 2
    correlation_map = generate_cross_correlation_map(contact_matrix)
    pca = PCA(3)
    pca.fit(correlation_map)
    pc1 = pca.components_[0]
    norm_pc1 = pc1/np.std(pc1)

    pc2 = pca.components_[1]
    norm_pc2 = pc2/np.std(pc2)

    pc3 = pca.components_[2]
    norm_pc3 = pc3/np.std(pc3)
    
    # calculate the sign based on bulk (cell-type) mCG levels (or other modality)
    from scipy import stats
    res_pc1 = stats.spearmanr(norm_pc1,mCG_levels)
    res_pc2 = stats.spearmanr(norm_pc2,mCG_levels)
    res_pc3 = stats.spearmanr(norm_pc3,mCG_levels)
    for res_corr, _norm_pc in zip([res_pc1[0], res_pc2[0],res_pc2[1]], [norm_pc1,norm_pc2,norm_pc3]):
        if res_corr <0:
            _norm_pc *= -1
        
    # use the pc with highest correlation to plot figure
    #if abs(res_pc1[0])>=abs(res_pc2[0]):
        #pc_flag = f'PC1 p={round(res_pc1[0],3)}'
        #sel_norm_pc = norm_pc1
    #else:
        #pc_flag = f'PC2 p={round(res_pc2[0],3)}'
        #sel_norm_pc = norm_pc2
    pc1_corr = f'PC1 p={round(res_pc1[0],2)}'
    pc2_corr = f'PC2 p={round(res_pc2[0],2)}'
    pc3_corr = f'PC3 p={round(res_pc3[0],2)}'
    
    pca_variance_ratio = pca.explained_variance_ratio_

    if save_fig:
        figure_folder_for_chrom = os.path.join(figure_folder, chrom)
        if not os.path.exists(figure_folder_for_chrom):
            os.mkdir(figure_folder_for_chrom)
        for _idx, (_norm_pc, _pc_corr) in enumerate(zip([norm_pc1,norm_pc2,norm_pc3],[pc1_corr,pc2_corr,pc3_corr])):
            figure_file = os.path.join(figure_folder_for_chrom, f'{cell_type}_{chrom}_AB_map_PC{_idx+1}.pdf')
            plot_cross_correlation_map_and_AB_compartments_v2(correlation_map, _norm_pc, cell_type, chrom, _pc_corr, save_fig=True, figure_file=figure_file)


    return {'celltype': cell_type, 'chrom':chrom, 
        'norm_pc1':norm_pc1, 'norm_pc2':norm_pc2, 'norm_pc3':norm_pc3, 'pca_explained_variance_ratio':pca_variance_ratio, 'correlation_map':correlation_map}
