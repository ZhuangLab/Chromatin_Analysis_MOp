import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')

## Define some global settings
_dpi = 300 # dpi required by figure
_single_col_width = 2.25 # figure width in inch if occupy 1 colomn
_double_col_width = 4.75 # figure width in inch if occupy 1 colomn
_single_row_height= 2 # comparable height to match single-colomn-width
_ticklabel_size=2
_ticklabel_width=0.5
_font_size=7.5

from matplotlib import cm
#from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns
sns.set_context("paper", rc={"font.size":_font_size,"axes.titlesize":_font_size+1,"axes.labelsize":_font_size})

def plot_cross_correlation_map_and_AB_compartments(correlation_map, norm_pc1, cell_type=None, chrom=None, save_fig=True, figure_file=None):
    fig, ax1 = plt.subplots(figsize=(_single_col_width,_single_col_width), dpi=200)
    # create a color map
    current_cmap = cm.get_cmap('seismic').copy()
    current_cmap.set_bad(color=[0.5,0.5,0.5,1])
    upper_indices = np.triu_indices(len(correlation_map),k=1)
    all_corr = correlation_map[upper_indices]
    bound = round(np.max([abs(np.percentile(all_corr,95)), abs(np.percentile(all_corr,5))]),1)
    vmin = -bound
    vmax = bound
    _pf = ax1.imshow(correlation_map, 
                     cmap=current_cmap, vmin=vmin, vmax=vmax)

    ax1.xaxis.set_tick_params(which='both', labelbottom=True)
    ax1.yaxis.set_tick_params(which='both', labelleft=True)
    ax1.set_title(f"AB comparments for {chrom} in {cell_type}", fontsize=_font_size)
    ax1.tick_params('both', labelsize=_font_size-1, 
                    width=_ticklabel_width, length=_ticklabel_size,
                    pad=1, labelleft=True, labelbottom=False) # remove bottom ticklabels for ax1
    [i[1].set_linewidth(_ticklabel_width) for i in ax1.spines.items()]
    # locate ax1
    divider = make_axes_locatable(ax1)
    # colorbar ax
    cax = divider.append_axes('right', size='6%', pad="2%")
    cbar = plt.colorbar(_pf,cax=cax, ax=ax1, ticks=[vmin, vmin/2, 0.0, vmax/2, vmax])
    cbar.ax.set_yticklabels([str(vmin), '', '0', '', str(vmax)])
    cbar.ax.tick_params('both', labelsize=_font_size-1, 
                    width=_ticklabel_width, length=_ticklabel_size-0.5,
                    pad=1, labelleft=False) # remove bottom ticklabels for ax1
    [i[1].set_linewidth(_ticklabel_width) for i in cbar.ax.spines.items()]
    #cbar.set_ticks([vmin,vmax])
    cbar.outline.set_linewidth(_ticklabel_width)
    cbar.set_label('Cross correlation', 
                   fontsize=_font_size, labelpad=2, rotation=270)
    cbar.ax.minorticks_off()

    # create bottom ax
    bot_ax = divider.append_axes('bottom', size='10%', pad="2%", 
                                 sharex=ax1, xticks=[])
    bot_ax.bar(np.where(norm_pc1>0)[0], norm_pc1[norm_pc1>0], color='r', width=1, bottom=0)
    bot_ax.bar(np.where(norm_pc1<=0)[0], norm_pc1[norm_pc1<=0],color='b', width=1, bottom=0)

    bot_ax.tick_params('x', labelsize=_font_size-1, 
                    width=_ticklabel_width, length=_ticklabel_size,
                    pad=1, labelleft=False, labelbottom=True) # remove bottom ticklabels for ax1
    bot_ax.tick_params('y', labelsize=_font_size-1, 
                    width=_ticklabel_width, length=0,
                    pad=1, labelleft=False, labelbottom=True) # remove bottom ticklabels for ax1
    bot_ax.set_ylabel('PC1', fontsize=_font_size-1, labelpad=1)

    # save
    if save_fig:
        plt.savefig(figure_file,
                    transparent=True, bbox_inches='tight', pad_inches=0.2, dpi=300)
        plt.savefig(figure_file.replace('.pdf', '.png'),
                    transparent=True, bbox_inches='tight', pad_inches=0.2, dpi=300)
    
    plt.show()






def plot_cross_correlation_map_and_AB_compartments_v2(correlation_map, norm_pc1, cell_type=None, chrom=None, pc_flag=None, save_fig=True, figure_file=None):
    fig, ax1 = plt.subplots(figsize=(_single_col_width,_single_col_width), dpi=200)
    # create a color map
    current_cmap = cm.get_cmap('seismic').copy()
    current_cmap.set_bad(color=[0.5,0.5,0.5,1])
    upper_indices = np.triu_indices(len(correlation_map),k=1)
    all_corr = correlation_map[upper_indices]
    bound = round(np.max([abs(np.percentile(all_corr,95)), abs(np.percentile(all_corr,5))]),1)
    vmin = -bound
    vmax = bound
    _pf = ax1.imshow(correlation_map, 
                     cmap=current_cmap, vmin=vmin, vmax=vmax)

    ax1.xaxis.set_tick_params(which='both', labelbottom=True)
    ax1.yaxis.set_tick_params(which='both', labelleft=True)
    ax1.set_title(f"AB comparments for {chrom} in {cell_type} ({pc_flag})", fontsize=_font_size)
    ax1.tick_params('both', labelsize=_font_size-1, 
                    width=_ticklabel_width, length=_ticklabel_size,
                    pad=1, labelleft=True, labelbottom=False) # remove bottom ticklabels for ax1
    [i[1].set_linewidth(_ticklabel_width) for i in ax1.spines.items()]
    # locate ax1
    divider = make_axes_locatable(ax1)
    # colorbar ax
    cax = divider.append_axes('right', size='6%', pad="2%")
    cbar = plt.colorbar(_pf,cax=cax, ax=ax1, ticks=[vmin, vmin/2, 0.0, vmax/2, vmax])
    cbar.ax.set_yticklabels([str(vmin), '', '0', '', str(vmax)])
    cbar.ax.tick_params('both', labelsize=_font_size-1, 
                    width=_ticklabel_width, length=_ticklabel_size-0.5,
                    pad=1, labelleft=False) # remove bottom ticklabels for ax1
    [i[1].set_linewidth(_ticklabel_width) for i in cbar.ax.spines.items()]
    #cbar.set_ticks([vmin,vmax])
    cbar.outline.set_linewidth(_ticklabel_width)
    cbar.set_label('Cross correlation', 
                   fontsize=_font_size, labelpad=2, rotation=270)
    cbar.ax.minorticks_off()

    # create bottom ax
    bot_ax = divider.append_axes('bottom', size='10%', pad="2%", 
                                 sharex=ax1, xticks=[])
    bot_ax.bar(np.where(norm_pc1>0)[0], norm_pc1[norm_pc1>0], color='r', width=1, bottom=0)
    bot_ax.bar(np.where(norm_pc1<=0)[0], norm_pc1[norm_pc1<=0],color='b', width=1, bottom=0)

    bot_ax.tick_params('x', labelsize=_font_size-1, 
                    width=_ticklabel_width, length=_ticklabel_size,
                    pad=1, labelleft=False, labelbottom=True) # remove bottom ticklabels for ax1
    bot_ax.tick_params('y', labelsize=_font_size-1, 
                    width=_ticklabel_width, length=0,
                    pad=1, labelleft=False, labelbottom=True) # remove bottom ticklabels for ax1
    bot_ax.set_ylabel('PC1', fontsize=_font_size-1, labelpad=1)

    # save
    if save_fig:
        plt.savefig(figure_file,
                    transparent=True, bbox_inches='tight', pad_inches=0.2, dpi=300)
        plt.savefig(figure_file.replace('.pdf', '.png'),
                    transparent=True, bbox_inches='tight', pad_inches=0.2, dpi=300)
    
    plt.show()







def plot_cross_correlation_map_and_AB_compartments_v3(correlation_map, norm_pc1, ref_measures, cell_type=None, chrom=None, pc_flag=None, save_fig=True, figure_file=None):
    fig, ax1 = plt.subplots(figsize=(_single_col_width,_single_col_width), dpi=200)
    # create a color map
    current_cmap = cm.get_cmap('seismic').copy()
    current_cmap.set_bad(color=[0.5,0.5,0.5,1])
    upper_indices = np.triu_indices(len(correlation_map),k=1)
    all_corr = correlation_map[upper_indices]
    bound = round(np.max([abs(np.percentile(all_corr,95)), abs(np.percentile(all_corr,5))]),1)
    vmin = -bound
    vmax = bound
    _pf = ax1.imshow(correlation_map, 
                     cmap=current_cmap, vmin=vmin, vmax=vmax)

    ax1.xaxis.set_tick_params(which='both', labelbottom=True)
    ax1.yaxis.set_tick_params(which='both', labelleft=True)
    ax1.set_title(f"AB comparments for {chrom} in {cell_type} ({pc_flag})", fontsize=_font_size)
    ax1.tick_params('both', labelsize=_font_size-1, 
                    width=_ticklabel_width, length=_ticklabel_size,
                    pad=1, labelleft=True, labelbottom=False) # remove bottom ticklabels for ax1
    [i[1].set_linewidth(_ticklabel_width) for i in ax1.spines.items()]
    # locate ax1
    divider = make_axes_locatable(ax1)
    # colorbar ax
    cax = divider.append_axes('right', size='6%', pad="2%")
    cbar = plt.colorbar(_pf,cax=cax, ax=ax1, ticks=[vmin, vmin/2, 0.0, vmax/2, vmax])
    cbar.ax.set_yticklabels([str(vmin), '', '0', '', str(vmax)])
    cbar.ax.tick_params('both', labelsize=_font_size-1, 
                    width=_ticklabel_width, length=_ticklabel_size-0.5,
                    pad=1, labelleft=False) # remove bottom ticklabels for ax1
    [i[1].set_linewidth(_ticklabel_width) for i in cbar.ax.spines.items()]
    #cbar.set_ticks([vmin,vmax])
    cbar.outline.set_linewidth(_ticklabel_width)
    cbar.set_label('Cross correlation', 
                   fontsize=_font_size, labelpad=2, rotation=270)
    cbar.ax.minorticks_off()

    # create bottom ax
    bot_ax = divider.append_axes('bottom', size='10%', pad="2%", 
                                 sharex=ax1, xticks=[])
    bot_ax.bar(np.where(norm_pc1>0)[0], norm_pc1[norm_pc1>0], color='r', width=1, bottom=0)
    bot_ax.bar(np.where(norm_pc1<=0)[0], norm_pc1[norm_pc1<=0],color='b', width=1, bottom=0)

    bot_ax.tick_params('x', labelsize=_font_size-1, 
                    width=_ticklabel_width, length=_ticklabel_size,
                    pad=1, labelleft=False, labelbottom=True) # remove bottom ticklabels for ax1
    bot_ax.tick_params('y', labelsize=_font_size-1, 
                    width=_ticklabel_width, length=0,
                    pad=1, labelleft=False, labelbottom=True) # remove bottom ticklabels for ax1
    bot_ax.set_ylabel('PC1', fontsize=_font_size-1, labelpad=1)
    bot_ax.set_ylim([-.1, .1])


    # create 2nd bottom ax
    bot_ax2 = divider.append_axes('bottom', size='10%', pad="5%", 
                                 sharex=bot_ax, xticks=[])
    bot_ax2.bar(np.where(norm_pc1>0)[0], ref_measures[norm_pc1>0], color='r', width=1, bottom=0)
    bot_ax2.bar(np.where(norm_pc1<=0)[0], ref_measures[norm_pc1<=0],color='b', width=1, bottom=0)

    bot_ax2.tick_params('x', labelsize=_font_size-1, 
                    width=_ticklabel_width, length=_ticklabel_size,
                    pad=1, labelleft=False, labelbottom=True) # remove bottom ticklabels for ax1
    bot_ax2.tick_params('y', labelsize=_font_size-1, 
                    width=_ticklabel_width, length=0,
                    pad=1, labelleft=False, labelbottom=True) # remove bottom ticklabels for ax1
    bot_ax2.set_ylabel('Ref', fontsize=_font_size-1, labelpad=1)
    bot_ax2.set_ylim([0.2, 1.2])
    bot_ax2.set_ylim([0.2, 1])

    # save
    if save_fig:
        plt.savefig(figure_file,
                    transparent=True, bbox_inches='tight', pad_inches=0.2, dpi=300)
        plt.savefig(figure_file.replace('.pdf', '.png'),
                    transparent=True, bbox_inches='tight', pad_inches=0.2, dpi=300)
    
    plt.show()