#collection of functions for different plot types to improve constancy.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib


_markers = ['o', '^', 's', 'd', 'v', '<', '>', 'x', '+']
_linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
_colours  = plt.cm.Dark2

def multi_lines_scatter(x=None, ys=None, ax=None, labels=None, scale='linear', 
                        log_scale_base=2, num_grid_lines=5, errors=[], 
                        alpha=0.5, ax_labels=None, leg=True, m_size=20, 
                        line=True, err_type='cont'):
    
    
    #pad lable if none so can loop over zip 
    if not labels:
        labels = [labels] * len(ys)
    #chose colors used for plot
    colours = _colours(np.arange(0, len(ys)))
    
    #scatter plot of the y values and labels
    if len(errors) > 0:
        for i, (y, label, error) in enumerate(zip(ys, labels, errors)):
            '''
            if len(ys) < 4:
                linestyle = _linestyles[i]
            else:
                 linestyle = _linestyles[0]
            '''
            linestyle = _linestyles[0]
            if line and err_type != 'bar':
                ax.plot(x, y, label=label, marker=_markers[i], 
                        linestyle=linestyle, color=colours[i])
            #error bar plot
            elif err_type == 'bar':
                ax.errorbar(x, y, yerr=error, label=label, marker=_markers[i], 
                            s=m_size, color=colours[i], capsize=2)
              
            else:
                ax.scatter(x, y, label=label, marker=_markers[i], 
                           s=m_size, color=colours[i])
                
            #add contous errors around plot  
            if err_type == 'cont':   
                ax.fill_between(x, y - error, y + error, alpha=alpha,
                                color=colours[i])
                
                
    else:
        for i, (y, label) in enumerate((ys, labels)):
            if len(ys) < 4:
                linestyle = _linestyles[i]
            else:
                linestyle = linestyle[0]
            ax.scatter(x, y, label=label, marker=_markers[i], linestyle=linestyle)
            
    #change the scale     
    if scale == 'log':
        plt.xscale('log', base=log_scale_base)
        plt.yscale('log', base=log_scale_base)
    #add grid lines 
    if num_grid_lines:
        ymin, ymax = np.log2(ax.get_ylim())
        yticks = np.logspace(ymin, ymax, base=log_scale_base, 
                             num=num_grid_lines)
        ax.set_yticks(yticks)
        if labels[0] and leg:
            pass
            ax.legend(fontsize=8)
        plt.grid()
        
    if ax_labels:
        if ax_labels[0]:
            ax.set_xlabel(ax_labels[0])
        if ax_labels[1]:
            ax.set_ylabel(ax_labels[1])

