import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import collections
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import AxesGrid

def imshow_func(fig_nr, arr_2d, x_min, x_max, z_min, z_max, x_label, z_label, title, cbar_min=None, cbar_max=None, cbar_format = None, cbar_title = None, ftsize = 8, savename=None, figsize = None, cmap = None, savedpi=300.0, align_cbar_vals_right = False, x_align_val_cbar = None, aspect_ratio = 1.0):
    #INSTEAD OF SO MANY KEYWORDS I SHOULD JUST PASS A DICTIONARY. AT THE START OF THIS FUNCTION I WOULD DEFINE A DEFAULT DICTIONARY. THEN I UPDATE THE DEFAULT DICTIONARY WITH ANY KEYWORDS THAT ARE IN THE DICTIONARY THAT IS PASSED TO THE FUNCTION. THAT WAY DEFAULT VALUES ARE OVERWRITTEN.    
    if figsize != None:
        fig = plt.figure(fig_nr, figsize = figsize)
    else:
        fig = plt.figure(fig_nr)

    ax = fig.add_subplot(111)
    if cmap == None:
        im = ax.imshow(arr_2d, extent=[x_min,x_max,z_max,z_min], interpolation="nearest", aspect = aspect_ratio)
    else:
        im = ax.imshow(arr_2d, extent=[x_min,x_max,z_max,z_min], interpolation="nearest", cmap = cmap, aspect = aspect_ratio)
        
    im.axes.yaxis.set_label_text(z_label, fontsize = ftsize)
    im.axes.xaxis.set_label_text(x_label, fontsize = ftsize)
    im.axes.set_title(title, fontsize = ftsize)

    if cbar_min != None and cbar_max != None: #ALWAYS USE cbar_min and cbar_max for norm, if available even if not plotting the cbar itself
        norm =  mpl.colors.Normalize(vmin=cbar_min, vmax=cbar_max)
        im.set_norm(norm)

    if cbar_title is not None: #Plot the cbar, if title is given
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05) #http://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        
        if cbar_min !=None and cbar_max !=None:
            cb = plt.colorbar(im, cax = cax, ticks=np.linspace(cbar_min, cbar_max, 5), format = cbar_format)
        else:
            cb = plt.colorbar(im, cax = cax, format = cbar_format)
        
        cb.ax.tick_params(labelsize = ftsize)
        cb.set_label(cbar_title, rotation=90, fontsize=ftsize)
                
        for t in cb.ax.get_yticklabels():
            if align_cbar_vals_right:
                t.set_horizontalalignment('right')    
            if x_align_val_cbar != None:
                t.set_x(x_align_val_cbar)

    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(ftsize)
        
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(ftsize)

    if savename != None:
        fig_dir = "figdir/"
        fig.savefig(fig_dir + savename, dpi=savedpi)
    
    return fig

def plot_lines_nicely(fig_nr, figsize, xvals, list_of_1d_arrays, list_of_legend_texts, color_list = None,  marker_style = None, marker_size = 1, x_label = "", y_label = "", title = "", fontsize = 8, line_style = '-', lw = 1, leg_args = {'loc':1}, xlim_l = None, xlim_r = None, ylim_l = None, ylim_r = None, rot_val = 45):
    ###
    #list_of_legend_texts should have None for entries that should not have a label
    #marker_style and marker_size, lw and linestyle can be lists
    ###
    
    def make_iterable(val, n):
        val = [val for i in xrange(n)]
        return val
    
    n_lines = len(list_of_1d_arrays)

    if not isinstance(marker_style, collections.Iterable) or type(marker_style) == str: #make iterable by repeating non iterable value 
        marker_style = make_iterable(marker_style, n_lines)
        
    if not isinstance(marker_size , collections.Iterable) or type(marker_size) == str: #make iterable by repeating non iterable value
        marker_size = make_iterable(marker_size, n_lines)

    if not isinstance(lw , collections.Iterable) or type(lw) == str: #make iterable by repeating non iterable value
        lw = make_iterable(lw, n_lines)

    if not isinstance(line_style , collections.Iterable) or type(line_style) == str: #make iterable by repeating non iterable value
        line_style = make_iterable(line_style, n_lines)
    
    fig = plt.figure(fig_nr, figsize=figsize)
    ax  = fig.add_subplot(111)
    
    if color_list:
        for arr_1d, legend_text, color, marker_st, marker_sz, l_w, ls in zip(list_of_1d_arrays, list_of_legend_texts, color_list, marker_style, marker_size, lw, line_style):
            ax.plot(xvals, arr_1d, color = color, linestyle = ls, marker=marker_st, markersize=marker_sz, linewidth=l_w, label=legend_text  )
    else:
        for arr_1d, legend_text, marker_st, marker_sz, l_w, ls in zip(list_of_1d_arrays, list_of_legend_texts, marker_style, marker_size, lw, line_style):
            ax.plot(xvals, arr_1d, linestyle = ls, marker=marker_st, markersize=marker_sz, linewidth=l_w, label=legend_text  )   

    plt.legend(fontsize=fontsize, **leg_args)
    plt.title(title, fontsize = fontsize)
    plt.xlabel(x_label, fontsize = fontsize)
    plt.ylabel(y_label, fontsize = fontsize)

    if xlim_l != None and xlim_r != None:
        plt.xlim(xlim_l, xlim_r)
    elif xlim_l != None:
        plt.xlim(xlim_l, np.max(xvals))
    elif xlim_r != None:
        plt.xlim(np.min(xvals), xlim_r)
    elif not xlim_l and not xlim_r:    
        pass #Change nothing
    else:
        raise Exception("???")

    if type(ylim_l) != type(None) and type(ylim_r) != type(None):
        plt.ylim(ylim_l, ylim_r)
    elif type(ylim_l) == type(None) and type(ylim_r) == type(None):
        pass #Change nothing
    else:
        raise Exception("no fancier thing implemented yet")


    for label in ax.xaxis.get_ticklabels(): #some axes are shared so this will do some extra work
            label.set_color('black')
            label.set_rotation(rot_val)
            label.set_fontsize(fontsize)


    for label in ax.yaxis.get_ticklabels(): #some axes are shared so this will do some extra work
        label.set_color('black')
        label.set_fontsize(fontsize)

    return fig

def plot_in_grid(fig_nr, list_of_2d_ndarrays, list_of_titles, n_rows, n_cols, extents, cbar_title = None, cbar_min = None, cbar_max = None, cbar_mode = 'single', cbar_format = None, ftsize = 8, figsize = None, cmap = None, x_align_val_cbar = 3.0, axes_pad = 0.3, xlabel = 'Horizontal coordinate (km)', ylabel = 'Depth (km)', create_fig = True, retfig = True, aspect_ratio = 1.0, nticks_y = 5):
    #INSTEAD OF SO MANY KEYWORDS I SHOULD JUST PASS A DICTIONARY. AT THE START OF THIS FUNCTION I WOULD DEFINE A DEFAULT DICTIONARY. THEN I UPDATE THE DEFAULT DICTIONARY WITH ANY KEYWORDS THAT ARE IN THE DICTIONARY THAT IS PASSED TO THE FUNCTION. THAT WAY DEFAULT VALUES ARE OVERWRITTEN.
    
    #fig = plt.figure(fignr, figsize = (fig_width_3, fig_height_3))
    x_min = extents['x_min']
    x_max = extents['x_max']
    z_min = extents['z_min']
    z_max = extents['z_max']
    
    if create_fig:
        if figsize != None:
            fig = plt.figure(fig_nr, figsize = figsize)
        else:
            fig = plt.figure(fig_nr)
    else:
        fig = plt.gcf()
    
    grid = AxesGrid(fig, 111, # similar to subplot(111)
    nrows_ncols = (n_rows, n_cols), # creates 3x1 grid of axes
    axes_pad=axes_pad, # pad between axes in inch.
    share_all=True,
    cbar_location="right",
    cbar_pad = "0%",
    cbar_size = "3%",
    cbar_mode=cbar_mode,
    aspect = True
    )
    
    ims = []
    nims = len(list_of_2d_ndarrays)
    for i in xrange(nims):
        arr_2d = list_of_2d_ndarrays[i]
        if arr_2d is not None: #Else skip this i. Can be used for layout purposes
            if cmap == None:
                im = grid[i].imshow(arr_2d, extent=[x_min,x_max,z_max,z_min], interpolation="nearest", aspect = aspect_ratio)
            else:
                im = grid[i].imshow(arr_2d, extent=[x_min,x_max,z_max,z_min], interpolation="nearest", cmap = cmap, aspect = aspect_ratio)
            ims.append(im)
        else:
            ims.append(None)
 
    for i in xrange(nims):
        if list_of_2d_ndarrays[i] is not None: #Else skip this i. Can be used for layout purposes
            grid[i].axes.yaxis.set_label_text(ylabel, fontsize = ftsize)
            grid[i].locator_params(axis = 'y', nbins = nticks_y)
    
    grid[-1].axes.xaxis.set_label_text(xlabel, fontsize = ftsize)
       
    for ax in grid:
        for label in ax.xaxis.get_ticklabels(): #some axes are shared so this will do some extra work
            label.set_color('black')
            #label.set_rotation(45)
            label.set_fontsize(ftsize)
    
    for ax in grid:
        for label in ax.yaxis.get_ticklabels(): #some axes are shared so this will do some extra work
            label.set_color('black')
            label.set_fontsize(ftsize)
    
    if type(cbar_min) == list and type(cbar_max) == list: #list of cbar_min and cbar_max
        norm = []
        for cbar_min_val, cbar_max_val in zip(cbar_min,cbar_max):
            norm.append(mpl.colors.Normalize(vmin=cbar_min_val, vmax=cbar_max_val))
    elif cbar_min !=None and cbar_max !=None:
        norm =  mpl.colors.Normalize(vmin=cbar_min, vmax=cbar_max)            
    else:
        cbar_min =  np.inf
        cbar_max = -np.inf
        for i in xrange(nims):
            if list_of_2d_ndarrays[i] is not None: #Else skip this i. Can be used for layout purposes
                arr_2d = list_of_2d_ndarrays[i]
                cbar_min = min(np.min(arr_2d), cbar_min)
                cbar_max = max(np.max(arr_2d), cbar_max) 
        
        norm =  mpl.colors.Normalize(vmin=cbar_min, vmax=cbar_max)            

    cb_list = []        
    for i in xrange(nims):
        if list_of_2d_ndarrays[i] is not None: #Else skip this i. Can be used for layout purposes
            im = ims[i]    
            im.axes.set_title(list_of_titles[i], fontsize = ftsize, color='black')
            if type(norm) == list: #This conditional programming based on type is not very pythonic...
                im.set_norm(norm[i])
                cb = grid.cbar_axes[i].colorbar(im, ticks=np.linspace(cbar_min[i], cbar_max[i], 5), format = cbar_format) #The single color bar
            else:
                im.set_norm(norm)    
                cb = grid.cbar_axes[i].colorbar(im, ticks=np.linspace(cbar_min, cbar_max, 5), format = cbar_format) #The single color bar
                
            cb.ax.tick_params(labelsize = ftsize)
            if cbar_title == None:
                cbar_title_im = 'Velocity'
            elif type(cbar_title) == list:
                cbar_title_im = cbar_title[i]
            else: #if single value
                cbar_title_im = cbar_title
                
            cb.set_label_text(cbar_title_im, rotation=90, fontsize=ftsize)
            for t in cb.ax.get_yticklabels():
                t.set_horizontalalignment('right')
                t.set_x(x_align_val_cbar)
    
            cb_list.append(cb)
    
    if retfig:
        ret = fig
    else:
        ret = {'grid': grid, 'ims': ims}
        
    return ret