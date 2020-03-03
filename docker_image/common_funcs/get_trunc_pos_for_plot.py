from matplotlib.lines import Line2D

def get_trunc_pos_arrs():
    pos_x_arrs = []
    pos_z_arrs = []
    
    #region 1
    #x_l = 4100.0/1000.0
    #z_t = 1060.0/1000.0
    #w   =  740.0/1000.0
    #h   =  500.0/1000.0 
    
    #clockwise, start top left
    #pos_x_arr = [x_l, x_l + w, x_l + w, x_l    , x_l]
    #pos_z_arr = [z_t, z_t    , z_t + h, z_t + h, z_t]
    #pos_x_arrs.append(pos_x_arr)
    #pos_z_arrs.append(pos_z_arr)

    #region 2
    x_l = 5940.0/1000.0
    z_t = 1860.0/1000.0
    w   =  860.0/1000.0
    h   =  480.0/1000.0 
    
    #clockwise, start top left
    pos_x_arr = [x_l, x_l + w, x_l + w, x_l    , x_l]
    pos_z_arr = [z_t, z_t    , z_t + h, z_t + h, z_t]
    pos_x_arrs.append(pos_x_arr)
    pos_z_arrs.append(pos_z_arr)
    
    #region 3
    #x_l = 5500.0/1000.0
    #z_t = 2760.0/1000.0
    #w   = 2800.0/1000.0
    #h   =  540.0/1000.0 
    
    #clockwise, start top left
    #pos_x_arr = [x_l, x_l + w, x_l + w, x_l    , x_l]
    #pos_z_arr = [z_t, z_t    , z_t + h, z_t + h, z_t]
    #pos_x_arrs.append(pos_x_arr)
    #pos_z_arrs.append(pos_z_arr)
    
    return pos_x_arrs, pos_z_arrs  

def plot_trunc_lines(fig, lw, c):
    pos_x_arrs, pos_z_arrs = get_trunc_pos_arrs()

    boundary_lines_1 = Line2D(pos_x_arrs[0], pos_z_arrs[0], linewidth = lw, color = c)
    #boundary_lines_2 = Line2D(pos_x_arrs[1], pos_z_arrs[1], linewidth = lw, color = c)
    #boundary_lines_3 = Line2D(pos_x_arrs[2], pos_z_arrs[2], linewidth = lw, color = c)
    fig.axes[0].add_line(boundary_lines_1)
    #fig.axes[0].add_line(boundary_lines_2)
    #fig.axes[0].add_line(boundary_lines_3)    
  
def plot_zoom_box(fig, lw, c, x_l, x_r, z_t, z_b):
    line_t = Line2D([x_l, x_r], [z_t, z_t], linewidth = lw, color = c)
    line_r = Line2D([x_r, x_r], [z_t, z_b], linewidth = lw, color = c)
    line_b = Line2D([x_r, x_l], [z_b, z_b], linewidth = lw, color = c)
    line_l = Line2D([x_l, x_l], [z_b, z_t], linewidth = lw, color = c)
    
    fig.axes[0].add_line(line_t)
    fig.axes[0].add_line(line_r)
    fig.axes[0].add_line(line_b)
    fig.axes[0].add_line(line_l)