from matplotlib.lines import Line2D

def get_trunc_pos_arrs():
    pos_x_arrs = []
    pos_z_arrs = []
    
    dx = 20.0 / 1000.0
    dz = 20.0 / 1000.0
    
    #############################################
    #bot perturb (start top left and go clockwise)
    #############################################
    #pos_x_arr = []
    #pos_z_arr = []
    
    #top
    #pos_x_arr.extend([312*dx, 359*dx])
    #pos_z_arr.extend([145*dz, 145*dz])
    
    #right side
    #pos_x_arr.extend([377*dx, 386*dx, 393*dx, 399*dx])
    #pos_z_arr.extend([146*dz, 147*dz, 148*dz, 149*dz])
    
    #bot
    #pos_x_arr.extend([287*dx])
    #pos_z_arr.extend([149*dz])
    
    #left
    #pos_x_arr.extend([291*dx, 296*dx, 303*dx, 312*dx])
    #pos_z_arr.extend([148*dz, 147*dz, 146*dz, 145*dz])
    
    #pos_x_arrs.append(pos_x_arr)
    #pos_z_arrs.append(pos_z_arr)
    
    #############################################
    #top perturb (start top left and go clockwise)
    #############################################
    #pos_x_arr = []
    #pos_z_arr = []
    
    #top
    #pos_x_arr.extend([229*dx, 230*dx])
    #pos_z_arr.extend([ 58*dz,  58*dz])    
    
    #right
    #pos_x_arr.extend([231*dx, 232*dx, 233*dx, 232*dx, 231*dx, 230*dx, 229*dx, 228*dx, 227*dx, 225*dx, 224*dx])
    #pos_z_arr.extend([ 59*dz,  60*dz,  61*dz,  62*dz,  63*dz,  64*dz,  65*dz,  66*dz,  67*dz,  68*dz,  69*dz])
              
    #bot
    #pos_x_arr.extend([215*dx])
    #pos_z_arr.extend([ 69*dz])
    
    #left
    #pos_x_arr.extend([217*dx, 218*dx, 220*dx, 221*dx, 222*dx, 223*dx, 225*dx, 226*dx, 227*dx, 228*dx, 229*dx])
    #pos_z_arr.extend([ 68*dz,  67*dz,  66*dz,  65*dz,  64*dz,  63*dz,  62*dz,  61*dz,  60*dz,  59*dz,  58*dz])    
        
    #pos_x_arrs.append(pos_x_arr)
    #pos_z_arrs.append(pos_z_arr)    

    #############################################
    #middle perturb (start top left and go clockwise)
    #############################################
    pos_x_arr = []
    pos_z_arr = []
    
    #top
    pos_x_arr.extend([317*dx, 319*dx])
    pos_z_arr.extend([101*dz, 101*dz])    
    
    #right
    pos_x_arr.extend([320*dx, 322*dx, 324*dx, 326*dx, 327*dx, 329*dx, 331*dx, 333*dx])
    pos_z_arr.extend([102*dz, 103*dz, 104*dz, 105*dz, 106*dz, 107*dz, 108*dz, 109*dz])    
    
    #bot
    pos_x_arr.extend([307*dx])
    pos_z_arr.extend([109*dz])    
    
    #left
    pos_x_arr.extend([306*dx, 308*dx, 309*dx, 310*dx, 312*dx, 313*dx, 315*dx, 317*dx])
    pos_z_arr.extend([108*dz, 107*dz, 106*dz, 105*dz, 104*dz, 103*dz, 102*dz, 101*dz])    

    pos_x_arrs.append(pos_x_arr)
    pos_z_arrs.append(pos_z_arr)    
    
    return pos_x_arrs, pos_z_arrs

def plot_perturb_lines(fig, lw, c):
    pos_x_arrs, pos_z_arrs = get_trunc_pos_arrs()

    perturb_lines_1 = Line2D(pos_x_arrs[0], pos_z_arrs[0], linewidth = lw, color = c)
    #perturb_lines_2 = Line2D(pos_x_arrs[1], pos_z_arrs[1], linewidth = lw, color = c)
    #perturb_lines_3 = Line2D(pos_x_arrs[2], pos_z_arrs[2], linewidth = lw, color = c)
    fig.axes[0].add_line(perturb_lines_1)
    #fig.axes[0].add_line(perturb_lines_2)
    #fig.axes[0].add_line(perturb_lines_3)   