import numpy as np
from lmfit import Model
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
from matplotlib import cm
import math
from scipy import optimize,interpolate

def gaussian(x, amp, mean, sigma):
    return (amp / ( np.sqrt(2*np.pi) * sigma)) * np.exp(-(x-mean)**2 / (2*sigma**2))

def gaussian_asym():
    return 

def gauss(Xs, rho, xmin, xmax, size_of_bin):
    xs = Xs[ (Xs<=xmax) & (Xs>=xmin)]
    rho = rho[(Xs<=xmax) & (Xs>=xmin)]

    xs_max  = []
    rho_max = []
    deltaR  = size_of_bin
    step = 0

    maximo = np.max(xs)
    minimo = np.min(xs)

    #print('min, max = {}, {}'.format(minimo,maximo))
    while(minimo + step + deltaR <= maximo):
        #print(minimo+step+deltaR)
        #a = np.max( rho[(xs >= minimo) & (xs >= minimo + step ) & (xs <= minimo + step + deltaR) ] )
        #b = np.max( xs[(xs >= minimo ) & (xs >= minimo + step ) & (xs <= minimo + step + deltaR) ] )
        rho_tmp = rho[(xs >= minimo) & (xs >= minimo + step ) & (xs <= minimo + step + deltaR) ]
        x_tmp   = xs[(xs >= minimo ) & (xs >= minimo + step ) & (xs <= minimo + step + deltaR) ]
        
        #print(len(rho_tmp), len(x_tmp))
        if (len(rho_tmp) == 0) or (len(x_tmp) == 0):
            max_rho = np.nan
            max_x   = np.nan
        else:
            max_rho = np.max(rho_tmp)
            max_x   = np.max(x_tmp)
        
        rho_max.append(max_rho)
        xs_max.append(max_x)
        
        step = step + deltaR

    #print(xs_max)
    xs_max  = [x for x in xs_max if str(x) != 'nan' ]
    rho_max = [x for x in rho_max if str(x) != 'nan']

    #print(xs_max)
    rho_plus0 = []
    xs_plus0  = []
    for i in range(len(xs_max)):
        if rho_max[i] > 0:
            xs_plus0.append(xs_max[i])
            rho_plus0.append(rho_max[i])
            
    xs_max  = xs_plus0
    rho_max = rho_plus0 
    gmodel = Model(gaussian)
    mean_guess = np.mean(xs_plus0)
    sigma_guess = (np.absolute(xmax-xmin)*0.5)
    
    result = gmodel.fit(rho_plus0, x=xs_plus0, amp=1, mean=mean_guess, sigma=0.3)
    report = result.fit_report()
    
    best_values = result.best_values
    
    amp   = best_values['amp']
    deltaamp = result.params['amp'].stderr
    mean  = best_values['mean']
    deltamean = result.params['mean'].stderr
    sigma = best_values['sigma']
    deltasigma = result.params['sigma'].stderr
    
    xs_map = np.linspace(np.min(xs_plus0)-2,np.max(xs_plus0)+2,100)
    rho_map = gaussian(xs_map,amp,mean,sigma)

    return{'amp':amp,'mean':mean,'sigma':sigma,'xs_map':xs_map,'rho_map':rho_map,
           'xs_max':xs_max,'rho_max':rho_max,'deltamean':deltamean,'deltasigma':deltasigma,'deltaamp':deltaamp}

def plot_gaussian(gauss_fit,ax):
    ax.scatter(gauss_fit['xs_max'],gauss_fit['rho_max'], s=10, color = 'magenta')
    ax.plot(gauss_fit['xs_map'],gauss_fit['rho_map'], 'r-',label=(r" $\mu$ = %.2f"
                                        "\n"
                                        r" $\sigma$ = %.2f ")%(gauss_fit['mean'],gauss_fit['sigma']))
    ax.set_xlabel(r'x [kpc]')
    ax.set_ylabel(r'Property',fontsize=10)
    
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.legend()
    
def density_scatter_plot( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax, fraction=0.046, pad=0.04,cmap='jet')
    cbar.ax.set_ylabel('Density (KDE)')
    
    return {'fig':fig, 'ax':ax, 'cbar':cbar,'x':x,'y':y,'z':z}

def density_scatter( x , y, sort = True, bins = 20)   :

    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    return {'x':x,'y':y,'z':z}

def rotate(x, y, radians):
    x_rot = x * np.cos(radians) + y * np.sin(radians)
    y_rot = -x * np.sin(radians) + y * np.cos(radians)
    return {'x_rot':x_rot,'y_rot':y_rot}

def get_gaussian_fit(dict_lims,df_arm,n,bin_size):

    # Limits  
    limits = dict_lims['limits']
    limits_x = limits[:,0]
    limits_y = limits[:,1]
    
    # Interpolation of the limits
    f = interpolate.interp1d(limits_x, limits_y, kind='linear')

    x_band = df_arm.iloc[n].rotated_arm_profile['x_band']
    y_band = df_arm.iloc[n].rotated_arm_profile['y_band']

    x_band_lim = x_band[(x_band >= np.min(limits_x)) & (x_band <= np.max(limits_x))]
    y_band_lim = y_band[(x_band >= np.min(limits_x)) & (x_band <= np.max(limits_x))]


    x_band_final = []
    y_band_final = []

    for x,y in zip(x_band_lim,y_band_lim):
        if y < f(x):
            x_band_final.append(x)
            y_band_final.append(y)

    x_band_final = np.array(x_band_final)
    y_band_final = np.array(y_band_final)

    # Get fit
    arm_fit = gauss(x_band_final,y_band_final, np.min(x_band_final), np.max(x_band_final),bin_size)

    return {'arm_fit':arm_fit,'x_band_final':x_band_final,'y_band_final':y_band_final}