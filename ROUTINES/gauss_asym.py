import numpy as np
from lmfit import Model
from lmfit.models import SkewedGaussianModel
from scipy import optimize,interpolate
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
from matplotlib import cm
import math
from shapely.geometry import LineString
from shapely.geometry import MultiPoint, Point

tsize = 12
tdir = 'in'
major = 5.0
minor = 3.0
style = 'default'
plt.style.use(style)
plt.rcParams['text.usetex'] = True
plt.rcParams['legend.fontsize'] = tsize
plt.rcParams['xtick.direction'] = tdir
plt.rcParams['ytick.direction'] = tdir
plt.rcParams['xtick.major.size'] = major
plt.rcParams['xtick.minor.size'] = minor
plt.rcParams['ytick.major.size'] = major
plt.rcParams['ytick.minor.size'] = minor

def convert_from_listAsAString_to_listOfFloats(string):
    splitted = string.split(',')
    splitted[0] = splitted[0][1:]
    splitted[-1] = splitted[-1][:-1]
    stripped = [s.strip() for s in splitted]
    listOfFloats = [float(s) for s in stripped]
    return listOfFloats 

def asym_gauss(Xs, rho, xmin, xmax, size_of_bin):
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
        rho_tmp = rho[(xs >= minimo) & (xs >= minimo + step ) & (xs <= minimo + step + deltaR) ]
        x_tmp   = xs[(xs >= minimo ) & (xs >= minimo + step ) & (xs <= minimo + step + deltaR) ]
        
        if (len(rho_tmp) == 0) or (len(x_tmp) == 0):
            max_rho = np.nan
            max_x   = np.nan
        else:
            max_rho = np.max(rho_tmp)
            max_x   = np.max(x_tmp)
        
        rho_max.append(max_rho)
        xs_max.append(max_x)
        
        step = step + deltaR

    xs_max  = [x for x in xs_max if str(x) != 'nan' ]
    rho_max = [x for x in rho_max if str(x) != 'nan']

    rho_plus0 = []
    xs_plus0  = []
    for i in range(len(xs_max)):
        if rho_max[i] > 0:
            xs_plus0.append(xs_max[i])
            rho_plus0.append(rho_max[i])
            
    xs_max  = pd.Series(xs_plus0)
    rho_max = pd.Series(rho_plus0) 
    
    model = SkewedGaussianModel()
    params = model.guess(rho_max, x=xs_max)
    model_fit = model.fit(rho_max, params, x=xs_max)
    
    return {'model_fit':model_fit,'x_max':xs_max,'y_max':rho_max}

def get_asym_gaussian_fit(dict_lims,df_arm,n,bin_size):

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
    
    arm_fit = asym_gauss(x_band_final,y_band_final, np.min(x_band_final), np.max(x_band_final),bin_size)

    return {'arm_fit':arm_fit['model_fit'],'x_max':arm_fit['x_max'],
            'y_max':arm_fit['y_max']}

def add_row_with_param_info(model, df, arm_tracing_info, delta, fractured = False):
    modelResult = model['arm_fit']
    params = {}
    for param_name, param_info in modelResult.params.items():
        params[param_name] = {'value':param_info.value, 'stderr':param_info.stderr}  
    
    dict_values = {'amplitude':params['amplitude']['value'],'stderr_amplitude':params['amplitude']['stderr'],
                   'center'   :params['center']['value'],    'stderr_center'   :params['center']['stderr'],
                   'sigma'    :params['sigma']['value'],'stderr_sigma'    :params['sigma']['stderr'],
                   'gamma'    :params['gamma']['value'],    'stderr_gamma'    :params['gamma']['stderr'],
                   'x':list(model['x_max']),
                   'y':list(model['y_max']),
                   'delta':delta,
                   'x_tracing':arm_tracing_info['x'],
                   'y_tracing':arm_tracing_info['y'],
                   'id_tracing':arm_tracing_info['id'],
                   'fractured':fractured}
    df = pd.concat([df, pd.DataFrame.from_records([dict_values])], ignore_index=True)
    return df

def eval_fitted_model(model):
    modelResult = model['arm_fit']
    amplitude = modelResult.params['amplitude'].value
    center = modelResult.params['center'].value
    sigma = modelResult.params['sigma'].value
    gamma = modelResult.params['gamma'].value

    model_analytical = SkewedGaussianModel()
    params_analytical = model_analytical.make_params(amplitude = amplitude,
                                                    center = center,
                                                    sigma = sigma,
                                                    gamma = gamma)

    x_eval = np.arange(np.min(model['x_max'])-2, np.max(model['x_max'])+2, 0.001)
    model_analytical_eval = model_analytical.eval(params = params_analytical,
                                                x = x_eval)
    return {'x_eval':x_eval,'y_eval':model_analytical_eval}

def find_maximum(x,y):
    x = pd.Series(x)
    y = pd.Series(y)

    return x[y == np.max(y)].values[0], y[y == np.max(y)].values[0]

def get_param_of_model(model, param):
    modelResult = model['arm_fit']
    if param == 'amplitude':
        return modelResult.params['amplitude'].value
    elif param == 'center':
        return modelResult.params['center'].value
    elif param == 'sigma':
        return modelResult.params['sigma'].value
    elif param == 'gamma':
        return modelResult.params['gamma'].value

def eval_fitted_model_with_given_params(params, dx = 0.001):
    amplitude = params['amplitude']
    center = params['center']
    sigma = params['sigma']
    gamma = params['gamma']

    x = params['x']
    y = params['y']

    model_analytical = SkewedGaussianModel()
    params_analytical = model_analytical.make_params(amplitude = amplitude,
                                                    center = center,
                                                    sigma = sigma,
                                                    gamma = gamma)

    x_eval = np.arange(np.min(x)-2, np.max(x)+2, dx)
    model_analytical_eval = model_analytical.eval(params = params_analytical,
                                                x = x_eval)
    return {'x_eval':x_eval,'y_eval':model_analytical_eval}

from shapely.geometry import MultiPoint

def find_widths_of_skewed_gaussian_new(params, dx=0.001):
    # Evaluar la función gaussiana ajustada en un conjunto de puntos
    dict_eval = eval_fitted_model_with_given_params(params, dx=dx)
    x_eval = dict_eval['x_eval']
    y_eval = dict_eval['y_eval']
    
    # Encontrar las intersecciones con el eje y = y_max / 2
    y_max = np.max(y_eval)
    y_half = 0.5 * y_max
    
    # Definir la línea en y = y_max / 2
    line_half_max = LineString([(x_eval[0], y_half), (x_eval[-1], y_half)])
    
    # Encontrar intersección
    curve = LineString(np.column_stack([x_eval, y_eval]))
    intersection = curve.intersection(line_half_max)
    
    # Convertir la intersección a una lista, ya sea de un solo punto o múltiples puntos
    if isinstance(intersection, (MultiPoint, Point)):
        points_intercept = list(intersection) if isinstance(intersection, MultiPoint) else [intersection]
    else:
        points_intercept = []

    # Si hay puntos de intersección, obtener las coordenadas
    if points_intercept:
        points_intercept = [[point.x, point.y] for point in points_intercept]

    # Procesar los puntos para obtener w1 y w2
    x_max = x_eval[np.argmax(y_eval)]
    if len(points_intercept) >= 2:
        w1 = np.abs(x_max - points_intercept[0][0])
        w2 = np.abs(x_max - points_intercept[-1][0])
    else:
        w1, w2 = np.nan, np.nan  # En caso de no haber suficientes intersecciones
    
    width_info = {
        'x_max': x_max,
        'y_max': y_max,
        'w1': w1,
        'w2': w2
    }
    
    return width_info


def find_widths_of_skewed_gaussian(params, dx = 0.001):

    fitted_model = eval_fitted_model_with_given_params(params, dx = dx)
    x_eval = fitted_model['x_eval']
    y_eval = fitted_model['y_eval']

    x_max, y_max = find_maximum(x_eval,y_eval)

    xs_line = np.arange(np.min(x_eval), np.max(x_eval), 0.01)
    ys_line = np.full(len(xs_line), 0.5 * y_max)

    first_line = LineString(np.column_stack((x_eval, y_eval)))
    second_line = LineString(np.column_stack((xs_line, ys_line)))
    intersection = first_line.intersection(second_line)

    points_intercept = []
    for point in intersection:
        points_intercept.append([point.x, point.y])

    #print('p1:', points_intercept[0][0],'p2:', points_intercept[1][0])   
        
    w1 = x_max - points_intercept[1][0]
    w2 = points_intercept[0][0] - x_max
    return {'x_max':x_max, 'y_max':y_max, 'w1':w1, 'w2':w2}

def plot_fit_and_widths(params, ax, dx = 0.001):

    width_info = find_widths_of_skewed_gaussian_new(params, dx = dx)

    dict_eval = eval_fitted_model_with_given_params(params, dx = dx)
    x_eval = dict_eval['x_eval']
    y_eval = dict_eval['y_eval']

    ax.plot(x_eval, y_eval, lw = 2.5, label = 'Fitted Function')
    ax.axvline(x = width_info['x_max'], color = 'red', lw = 1.5, label = 'x_max = {:.3f}'.format(width_info['x_max']))
    ax.axhline(y = width_info['y_max'], ls = '--', color = 'k', label = 'y_max = {:.2e}'.format(width_info['y_max']))
    ax.axhline(y = 0.5 * width_info['y_max'], color = 'green', lw = 1.5, label = 'y_max / 2 = {:.2e}'.format(0.5 * width_info['y_max']))

    ax.axvline(x = np.abs(width_info['x_max'] - width_info['w1']), color = 'magenta', lw = 1.5, ls = '--', label = 'w1 = {:.3f}'.format(width_info['w1']))
    ax.axvline(x = np.abs(width_info['x_max'] + width_info['w2']), color = 'purple', lw = 1.5, ls = '--', label = 'w2 = {:.3f}'.format(width_info['w2']))

    xt = ax.get_xticks()
    tick1 = '{:.2f}'.format(np.abs(width_info['x_max'] - width_info['w1']))
    tick2 = '{:.2f}'.format(np.abs(width_info['x_max'] + width_info['w2']))
    xt=np.append(xt,[float(tick1), float(tick2)])
    
    ax.set_xticks(xt)
    ax.set_xticklabels(xt)

    ax.grid(alpha = 0.5, lw = 1, ls = '--')
    ax.legend(loc ='upper left')