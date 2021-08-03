#!/usr/bin/env python

from astropy.coordinates import SkyCoord, Galactic, ICRS, FK5
from astropy.table import Column, Table, join
from ligo.skymap.io import read_sky_map
from matplotlib import pyplot as plt
from astropy_healpix import HEALPix
import astropy.coordinates as coord
from astropy import units as u
from astropy.io import fits, ascii
import ligo.skymap.plot
from ligo.skymap.postprocess.contour import contour
import ligo.skymap.moc
import numpy as np
import healpy as hp
import scipy
import math

# The plotting code below was written by Stef Countryman:

DPI = 200
WIDTH = 6  # [inches]
HEIGHT = 4  # [inches]
LEFT_SHIFT_COEFF = 1/20.  # quadratic curve dec labels away from meridian
LEFT_SHIFT_BASE = -20  # baseline shift from leftmost meridian [deg]
XMARGIN = 0.4  # additional margin in x (make sure text fits) [inches]
TOPMARGIN = -0.0  # additional margin at top [inches]
BOTTOMMARGIN = -0.0  # additional margin at bottom [inches]
DEC_X_OFFSET = -0.37  # [inches]
DEC_Y_OFFSET = -0.02  # [inches]
RA_X_OFFSET = 0  # [inches]
RA_Y_OFFSET = 0.06  # [inches]
N_X_OFFSET = 0.08  # [inches]
N_Y_OFFSET = 0.08  # [inches]
CENTRAL_LON = 180  # longitude to place at center of skymap [deg]
CENTRAL_LAT = 0  # latitude to place at center of skymap [deg]
PSI = 0  # additional rotation of skymap about center
DELTA_PARALLEL = 15  # space between parallels in the graticule [deg]
DELTA_MERIDIAN = 30  # space between meridians in the graticule [deg]
GRATICULE_COLOR = "#B0B0B0"
GRATICULE_LABEL_COLOR = (0.2, 0.2, 0.2)
MERIDIAN_FONT_SIZE = 11
FONT_SIZE = 14  # matplotlib font size
UNCERTAINTY_ALPHA = 0.4  # opacity in [0,1] for scatterplot uncertainty discs
DEFAULT_PLOT_EXTENSION = 'pdf'  # file type to save plots as

def plot_mollview(skymap,
                  nest,
                  title=None,
                  unit="Prob. Density",
                  central_longitude=CENTRAL_LON,
                  central_latitude=CENTRAL_LAT,
                  rotation_about_center=PSI,
                  dpi=DPI,
                  graticule=15,
                  cbar=None,
                  cmap=None,
                  **kwargs):
    ''' 
    Function to plot with the RA and DEC in Mollweid view.

    Parameters:
    -----------
    skymap: NdArray or Column
        Column containing pixel data.
    nest: Boolean
        True if data is nested. 
        Default: False (ring ordering)
    title: Str
        Title of the plot. 
        Default: None
    unit: Str
        Unit to be written on colorbar (if True). 
        Default: 'Prob. Density'
    central_longitude: Float [deg]
        Longitude desired for center of skymap.
        Default: 180 [deg]
    central_latitude: Float [deg]
        Latitude desired for center of skymap.
        Default: 0 [deg]
    rotation_about_center: Float [deg]
        Additional rotation of skymap about the center.
        Default: 0 [deg]
    dpi: Float
        Default: 200
    graticule: Int
        Spacing between graticule lines.
        Default: 15
    cbar: Boolean
        If True, show color bar.
    cmap: Str
        Color map. From Matplotlib cmap.
        Default: None
        
    Returns:
    -----------
    fig: Figure
        Final skymap.
    '''
    from matplotlib.transforms import ScaledTranslation
    kwargs['rot'] = kwargs.get('rot', (180, 0, 0))
    
    
    hp.mollview(skymap, 
                nest=nest, 
                cmap=cmap,  
                unit=unit, 
                rot=(central_longitude, central_latitude, rotation_about_center),
                cbar=cbar)
    hp.graticule(graticule)
    
    # size the figure
    fig = plt.gcf() # gcf = get current figure
    fig.set_size_inches(WIDTH, HEIGHT)
    fig.set_dpi(dpi)
    if title is not None:
        plt.title(title)
        
    # add extra margin so that the declination labels fit
    x_1, x_2, y_1, y_2 = plt.axis()
    plt.axis((x_1-XMARGIN, x_2+XMARGIN, y_1-TOPMARGIN, y_2+BOTTOMMARGIN))
    
    # Make a matplotlib translation to offset text labels by a bit so that
    # they don't cover up the scatter plot markers they are labeling. See:
    # https://matplotlib.org/users/transforms_tutorial.html
    transdata = plt.gca().transData # gca = get current axis
    dectrans = transdata + ScaledTranslation(DEC_X_OFFSET, DEC_Y_OFFSET, fig.dpi_scale_trans)
    ratrans = transdata + ScaledTranslation(RA_X_OFFSET, RA_Y_OFFSET, fig.dpi_scale_trans)
    
    # add a graticule
    hp.graticule(DELTA_PARALLEL,
                 DELTA_MERIDIAN,
                 color=GRATICULE_COLOR,)
    
    # label declinations
    for dec in np.arange(DELTA_PARALLEL-90, 90, DELTA_PARALLEL):
        hp.projtext(# curve dec labels away from leftmost meridian
                    LEFT_SHIFT_BASE+abs(dec*LEFT_SHIFT_COEFF)**2,
                    dec,
                    # label dec, right-pad to avoid hitting plot
                    str(dec) + '$\\degree$',
                    lonlat=True,
                    color=GRATICULE_LABEL_COLOR,
                    transform=dectrans,
                    horizontalalignment='right',
                    verticalalignment='center')
    # label right ascensions
    for right_ascension in np.arange(DELTA_MERIDIAN, 
                                     360, 
                                     DELTA_MERIDIAN):
        hp.projtext(right_ascension,
                    0,
                    str(right_ascension) + '$\\degree$',
                    lonlat=True,
                    color=GRATICULE_LABEL_COLOR,
                    transform=ratrans,
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    fontsize=MERIDIAN_FONT_SIZE)
    return fig

# the following functions was written by Amanda Beck and Yasmeen Asali:

def convert_RaDec_Index_manual(data, n_side):
    ''' 
    Manually converting RA and DEC to healpix nested pixel index (marginally faster than using astropy's methods).
    Adds pixel index, theta, and phi columns to data set.

    Parameters:
    -----------
    data: Table or DataFrame
        Must have Ra and Dec columns named as RAJ2000 and DEJ2000.
    n_side: float or int
        nside value corresponding to LIGO skymap to be crossmatched with the AGN data.
    '''
    n_side_array = np.full(len(data['RAJ2000']), int(n_side))
    theta = 0.5 * np.pi - np.deg2rad(data['DEJ2000'])
    phi = np.deg2rad(data['RAJ2000'])
    ipix = hp.ang2pix(n_side_array, theta, phi, nest=True)
    data.add_columns([np.array(ipix), theta, phi],
                     names=['INDEX', 'THETA', 'PHI'])

def agn_counts_per_pixel(agn_index, bbh_index):
    ''' 
    Using the AGN pixel indexes, identifies how many AGN per pixel in a skymap.

    Parameters:
    -----------
    agn_indes: NdArray or Column
        AGN pixel indexes in nested ordering.
    bbh_index: NdArray or Column
        Pixel indexes in nested ordering of the current skymap. In this case, this is the INDEX column of the LIGO data.
        
    Returns:
    -----------
    agn_counts: Table or DataFrame
        AGN counts per pixel.
    hpx_map: NdArray
        AGN Healpix skymap.
    '''
    idx, counts = np.unique(agn_index, return_counts=True)
    hpx_map = np.zeros(len(bbh_index), dtype=int)
    hpx_map[idx] = counts
    agn_counts = Table([idx, counts], names=['INDEX', 'AGN COUNT'])
    
    return agn_counts, hpx_map

def isolate_90_area(data, order=None):
    ''' 
    Isolates the 90% highest probability area.

    Parameters:
    -----------
    data: Table or DataFrame
        LIGO data containing pixel probability column.
        
    Returns:
    -----------
    ninety_area: Table or DataFrame
        Isolated 90% probability area pixels of the LIGO data.  
    '''
    # ordering table by probability:
    data = np.sort(data, order=order)
    # adding from top to bottom till 90% prob area:
    data = Table(data)[::-1]
    s = 0
    for i in range(len(data['PROB'])):
        ninety_area = data[:i]
        s += data['PROB'][i]
        if (s >= 0.9):
            break
            
    return ninety_area