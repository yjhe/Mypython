# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 15:18:43 2015

@author: Yujie
"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import pylab
import os
import itertools

from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap, cm
from matplotlib.colors import Normalize

_markeriter = itertools.cycle((',', '+', '.', 'o', '*', 'v', '^', '<', '>', \
                               '1', '2', '3', '4', '8', 's', 'p', 'h', 'H', \
                               'x', 'D', 'd', '|', '_', 0, 1, 2, 3)) 

#class nlcmap(LinearSegmentedColormap):
#    """A nonlinear colormap"""
#
#    name = 'nlcmap'
#
#    def __init__(self, cmap, levels):
#        self.cmap = cmap
#        self.name = cmap.name
#        self.N = cmap.N
#        self.monochrome = self.cmap.monochrome
#        self.levels = np.asarray(levels, dtype='float64')
#        self.newlevels = self.levels + self.levels.min()
#        self._x = self.newlevels/ self.newlevels.max()
#        self._y = np.linspace(0.0, 1.0, len(self.levels))
#
#
#    def __call__(self, xi, alpha=1.0, **kw):
#        """docstring for fname"""
#        yi = np.interp((xi*self.levels.max()+self.levels.min())/self.newlevels.max(), self._x, self._y)
#        return self.cmap(yi, alpha)

class nlcmap(LinearSegmentedColormap):
    """A nonlinear colormap"""

    name = 'nlcmap'

    def __init__(self, cmap, levels):
        self.cmap = cmap
        self.name = cmap.name
        self.N = cmap.N
        self.monochrome = self.cmap.monochrome
        self.levels = np.asarray(levels, dtype='float64')
        self._x = self.levels/ self.levels.max()
        self._y = np.linspace(0.0, 1.0, len(self.levels))


    def __call__(self, xi, alpha=1.0, **kw):
        """docstring for fname"""
        yi = np.interp(xi, self._x, self._y)
        return self.cmap(yi, alpha)
        
# ----- coastline data
def _get_lon_lat():
    """ get coast line data (lon,lat) 
    """
    this_dir, this_filename = os.path.split(__file__)
    lon_path = os.path.join(this_dir,'lon.npy')
    lat_path = os.path.join(this_dir,'lat.npy')
    coast_lon = np.load(lon_path)
    coast_lat = np.load(lat_path)
    return coast_lon, coast_lat
_coast_lon, _coast_lat = _get_lon_lat()


def refline():
    """ add [0, 1] reference line to current plot
    """
    gca = plt.gca()
    gca.set_autoscale_on(False)
    lim_min = min(gca.get_xlim()[0], gca.get_ylim()[0])    
    lim_max = max(gca.get_xlim()[1], gca.get_ylim()[1])
    gca.set_xlim([lim_min, lim_max])
    gca.set_ylim([lim_min, lim_max])
    gca.plot(gca.get_xlim(),gca.get_ylim())
    
def xticks2lon(new_xticks=None):
    if new_xticks is not None:
        plt.gca().set_xticks(new_xticks)
    current_xticks = plt.gca().get_xticks()
    current_xticklabels = plt.gca().get_xticklabels()
    new_xticklabels = current_xticklabels
    for i, x in enumerate(current_xticks):
        x = np.mod(x,360)
        if x>0 and x<180:
            new_xticklabels[i] = str(int(x)) + u"\u00b0" + 'E'
        elif x>180 and x<360:
            new_xticklabels[i] = str(int(360-x)) + u"\u00b0" + 'W'
        elif x==0 or x==180:
            current_xticklabels[i] = str(int(abs(x))) 
    plt.gca().set_xticklabels(new_xticklabels)
    plt.draw()

def yticks2lat(new_yticks=None):
    if new_yticks is not None:
        plt.gca().set_yticks(new_yticks)
    current_yticks = plt.gca().get_yticks()
    current_yticklabels = plt.gca().get_yticklabels()
    new_yticklabels = current_yticklabels
    for i, y in enumerate(current_yticks):
        if y>0:
            new_yticklabels[i] = str(int(y)) + u"\u00b0" + 'N'
        elif y<0 :
            new_yticklabels[i] = str(int(-y))+ u"\u00b0" + 'S'
        else:
            new_yticklabels[i] = str(int(y))
    plt.gca().set_yticklabels(new_yticklabels)
    plt.draw() 

def vcolorbar(extend, size=None, pad=None,im=None,units=None, clim=None,
              cbartitle=None, ticks=None, cbarticklab=None, rotation=None,
              **kwargs):
    if size is None:
        size = '2.5%'
    if pad is None:
        pad = 0.1
    current_ax = plt.gca()        
    divider = make_axes_locatable(current_ax)
    # cax = divider.new_horizontal(size=size, pad=pad)
    # fig = plt.gcf()
    # fig.add_axes(cax)
    cax = divider.append_axes('right',size=size, pad=pad)
    if im is None:
        cbar = plt.colorbar(cax=cax,extend=extend,ticks=ticks,
                            **kwargs)
    else:
        cbar = plt.colorbar(im,cax=cax,extend=extend,**kwargs)
    if units is not None:
        # cbar.ax.set_title(units,loc='left',fontdict={'fontsize':'medium'})
        cbar.ax.text(1.01,1.0,units,transform=cbar.ax.transAxes)
    if cbarticklab is not None:
        cbar.set_ticklabels(cbarticklab)
        cbar.ax.set_yticklabels(cbarticklab,rotation=rotation)    
    if cbartitle is not None:
        cbar.set_label(cbartitle)
    plt.sca(current_ax)
    plt.clim(clim)
    return cbar
    
def hcolorbar(extend, size=None, pad=None,im=None,clim=None,units=None,
              cbartitle=None, ticks=None, cbarticklab=None, rotation=None,
              **kwargs):    
    """ horizontal color bar
    """
    if size is None:
        size = '5%'
    if pad is None:
        pad = 0.4
    current_ax = plt.gca()
    divider = make_axes_locatable(current_ax)
    cax = divider.append_axes('bottom',size=size, pad=pad)
    if im is None:
        cbar = plt.colorbar(extend=extend,cax=cax,orientation='horizontal',
                            ticks=ticks,**kwargs)
    else:
        cbar = plt.colorbar(im,cax=cax,extend=extend,orientation='horizontal',
                            ticks=ticks,**kwargs)
    if units is not None:
        cbar.ax.set_xlabel(units)
    if cbartitle is not None:
        cbar.set_label(cbartitle)
    if cbarticklab is not None:
        cbar.set_ticklabels(cbarticklab)
        cbar.ax.set_xticklabels(cbarticklab,rotation=rotation)
    plt.clim(clim)
    plt.sca(current_ax)
    return cbar

def contour(*args, **kwargs):
    cs = plt.contour(*args, colors='gray',lw=.2, **kwargs)
    plt.clabel(cs,cs.levels[::2],fmt='%g')
    return cs
    
def add_regress(x,y,Ndigits=None,**kwargs):
    """ add regression line to time series plot
    """
    b,a,r,p,e = st.linregress(x,y)
    plt.plot(x,x*b+a,**kwargs)
    if Ndigits is None:
        theText = 'b = ' + '%.1e'%b + '\n' \
                + 'a = ' + '%.1e'%a + '\n' \
                + 'r = ' + '%.1e'%r + '\n' \
                + 'p = ' + '%.1e'%p + '\n' \
                + 'e = ' + '%.1e'%e + '\n'
    else:
        N = Ndigits
        theText = 'b = ' + str(round(b,N)) + '\n' \
                + 'a = ' + str(round(a,N)) + '\n' \
                + 'r = ' + str(round(r,N)) + '\n' \
                + 'p = ' + str(round(p,N)) + '\n' \
                + 'e = ' + str(round(e,N)) + '\n' 
    pylab.text(.02,.98,theText,transform=plt.gca().transAxes,verticalalignment='top')
    pylab.draw()

def imshow(A,x=None,y=None, cmap=None, **kwargs):
    Ashape = A.shape
    if x is None:
        x = np.arange(Ashape[1])
    if y is None:
        y = np.arange(Ashape[0])
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    im = plt.imshow(A, extent=[x[0]-dx/2., x[-1]+dx/2., y[0]-dy/2., y[-1]+dy/2.],
                    cmap=cmap, **kwargs)
    return im  

def drawcoastlines(lon=None,lat=None,lonticks=None, latticks=None, scalar=1.):
    if lon is None:
        lon = np.array([0,360])
    if lat is None:
        lat = np.array([-90,90])
    if lonticks is None:
        lonticks = np.arange(-180,360,30)
    if latticks is None:
        latticks = np.arange(-90,91,15)
    lines = plt.plot(_coast_lon,scalar*_coast_lat,'k-',linewidth=0.5)
    plt.xticks(lonticks)
    plt.yticks(latticks)
    plt.xlim(lon[0],lon[-1])
    plt.ylim(lat[0],lat[-1])  
    xticks2lon()
    yticks2lat()
    return lines

# ---- cmap related -----
def make_cmap(cmap_name='bwr', N=20):
    """ self-defined color map
    """
    if cmap_name == 'br':
        cdict = {
            'red'  : ((0., 0., 0.), (0.5, 1., 1.), (1., 1., 1.)),
            'green': ((0., 0., 0.), (0.5, 1., 1.), (1., 0., 0.)),
            'blue' : ((0., 1., 1.), (0.5, 1., 1.), (1., 0., 0.))
        }
    elif cmap_name == 'bwr':
        cdict = {
            'red'  : ((0., 0., 0.), (0.45, 1., 1.), (0.55, 1., 1.), (1., 1., 1.)),
            'green': ((0., 0., 0.), (0.45, 1., 1.), (0.55, 1., 1.), (1., 0., 0.)),
            'blue' : ((0., 1., 1.), (0.45, 1., 1.), (0.55, 1., 1.), (1., 0., 0.))
        }
    elif cmap_name == 'drywet' or cmap_name=='pra':
        cdict = {
            'red'  : ((0., 0.4, 0.4), (0.5, 1., 1.), (1., 0., 0.)),
            'green': ((0., 0.3, 0.3), (0.5, 1., 1.), (1., 0.6, 0.6)),
            'blue' : ((0., 0.2, 0.2), (0.5, 1., 1.), (1., 0., 0.))
        }
    elif cmap_name == 'dry-wet' or cmap_name=='pr_a':
        cdict = {
            'red'  : ((0., 0.4, 0.4), (0.45, 1., 1.), (0.55, 1., 1.), (1., 0., 0.)),
            'green': ((0., 0.3, 0.3), (0.45, 1., 1.), (0.55, 1., 1.), (1., 0.6, 0.6)),
            'blue' : ((0., 0.2, 0.2), (0.45, 1., 1.), (0.55, 1., 1.), (1., 0., 0.))
        }
    elif cmap_name == 'prcp':
        cdict = {
            'red'  : ((0., 1., 1.), (1., 0., 0.)),
            'green': ((0., 1., 1.), (1., 0.6, 0.6)),
            'blue' : ((0., 1., 1.), (1., 0., 0.))
        }
    return LinearSegmentedColormap('mycolormap',cdict,N)

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
   

# ---- advanced functions -----------
def geoshow(zz, lon, lat, theTitle=None, cmap=None, clim=None, \
            levels=None, plottype='imshow',maskColor='gray', \
            cbar='v', cbartitle=None, units=None, ticks=None, cbarticklab=None, 
            rotation=None, extend='neither',**kwargs):
    """ geographical projection, imshow or contourf plot with default of 5 levels
    params:
        zz : passed in map matrix, 90S-90N
        levels : number of levels
    """
    # default cmap
    if cmap is None:
        if levels is not None:
            cmap = plt.cm.get_cmap('jet',levels)
        else:
            cmap = plt.cm.jet
    elif isinstance(cmap, str) and cmap.startswith('GMT',0,3):
        if levels is not None:
            cmap = plt.cm.get_cmap('cm.'+cmap,levels)
        else:
            cmap = eval('cm.' + cmap)
    # default clim
    if clim is None:
        if isinstance(zz,np.ma.core.MaskedArray):
            zz1d = zz.compressed()
        else:
            zz1d = zz.ravel()
            notNaNs = np.logical_not(np.isnan(zz1d))
            zz1d = zz1d[notNaNs]
        clim = np.percentile(zz1d,1), np.percentile(zz1d,99)

    # base map coastal lines
    plt.gca().set_aspect('equal','box-forced') 
    drawcoastlines(lon,lat,scalar=1.)
    
    # image show or contourf figure
    if plottype == 'contourf':
        # contourf
        if levels is None:
            levels = np.arange(clim[0],clim[-1],5) # default 5 levels
        im = plt.contourf(lon,lat,zz,levels=levels,cmap=cmap,
            extend='both',**kwargs)            
        # default mask color
        if maskColor is not None:
            plt.gca().patch.set_color(maskColor)
    elif plottype == 'imshow':
        if levels is not None:
            cmap = plt.get_cmap(cmap,levels)
        im = imshow(np.flipud(zz),lon,lat,cmap=cmap,**kwargs)
        if maskColor is not None:
            cmap.set_bad(maskColor)
            
    plt.clim(clim)
    
    # colorbar
    if cbar=='v':
        vcolorbar(extend,im=im,cbartitle=cbartitle,ticks=ticks,
                  cbarticklab=cbarticklab,units=units,rotation=rotation)
    elif cbar=='h':
        hcolorbar(extend,im=im,cbartitle=cbartitle,units=units,ticks=ticks,
                  cbarticklab=cbarticklab,rotation=rotation)
    
    if theTitle is not None:
        plt.title(theTitle)
    plt.show()  
    # return a mappable quantity to be used for colorbar
    return im
    
def millshow(zz,lon,lat,lonmin=-180.,lonmax=180.,latmin=-70,latmax=90, \
            theTitle=None,cmap=None,clim=None,map_type='coastline', \
            cbar='v',cbartitle=None,units=None):
    """ mill projection, imshow plot 
    """
    if cmap is None:
        cmap = plt.cm.jet
    elif cmap.startswith('GMT',0,3):
        cmap = eval('cm.' + cmap) # use cmap from Basemap module cm
    if clim is None:
        if isinstance(zz,np.ma.core.MaskedArray):
            zz1d = zz.compressed()
        else:
            zz1d = zz.ravel()
        clim = np.percentile(zz1d,1), np.percentile(zz1d,99)
    
    # base map
    lon_0 = (lon[0] + lon[-1])/2.
    m = Basemap(projection='mill', lon_0=lon_0, llcrnrlon=lonmin, \
                llcrnrlat=latmin, urcrnrlon=lonmax, urcrnrlat=latmax)
    if map_type=='coastline':
        m.drawcoastlines(linewidth=0.25)
    elif map_type=='filled':
        m.fillcontinents(color=[.33, .33, .33])
    
    # show data
    nx = len(lon); ny = len(lat)
    data = m.transform_scalar(zz,lon,lat,nx,ny)
    im = m.imshow(data,cmap)
    
    # labels        
    plt.clim(clim)
    m.drawmapboundary(fill_color='grey')
    m.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0])
    m.drawmeridians(np.arange(0.,420.,60.),labels=[0,0,0,1])
    if theTitle is not None:
        plt.title(theTitle)
    if cbar=='v':
        vcolorbar(im=im,cbartitle=cbartitle,units=units)
    elif cbar=='h':
        hcolorbar(im=im,cbartitle=cbartitle,units=units)
    return im    
    
def mollshow(zz,lon,lat,lonmin=-180.,lonmax=180.,latmin=-70,latmax=90, \
             theTitle=None,cmap=None,clim=None,levels=None,map_type='coastline',\
             cbartitle=None,units=None,cbar='v'):
    """ moll projection, contourf plot with default of 5 levels
    """
    if cmap is None:
        cmap = plt.cm.jet
    elif cmap.startswith('GMT',0,3):
        cmap = eval('cm.' + cmap) # use cmap from Basemap module cm
    if clim is None:
        if isinstance(zz,np.ma.core.MaskedArray):
            zz1d = zz.compressed()
        else:
            zz1d = zz.ravel()
        clim = np.percentile(zz1d,1), np.percentile(zz1d,99)
    if levels is None:
        levels = np.linspace(clim[0],clim[-1],5)
    
    # base map
    lon_0 = (lon[0] + lon[-1])/2
    m = Basemap(projection='moll', lon_0=lon_0, llcrnrlon=lonmin, \
                llcrnrlat=latmin, urcrnrlon=lonmax, urcrnrlat=latmax)
    if map_type=='coastline':
        m.drawcoastlines()
    elif map_type=='filled':
        m.fillcontinents(color=[.33, .33, .33])
    
    # show data
    Lon, Lat = np.meshgrid(lon,lat)
    X,Y = m(Lon,Lat)
    cs = m.contourf(X,Y,zz,cmap=cmap,levels=levels,extend='both')
    # cs = m.pcolormesh(X,Y,zz,cmap=cmap)
    
    # labels        
    plt.clim(clim)
    m.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0])
    m.drawmeridians(np.arange(0.,420.,60.),labels=[0,0,0,1])
    if theTitle is not None:
        plt.title(theTitle)
    if cbar=='v':
        vcolorbar(im=cs,cbartitle=cbartitle,units=units)
    elif cbar=='h':
        hcolorbar(im=cs,cbartitle=cbartitle,units=units)        
    return cs    

def robinshow(zz,lon,lat,lonmin=-180,lonmax=180,latmin=-80,latmax=90,
              theTitle=None, cmap=None, clim=None, \
              plottype='contourf',nlev=None, maskColor='gray', \
              cbar='v', cbartitle=None, units=None, clev=None, 
              extend='neither',ticks=None,cbarticklab=None,**kwargs):
    """ robin projection, imshow or contourf plot with default of 5 levels
    params:
        nlev : number of levels for contourf plot
        clev   : exact levels (array) you want 
    """
    # default cmap
    if cmap is None:
        if nlev is not None:
            cmap = plt.cm.get_cmap('jet',nlev)
        else:
            cmap = plt.cm.jet
    elif isinstance(cmap, str) and cmap.startswith('GMT',0,3):
        if nlev is not None:
            cmap = plt.cm.get_cmap('cm.'+cmap,nlev)
        else:
            cmap = eval('cm.' + cmap)
    else:
        if nlev is not None:
            cmap = plt.cm.get_cmap(cmap.name,nlev)

    # default clim
    if clim is None:
        if isinstance(zz,np.ma.core.MaskedArray):
            zz1d = zz.compressed()
        else:
            zz1d = zz.ravel()
            notNaNs = np.logical_not(np.isnan(zz1d))
            zz1d = zz1d[notNaNs]
        clim = np.percentile(zz1d,1), np.percentile(zz1d,99)
    
    # base map coastal lines
    # base map
    lon_0 = (lon[0] + lon[-1])/2
    m = Basemap(projection='robin', lon_0=lon_0, llcrnrlon=lonmin, \
                llcrnrlat=latmin, urcrnrlon=lonmax, urcrnrlat=latmax)
    m.drawcoastlines(linewidth=0.4)
    # show data
    Lon, Lat = np.meshgrid(lon,lat)
    X,Y = m(Lon,Lat)
    # cs = m.pcolormesh(X,Y,zz,cmap=cmap)
    zm = np.ma.masked_where(np.isnan(zz),zz)
    # image show or contourf figure
    if plottype=='contourf':
        # contourf
        if nlev is None and clev is None:
            im = m.contourf(X,Y,zm,levels=np.arange(clim[0],clim[-1],(clim[-1]-clim[0])/5.),
                              cmap=cmap, extend=extend,**kwargs)
            # default mask color
            if maskColor is not None:
                plt.gca().patch.set_color(maskColor)
        elif nlev is not None and clev is None:
            im = m.contourf(X,Y,zm,levels=np.arange(clim[0],clim[-1],nlev),
                              cmap=cmap,extend=extend,**kwargs)
            # default mask color
            if maskColor is not None:
                plt.gca().patch.set_color(maskColor)
        elif clev is not None:
            im = m.contourf(X,Y,zm,levels=clev,cmap=cmap,
                extend=extend,**kwargs)
            # default mask color
            if maskColor is not None:
                plt.gca().patch.set_color(maskColor)
        
    # image show
    elif plottype == 'imshow':
        if maskColor is not None:
            cmap.set_bad(maskColor)
        im = imshow(np.flipud(zm),lon,lat,cmap=cmap,**kwargs)
    # colorbar
    plt.clim(clim)
    m.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0],fontsize=8)
    m.drawmeridians(np.arange(0.,420.,60.),labels=[0,0,0,1],fontsize=8)
    if cbar=='v':
        vcolorbar(extend=extend,cbartitle=cbartitle,ticks=ticks,
                  cbarticklab=cbarticklab,units=units)
    elif cbar=='h':
        hcolorbar(extend=extend,im=im,cbartitle=cbartitle,ticks=ticks,
                  cbarticklab=cbarticklab,units=units)       
    
    if theTitle is not None:
        plt.title(theTitle)
    plt.show()  
    # return a mappable quantity to be used for colorbar
    return im
      
def orthoshow(zz,lon,lat,lon_0=None,lat_0=None,\
              theTitle=None,cmap=None,clim=None,levels=None, map_type='coastline', \
              cbartitle=None,units=None,cbar='v'):
    """ ortho projection, contourf plot with default of 5 levels
    """
    if cmap is None:
        # cmap = make_cmap(cmap_name='br',N=20)
        # cmap = 'jet'
        cmap = plt.cm.jet
    elif cmap.startswith('GMT',0,3):
        cmap = eval('cm.' + cmap) # use cmap from Basemap module cm
    if clim is None:
        if isinstance(zz,np.ma.core.MaskedArray):
            zz1d = zz.compressed()
        else:
            zz1d = zz.ravel()
        clim = np.percentile(zz1d,1), np.percentile(zz1d,99)
    if levels is None:
        levels = np.linspace(clim[0],clim[-1],5)
    if lon_0 is None:        
        lon_0 = (lon[0] + lon[-1])/2
    if lat_0 is None:
        lat_0 = (lat[0] + lat[-1])/2
    m = Basemap(projection='ortho', lat_0=lat_0, lon_0=lon_0)
    if map_type=='coastline':
        m.drawcoastlines()
    elif map_type=='filled':
        m.fillcontinents(color=[.33, .33, .33])
    m.drawparallels(np.arange(-90.,90.,30.))
    m.drawmeridians(np.arange(-180.,360.,60.))
    Lon, Lat = np.meshgrid(lon,lat)
    X,Y = m(Lon,Lat)
    cs = m.contourf(X,Y,zz,cmap=cmap,levels=levels,extend='both')
    plt.clim(clim)
    if theTitle is not None:
        plt.title(theTitle)
    if cbar=='v':
        vcolorbar(im=cs,cbartitle=cbartitle,units=units)
    elif cbar=='h':
        hcolorbar(im=cs,cbartitle=cbartitle,units=units)        
    return cs    