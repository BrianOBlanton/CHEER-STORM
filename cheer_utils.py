# git@github-work:BrianOBlanton/CHEER-STORM/cheer_utils

"""
    CHEER_UTILS
    utilities for CHEER-STORM
    from repo for codes/processing of STORM datasets for CHEER
    Version 1.6, 13 May 2024
    GitHub Repo: git@github.com:BrianOBlanton/CHEER-STORM.git
    Brian Blanton, RENCI
"""

import os
import errno
import re
import yaml
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs

# these are not really needed if using Cartopy plotting
coastline=np.loadtxt('static/coarse_us_coast.dat')
worldcoastline=np.loadtxt('static/worldcoast.dat')
statelines=np.loadtxt('static/states.dat')

def HbFromRmwLat(rmw,lat): 
    """
    Compute HollandB from RMW [km] and Lat [deg]
    
    Computes HollandB from RMW and Lat using Eqn 8 in Vickery & Wadhera 2008.
    "Statistical Models of Holland Pressure Profile Parameter and Radius to 
    Maximum Winds of Hurricanes from Flight-Level Pressure and H*Wind Data".
    J. Applied Met and Clim, Oct 2008, Vol 47
    
    hb = 1.881 - 0.0057*rmw - 0.01295*lat
  
    Parameters:
    rmw (float): radius to max winds [km]
    lat (float): latitude [deg]
  
    Returns:
    int: Description of return value
  
    """
    assert np.all(lat>0), 'Latitudes must be all >0 for this model.'
    return 1.881 - 0.0057*rmw - 0.01295*lat

def ss_scale(spd,units='m/s'):
    """
    assumes spd (max wind speed) default is in m/s
    ------------------------------------------------------------------------------
       Category     Value | wind speed (mph) |  wind speed (m/s) | pres (mb)
    ------------------------------------------------------------------------------
       Trop Dep      -1      0 <= s < 39          0 <= s < 17.4       
       Trop Storm     0     39 <= s < 74       17.4 <= s < 33.1 
          One         1     74 <= s < 96       33.1 <= s < 42.9    980 < p
          Two         2     96 <= s < 110      42.9 <= s < 49.2    965 <= p < 980  
         Three        3    110 <= s < 130      49.2 <= s < 58.1    944 <= p < 965
         Four         4    130 <= s < 155      58.1 <= s < 69.3    920 <= p < 944
         Five         5    155 <= s            69.3 <= s                  p < 920
    """
    if units not in {'mph','m/s'}:
        print('unknown units: setting ss to "XX"')
        return 'XX'
    if units == 'mph':
        if spd < 39: return 'TD'
        if spd < 74: return 'TS'
    elif units == 'm/s':
        if spd < 17.4: return 'TD'
        if spd < 33.1: return 'TS'
    return 'HU'


def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map
    """

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def TrackPlot(df, extent=None, axx=None, fname=None, circ=None, addcolorbar=True, norm=False):
    """
    """
    returnFigAx=False
    if axx is None:
        fig, axx = plt.subplots(nrows=1, ncols=1, constrained_layout=True, dpi=72,  figsize=(12, 12))
        returnFigAx=True
            
    IDX=np.unique(df.index).astype(int)
    
    for i,idx in enumerate(IDX): 
        x=df.loc[df.index==idx].Longitude.values
        y=df.loc[df.index==idx].Latitude.values
        c=df.loc[df.index==idx].DeltaP.values
       
        #c=df.loc[df.index==idx].HollandB
        #axx.plot(x, y, linewidth=.1, color='k')
        #axx.plot(x.iloc[0], y.iloc[0], marker='*', color='g')
        #axx.plot(x.iloc[-1], y.iloc[-1], marker='*', color='r')
        
        # whack NaNs
        if np.all(np.isnan(c)):  # skip if all values are NaN
            continue #print(f'{i,idx,x[0],y[0],c[0]}')
            
        if norm is False:
            cm=axx.scatter(x=x, y=y, c=c, cmap=cmap, s=10, transform=ccrs.PlateCarree())
        else:
            cm=axx.scatter(x=x, y=y, c=c, cmap=cmap, norm=norm, s=10, transform=ccrs.PlateCarree())
               
    if circ is not None:  axx.plot(circ['cirx'],circ['ciry'],linewidth=2, color='k')
    
    if addcolorbar:
        cb1 = plt.colorbar(cm, ax=axx, orientation='vertical', pad=0.02, aspect=15) # , shrink=0.15)
        cb1.ax.set_ylabel('[mb]', size=12)
        cb1.ax.tick_params(labelsize='large')
    
    # axx.plot(coastline[:,0],coastline[:,1],color='k',linewidth=.25)
    # axx.plot(statelines[:,0],statelines[:,1],color='k',linewidth=.25)
   
    if extent is not None: 
        axx.axis('equal')
        axx.set_xlim(extent[0:2])
        axx.set_ylim(extent[2:4])

    axx.grid(True)

    if fname is not None: fig.savefig(fname)
    
    if returnFigAx: return fig, axx
    
def fullTrackPlot(dfnc, extentnc, nc_circ, dftx, extenttx, tx_circ, fname=None):
    """
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, 
                           subplot_kw={'projection': ccrs.PlateCarree()},
                           constrained_layout=True, figsize=(24, 12)) # dpi=144,)

    #fig = plt.figure(figsize=(16, 10))
    
    #ax[0] = plt.axes(projection=ccrs.PlateCarree())
    ax[0].stock_img()
    ax[0].coastlines()
    #ax[1] = plt.axes(projection=ccrs.PlateCarree())
    ax[1].stock_img()
    ax[1].coastlines()
    
    TrackPlot(dftx, extent=extenttx, axx=ax[0], circ=tx_circ, addcolorbar=False)
    TrackPlot(dfnc, extent=extentnc, axx=ax[1], circ=nc_circ)

    if fname is not None: fig.savefig(fname)
                
    return fig, ax

def LoadPEPC(pathtopepcfiles,setnums=None):
    '''
    Loads the Princeton PEPC tracks into df with variable names equivalent to the loadIBTracs and LoadSTORM
    Longitudes are translated to negative west convention
    Units for wind etc are currently ambiguous, so no converstions are made
    '''
    
    if not os.path.isdir(pathtopepcfiles): 
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), pathtopepcfiles)
        
    storms=[]
    
    if setnums is None:
        setnums=[1]
    elif isinstance(setnums, list):
        #print('setnums must be a list')
        setnums=np.array(setnums)
    elif isinstance(setnums,int):
        # load first setnums sets
        setnums=np.arange(1,setnums+1)
    elif isinstance(setnums,str):
        if setnums.lower() != 'all': 
            print('if setnums is a string, it must be "all".')
            return
        setnums=np.arange(1,101)
    else:
        print('unknown setnums passed to LoadPEPC.')
        return
        
    setnums=np.array(setnums)

    #if setnums.min() < 1 || setnums.max()>101:
    ny=39
    
    for j,setnum in enumerate(setnums):

        fl=f'{pathtopepcfiles}/selected25_simSS_g{setnum:03d}/lon.csv'
        print(fl)
                
        lon=pd.read_csv(fl,header=None)
        lat=pd.read_csv(fl.replace('lon','lat'),header=None)
        year=pd.read_csv(fl.replace('lon','year'),header=None)+ j*ny  # add year offset 
        month=pd.read_csv(fl.replace('lon','month'),header=None)
        day=pd.read_csv(fl.replace('lon','day'),header=None)
        hour=pd.read_csv(fl.replace('lon','hour'),header=None)
        wind=pd.read_csv(fl.replace('lon','wind'),header=None)
        mld=pd.read_csv(fl.replace('lon','mld'),header=None)
        mpi=pd.read_csv(fl.replace('lon','mpi'),header=None)
        strat=pd.read_csv(fl.replace('lon','strat'),header=None)
        tspd=pd.read_csv(fl.replace('lon','tspd'),header=None)
        shr=pd.read_csv(fl.replace('lon','shr'),header=None)
        rhhi=pd.read_csv(fl.replace('lon','rhhi'),header=None)

        ntimes, nstorms = lon.shape

        for i in range(nstorms):
            lon2=lon.loc[:,i].values
            lon2 = np.where(lon2> 180, lon2-360, lon2)
            lat2=lat.loc[:,i].values
            year2=year.loc[:,i].values 
            month2=month.loc[:,i].values
            day2=day.loc[:,i].values
            hour2=hour.loc[:,i].values
            wind2=wind.loc[:,i].values
            mld2=mld.loc[:,i].values
            mpi2=mpi.loc[:,i].values
            strat2=strat.loc[:,i].values
            tspd2=tspd.loc[:,i].values
            shr2=shr.loc[:,i].values
            rhhi2=rhhi.loc[:,i].values

            abssn=(setnum-1)*1000+i*np.ones(lon2.shape[0])

            storm=pd.DataFrame({'abssn':abssn.astype(int), 'Longitude':lon2, 'Latitude':lat2, 'Year':year2, 'Month':month2, 
                                'Day':day2, 'Hour':hour2, 'wind':wind2, 'mld':mld2, 'mpi':mpi2, 'tspd': tspd2, 'shr':shr2, 'rhhi':rhhi2})
            storm.dropna(inplace=True)

            storms.append(storm)

    storms2=pd.concat(storms)
    storms2.set_index('abssn',inplace=True)
    return storms2

def LoadSTORMtracks(basin='NA',ensnum=0,climate='current',model='present',version='_V4',nyears=None,startingyear=None):
    """
    Returns STORM tracks in a dataframe. 
    
    Longitudes are converted to -180->180.  
    Day, Hour columns are added.
    An "absolute storm number" (abssn) is computed and set as the df index.
    TC_number|Time_step|Category|Landfall columns are dropped.
    
    Parameters:
        basin (str): Basin, 2-char, def = 'NA'  {only North Atlantic for now}
        ensnum (int): 1000-yr chunk to read, default 0 (first 1000 yrs), 0-10
        climate (str): climate spec, def = 'current', {'current','future'}
        model (str): model spec, def = 'present', {'present', 'CMCC', 'CNRM', 'ECEARTH', 'HADGEM'}
        version (str): STORM version tag, def = '_V4' {don't change}
        nyears (int): = number of years to return, from ensnum chunk, def = None (all years)
        startingyear (int): starting year to return, of nyears, def = None (random start year if nyears not None)
    
    Returns:
        dataframe: of STORM Tracks, with these columns: 
            Year	Month	Day	Hour	Basin_ID	Latitude	Longitude	Min_pres	MaxWindSpd	RMW	Dist2land
    
    """
        
    baseurl='https://tdsres.apps.renci.org/thredds/fileServer/datalayers/STORM_Bloemendaal_data'

    # column def of STORM files
    cols=[
        'Year',       # Starts at 0
        'Month', 
        'TC_number',  # For every year; starts at 0.
        'Time_step',  # 3-hr, For every TC; starts at 0.
        'Basin_ID',   # 0=EP, 1=NA, 2=NI, 3=SI, 4=SP, 5=WP
        'Latitude',   # Deg, Position of the eye.
        'Longitude',  # Deg, Position of the eye. Ranges from 0-360°, with prime meridian at Greenwich.
        'Min_pres',   # hPa [mb]
        'MaxWindSpd', # m/s
        'RMW',        # km
        'Category',   #
        'Landfall',   # 0= no landfall, 1= landfall
        'Dist2land'   # km
    ]

    # future model dict
    mod_dict= {'CMCC': 'CMCC-CM2-VHR4',
               'CNRM': 'CNRM-CM6-1-HR',
               'ECEARTH': 'EC-Earth3P-HR',
               'HADGEM': 'HadGEM3-GC31-HM'}

    if climate not in {'current','future'}:
        raise Exception(f'climate must be "current" or "future".')
    if model != 'present':
        if model not in mod_dict.keys():
            raise  Exception(f'model must be in {mod_dict.keys()}.')
    
    if climate == 'current':
        url=f'{baseurl}/present{version}/STORM_DATA_IBTRACS_{basin}_1000_YEARS_{ensnum}.txt'
    else:
        url=f'{baseurl}/future/{model}/STORM_DATA_{mod_dict[model]}_{basin}_1000_YEARS_{ensnum}_IBTRACSDELTA.txt'

    print(f'Reading STORM tracks from {url}')

    df=pd.read_csv(url, names=cols)

    # generate an "absolute storm number (abssn)" to uniquely identify each storm 
    # in the dataset, then set that to be the dataframe index
    df['abssn']=np.cumsum(1*(df.Time_step==0))
    df.set_index('abssn',inplace=True)
    
    #idx_all=np.unique(df.index).astype(int)

    # add a Day, Hour columns
    day=np.floor(df['Time_step']*3/24+1)
    df.insert(2, 'Day', day)
    hour=24*(df['Time_step']*3/24+1-df['Day'])
    df.insert(3, 'Hour', hour)

    #df['Longitude']=df['Longitude']-360
    df['Longitude'] = np.where(df['Longitude'] > 180, df['Longitude']-360, df['Longitude'])
    #df=df.sortby(dsout.longitude)
    
    df=df[df.columns.drop(list(df.filter(regex='TC_number|Time_step|Category|Landfall')))]

    if nyears:
        if not startingyear:
            # random starting year
            startingyear=np.random.randint(np.min(df.Year),np.max(df.Year)-nyears)
            
        print(f'Starting year = {startingyear}')
        df=df.loc[(df['Year'] >= startingyear) & (df['Year'] < startingyear+nyears)]
    
    return df

def LoadIBTrACS(minyear=None, maxyear=None):
    """
    Returns a dataframe with IBTrACS 
    
    minyear=None
    maxyear=None
    
    RMW and MaxWindSpd are converted to MKS
    
    """
    
    fl='https://tdsres.apps.renci.org/thredds/fileServer/datalayers/ibtracs/ibtracs.NA.list.v04r00.csv'
    
    dropcols=['SUBBASIN','USA_LAT','USA_LON','USA_AGENCY','IFLAG','LANDFALL',
              'TRACK_TYPE','WMO_AGENCY','WMO_WIND','WMO_PRES','USA_SEAHGT','USA_SEARAD_SW',
              'USA_SEARAD_NW','USA_SEARAD_NE','USA_SEARAD_SE',
              'USA_R64_SE','USA_R64_SW','USA_R64_NW','USA_POCI','USA_ROCI',
              'USA_R50_NE','USA_R50_SE','USA_R50_SW','USA_R50_NW','USA_R64_NE',
              'USA_SSHS','USA_R34_NE','USA_R34_SE','USA_R34_SW','USA_R34_NW','USA_RECORD','USA_EYE','USA_GUST'];

    renamecols={'SEASON': 'Year',
                'LAT': 'Latitude',
                'LON': 'Longitude',
                'USA_PRES': 'Min_pres', 
                'USA_WIND': 'MaxWindSpd', 
                'USA_RMW': 'RMW',
                #'LANDFALL': 'Landfall',
                'DIST2LAND': 'Dist2land',
                'BASIN': 'Basin_ID'}

    df=pd.read_csv(fl,skiprows = [1],low_memory=False).drop(dropcols,axis=1).replace(' ', np.nan)
    
    # drop other agency reports
    df=df[df.columns.drop(list(df.filter(regex='TOKYO|BOM|REUNION|WELLINGTON|CMA|NEUMANN|TD9636|DS824|NEWDELHI|HKO|TD9635|MLC|NADI')))]
    
    df.USA_PRES=pd.to_numeric(df.USA_PRES)
    df.USA_WIND=pd.to_numeric(df.USA_WIND)
    df.USA_RMW=pd.to_numeric(df.USA_RMW)

    df.rename(columns=renamecols,inplace=True)
    df=df[df['Basin_ID']!='NI']
    df=df[df['Basin_ID']!='EP']
    df['Basin_ID']='NA'

    PD_TIME=pd.DatetimeIndex(df.ISO_TIME)
    df['Month'] = pd.DatetimeIndex(PD_TIME).month
    df['Day']   = pd.DatetimeIndex(PD_TIME).day
    df['Hour']  = pd.DatetimeIndex(PD_TIME).hour

    #df['abssn']=np.cumsum(1*(df.Hour==0))
    #df.set_index('abssn',inplace=True)
    
    df2=df.copy()
    df2['abssn'] = -1
    unique_ids=df2['USA_ATCF_ID'].unique()

    for i,j in enumerate(unique_ids):
        idx=df2['USA_ATCF_ID']==j
        df2['abssn'].loc[idx]=i
    df=df2
    
    df.set_index('abssn',inplace=True)   
    
    df = df[['Year', 'Month', 'Day', 'Hour', 
             'Basin_ID', 'Latitude', 'Longitude', 
             'Min_pres', 'MaxWindSpd', 'RMW', 'Dist2land', 
             'NATURE', 'USA_ATCF_ID', 'SID', 'USA_STATUS']]
    
    if minyear:
        df=df.loc[(df['Year'] >= minyear)]
    if maxyear:
        df=df.loc[(df['Year'] <= maxyear)]
    
    df['RMW']=df['RMW']*1.852   # nm to km
    df['MaxWindSpd']=df['MaxWindSpd']*0.514444 # kts to m/s

    return df

def get_genesis_locations(df):
    '''
    returns a df of genesis positions, i.e., the first position in each track
    '''
    
    glo=[]
    gla=[]
    gdp=[]

    IDX=np.unique(df.index).astype(int)
    
    for i,idx in enumerate(IDX): 
        glo.append(df.loc[df.index==idx].Longitude.values[0])
        gla.append(df.loc[df.index==idx].Latitude.values[0])
    data = {'Longitude': glo, 'Latitude': gla}
    
    if 'DeltaP' in df.keys():
        for i,idx in enumerate(IDX): 
            gdp.append(df.loc[df.index==idx].DeltaP.values[0])
        data['DeltaP']=gdp
        
    return pd.DataFrame.from_dict(data)

def get_counts_per_year(grid_dict,df):
    """
    returns matrix of counts of df in grid_dict
    """
    
    nx=grid_dict['lon_bins'].shape[0]
    ny=grid_dict['lat_bins'].shape[0]
    counts=np.zeros((ny-1,nx-1))
    
    for i in range(ny-1):

        latmin=grid_dict['lat_bins'][i]
        latmax=grid_dict['lat_bins'][i+1]

        for j in range(nx-1):

            lonmin=grid_dict['lon_bins'][j]
            lonmax=grid_dict['lon_bins'][j+1]
            
            df_temp=df.loc[(df['Latitude']>=latmin)  &  (df['Latitude']<latmax) 
                         & (df['Longitude']>=lonmin) &  (df['Longitude']<lonmax)]
            
            IDX=np.unique(df_temp.index).astype(int)
            counts[i][j]=IDX.shape[0]
            #if counts[i][j]>0:print(lonmin,lonmax,latmin,latmax,int(counts[i][j]))

    number_of_years=(df["Year"].max()-df["Year"].min())+1
    print(f'number_of_years={number_of_years}')
    counts=counts/number_of_years
    counts=np.where(counts < .1, np.nan, counts)

    return counts

def storm_stall_nws67(dfin):
    """
    adds a stalled period at the end of storm
    
    """
    dfout=dfin.copy()
    stlen=10 # days
    dt=3 # hours
    decay=24 # hours
    ll=stlen*int(24/dt)
    lt=dfin.index[-1]
    lo=dfin['Longitude'].iloc[-1]
    la=dfin['Latitude'].iloc[-1]
    hb=dfin['HollandB'].iloc[-1]
    newt=dfin.index.tolist()

    for l in range(ll):

        nt=lt+(l+1)*dt/(24*365);

        fac=np.max([(decay-(l+1)*dt)/decay,0])

        dp =dfin['Dp'].iloc[-1]*fac
        du =dfin['du'].iloc[-1]*fac
        dv =dfin['dv'].iloc[-1]*fac
        rmw=dfin['RMW'].iloc[-1]/(fac+.1)
        dfout.loc[len(dfout.index)] = [lo, la, dp, du, dv, rmw, hb] 
        newt.append(nt)

        #print(f"{l} {nt:6f} {lo:3f} {la:3f} {dp:5.0f} {du:6.2f} {dv:6.2f} {fac:3f} {rmw:5f} {hb:5f}")

    dfout.index = newt
    
    return dfout

def out_to_nws8(df,basin='AL',tau=0,advr=0,fname=None,stormname='unknown'):
    '''
        STORM data units are: 
            Min_pres   [hPa, same as mb]
            MaxWindSpd [m/s]
            RMW        [km]
    
        NWS8 units are:        
            Min_pres   [hPa, same as mb]
            MaxWindSpd [kt]
            RMW        [ni]

    #b-deck and NWS8 format lineup

    0        1         2         3         4         5         6         7         8         9        10        11        12        13        14        15        16
    123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012
    xxxxxxxxiiiiiiiiiixxxxxxaaaaxxiiixiiiiaxxiiiiaxxiiixxiiiixxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxiiixxiii
          8x  i4i2i2i2    6x  a42x i3x  i4a2x  i4a2x i32x  i4                                            47x i32x i3
              yrmodyhr      type, inc, latNS,lonEW, spd,   pc,                                              RRP, RMW

    BN, CY, YYYYMMDDHH, MIN,TECH, TAU,latNS, lonEW,VMAX, MSLP, TY, RAD,  WC, RAD1, RAD2, RAD3, RAD4, POUT, ROUT, RMW,GUST, EYE,  SR,MAXS,INIT, DIR, SPD,  STORMNAME, 
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    AL, 13, 2003090700,   , BEST,   0, 135N,  358W,  55,  994, TS,  34, NEQ,   75,   75,   75,   75, 1012,  150,  25,   0,   0,
    AL, 06, 2000010303, 00, BEST,   0, 291N,  673W,  54,  944, HU,    ,    ,     ,     ,     ,     ,     ,     ,    ,    ,    ,    ,    ,    ,    ,    ,      x1454,
    AL, 00, 0990080121,   , BEST,   0, 216N,  948W,  46,  952, HU,    ,    ,     ,     ,     ,     ,     ,     ,    ,    ,    ,    ,    ,    ,    ,    ,    x10650,


    https://www.nrlmry.navy.mil/atcf_web/docs/database/new/abdeck.txt
    '''
    
    if not fname: return

    f = open(fname, "w")

    for index, row in df.iterrows():
        
        date=f"{row['Year']:04n}{row['Month']:02n}{row['Day']:02n}{row['Hour']:02n}"
        ilat=round(row['Latitude']*10)
        ilon=round(row['Longitude']*10)
        ispd=row['MaxWindSpd']  
        irmw=row['RMW']         
        icpress=round(row['Min_pres'])
        
        EW='E'
        if ilon<0:EW='W'
        NS='N'
        if ilat<0:NS='S'

        TY=ss_scale(ispd)
        
        ispd=round(ispd/0.51444444)         # conv to kts
        irmw=round(irmw/1.85200000)         # conv to ni

        outs=f"{basin}, {advr:02n}, {date},   , BEST,{tau:4n},{abs(ilat):4n}{NS},{abs(ilon):5n}{EW},{ispd:4n},{icpress:5d}, {TY:2s},    ,    ,     ,     ,     ,     ,     ,     ,{irmw:4n},    ,    ,    ,    ,    ,    ,    ,{stormname:>10s},\n"
    
        f.write(outs) 
        
    f.close()


cmap_N=14
cmap=discrete_cmap(cmap_N, 'jet_r')
norm_pres = mpl.colors.Normalize(vmin=880, vmax=1024)
norm_dpres = mpl.colors.Normalize(vmin=0, vmax=100)
#norm = mpl.colors.Normalize(vmin=0, vmax=2)