# CHEER-STORM
repo for codes/processing of STORM datasets for CHEER

Binder Notebook: 
https://mybinder.org/v2/gh/BrianOBlanton/CHEER-STORM/HEAD?labpath=STORM_LoadDemo-NoCartopy.ipynb

Draft track files in CoPe Google Drive: 

https://drive.google.com/drive/folders/1dPeUHDHJePOlrVup2u2e3Pp9c7a8x88U?usp=share_link

Trackfile naming: 
<Region>_<Climate>_<EnsembleSetNumber>_<AbsoluteStormNumber>.csv


Track file header: 

cols=[
    'Year',       # Starts at 0
    'Month', 
    'TC_number',  # For every year; starts at 0.
    'Time_step',  # 3-hr, For every TC; starts at 0.
    'Basin_ID',   # 0=EP, 1=NA, 2=NI, 3=SI, 4=SP, 5=WP
    'Latitude',   # Deg, Position of the eye.
    'Longitude',  # Deg, Position of the eye. Ranges from 0-360°, with prime meridian at Greenwich.
    'Min_pres',   # hPa
    'MaxWindSpd', # m/s
    'RMW',        # km
    'Category',   #
    'Landfall',   # 0= no landfall, 1= landfall
    'Dist2land',  # km
    'dist2nc',    # distance to NC screening circle center [deg]
    'dist2tx'     # distance to TX screening circle center [deg]
    ]


