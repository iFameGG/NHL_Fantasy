# Fantasy Points Prediction Model *by Antonio Nardi*

## Goal

The goal of this project is to create a model that will predict the amount of fantasy points a player will produce, given certain non related-variables. A non-related variable is a variable that doesn't directly affect the base fantasy league's formula for points. Which includes variables that directly result in a goal or an assist. I have made an exception for shots, even though they're included in the formula, because they help the model explain more of the data.

```
FANTASY_POINTS = goals*3 + assists*2 + power_play_points*1 + short_handed_points*2 + hat_tricks*1 + shots*0.1 + hits*0.1 + blocked_shots*0.2
```

# Setup

## Installs


```python
!pip install html5lib -q
!pip install statsmodels -q
!pip install pandas -q
!pip install numpy -q
!pip install seaborn -q
!pip install scikit-learn -q
!pip install matplotlib -q
```

## Imports


```python
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

import datetime
import time
from functools import reduce
```

# Data

## Define SeasonData Object


```python
class SeasonsData:
    def __init__(self, years):
        self.years = years
        self.season_tables = {year: self.get_season_tables(year) for year in self.years}
        self.season_datasets = {year: self.get_season_datasets(self.season_tables[year]) for year in self.years}
        self.complete_dataset =  self.get_complete()
        self.shape = self.complete_dataset.shape
        
    def get_season_tables(self, year):
        """
        Get each table from hockey-reference
        season databases
        """

        # -- Define URLS --

        ref_url = 'https://www.hockey-reference.com'

        basic_ext = f'/leagues/NHL_{year}_skaters.html'
        adv_ext = f'/leagues/NHL_{year}_skaters-advanced.html'
        shift_ext = f'/leagues/NHL_{year}_skaters-time-on-ice.html'
        
        #### HAD TO REMOVE MISC BECAUSE IT WASNT TRACKED FOR MOST YEARS
        # misc_ext = f'/leagues/NHL_{year}_skaters-misc.html'

        url_names = ['basic', 'adv', 'shift',]
        url_ext = [basic_ext, adv_ext, shift_ext]

        # --- Get Tables ---

        raw_tables = {}
        print(f'Grabbing {year} season data')
        for name, ext in zip(url_names, url_ext):
            req = False
            fails = 0
            while req == False:
                sleep_time = 2**fails
                try:
                    df = pd.read_html(ref_url+ext)[0].apply(lambda x: x.astype(str).str.upper())
                    req = True
                except:
                    fails += 1
                    print(fails, 'fails')
                    time.sleep(sleep_time)
                    pass
            li = list(df.columns)
            col = [ x[1] for x in li ]
            df.columns = col
            df = df[df['Rk'] != 'RK'].reset_index(drop=True).sort_values(by='Player')
            tot_tm = df[df['Tm'] == 'TOT']
            no_dupes = df.drop_duplicates(subset='Player', keep=False)
            df = pd.concat([no_dupes, tot_tm])
            df_dict = {name: df}
            raw_tables.update(df_dict)
            print(f'- Grabbed {name}')
        
        return raw_tables

    def get_season_datasets(self, raw_tables):
        """
        Cleans columns of each dataframe and 
        merges all dataframes into one complete 
        dataframe
        """

        # Basic Dataframe
        basic_df = raw_tables['basic'].drop(['FO%'], axis=1)
        basic_df.columns =    [
                                'Rk', 'Player', 'Age', 'Tm', 'Pos', 'GP', 'G', 'A', 'PTS', '+/-', 'PIM',
                                'PS', 'EV_G', 'PP_G', 'SH_G', 'GW_G', 'EV_A', 'PP_A', 'SH_A', 'S', 'S%', 'TOI',
                                'ATOI', 'BLK', 'HIT', 'FOW', 'FOL'
                            ]
        raw_tables['basic'] = basic_df

        # Adv Dataframe
        adv_df = raw_tables['adv'].drop(['GP', 'Age', 'Rk', 'Pos'], axis=1)
        try:
            adv_df = adv_df.drop(['E+/-'], axis=1)
        except:
            pass
        adv_df.columns =    [
                                'Player', 'Tm', 'CF', 'CA', 'CF%', 'CF% rel',
                                'FF', 'FA', 'FF%', 'FF% rel', 'oiSH%', 'oiSV%', 'PDO', 'oZS%', 'dZS%',
                                'TOI/60', 'TOI(EV)', 'TK', 'GV', 'SAtt.', 'Thru%'
                            ]
        raw_tables['adv'] = adv_df

        # Shift Dataframe
        shift_df = raw_tables['shift'].drop([
            'GP', 'Unnamed: 6_level_1', 'Unnamed: 11_level_1', 'Unnamed: 16_level_1', 'Rk', 'Pos'
        ], axis=1)
        shift_df.columns =  [
                                'Player', 'Tm', 'Shift', 'TOI_EVEN',
                                'CF% Rel_EVEN', 'GF/60_EVEN', 'GA/60_EVEN', 'TOI_PP', 'CF% Rel_PP',
                                'GF/60_PP', 'GA/60_PP', 'TOI_SH', 'CF% Rel_SH', 'GF/60_SH',
                                'GA/60_SH'
                            ]
        raw_tables['shift'] = shift_df
        
        #### HAD TO REMOVE MISC BECAUSE IT WASNT TRACKED FOR MOST YEARS
        # Misc Dataframe
        # misc_df = raw_tables['misc'].drop([
        #     'GP', 'Age', 'Made', 'Miss', 'Pct.', 'Rk', 'Pos', '+/-', 'PS'
        # ], axis=1)
        # try:
        #     misc_df = misc_df.drop(['xGF', 'xGA', 'E+/-'], axis=1)
        # except:
        #     pass
        # misc_df.columns  =  [
        #                         'Player', 'Tm', 'GC', 'G_PG', 'A_PG', 'PTS_PG', 'GC_PG',
        #                         'PIM_PG', 'S_PG', 'G_ADJ', 'A_ADJ', 'PTS_ADJ', 'GC_ADJ', 'TGF', 'PGF', 
        #                         'TGA', 'PGA', 'OPS', 'DPS', 'Att.'
        #                     ]
        # raw_tables['misc'] = misc_df

        # Full Dataframe
        full_data = reduce(
            lambda  left,right: pd.merge(
                left,right,on=['Player', 'Tm'],
                how='outer'
            ), [ raw_tables[x] for x in raw_tables.keys() ]
        ).drop_duplicates(subset='Player', keep='first').dropna(subset=['Pos'])
        
        full_data = full_data.replace({
            'C': 'F', 
            'LW': 'F', 
            'RW': 'F', 
            'W': 'F', 
            'NAN': np.nan, 
            '': np.nan
        }).dropna()
        
        # --- Calculate Fantasy Points ---

        g_pts = 3
        a_pts = 2
        pp_pts = 1
        sh_pts = 2
        hat_pts = 1
        sog_pts = 0.1
        hts_pts = 0.1
        blk = 0.2
        
        full_data['FTSY_PTS'] = round(sum([
            (full_data['G'].astype(int)*g_pts), 
            (full_data['A'].astype(int)*a_pts), 
            (full_data['PP_G'].astype(int)*pp_pts), 
            (full_data['SH_G'].astype(int)*sh_pts), 
            # (full_data['HAT']*hat_pts), 
            (full_data['S'].astype(int)*sog_pts), 
            (full_data['HIT'].astype(int)*hts_pts), 
            (full_data['BLK'].astype(int)*blk)
        ]),1)

        return full_data
    
    def get_sec(self, time_str):
        """Get seconds from minutes and seconds."""
        m, s = time_str.split(':')
        return float(int(m) * 60 + int(s))
    
    def get_complete(self):
        complete_df = pd.concat(self.season_datasets.values()).drop(['Rk'], axis=1)
        
        for col in complete_df:
            first = complete_df.loc[0, col].tolist()[0]
            try:
                float(first)
                complete_df[col] = complete_df[col].astype(float)
            except:
                complete_df[col] = complete_df[col].astype(str)
        
        time_cols = ['ATOI', 'TOI/60', 'TOI(EV)', 'Shift', 'TOI_EVEN', 'TOI_PP', 'TOI_SH']
        for col in time_cols:
            complete_df[col] = complete_df[col].apply(lambda time_str: self.get_sec(time_str))
        
        return complete_df.reset_index(drop=True)
    
    def __repr__(self):
        return repr(self.complete_dataset)
```

## Initialize SeasonData Object

Getting player data from 2008 to the present day would be the ideal dataset, but I keep getting an error:
``` Python
HTTPError: HTTP Error 429: Too Many Requests
```
To get arround this, I will be using a smaller dataset.


```python
years = list(range(2014, datetime.date.today().year))
data = SeasonsData(years)
```

    Grabbing 2014 season data
    - Grabbed basic
    - Grabbed adv
    - Grabbed shift
    Grabbing 2015 season data
    - Grabbed basic
    - Grabbed adv
    - Grabbed shift
    Grabbing 2016 season data
    - Grabbed basic
    - Grabbed adv
    - Grabbed shift
    Grabbing 2017 season data
    - Grabbed basic
    - Grabbed adv
    - Grabbed shift
    Grabbing 2018 season data
    - Grabbed basic
    - Grabbed adv
    - Grabbed shift
    Grabbing 2019 season data
    - Grabbed basic
    - Grabbed adv
    - Grabbed shift
    Grabbing 2020 season data
    - Grabbed basic
    - Grabbed adv
    - Grabbed shift
    Grabbing 2021 season data
    - Grabbed basic
    - Grabbed adv
    - Grabbed shift
    Grabbing 2022 season data
    - Grabbed basic
    - Grabbed adv
    - Grabbed shift


## Test Attributes and Methods


```python
data.shape
```




    (4880, 59)



59 columns


```python
data.complete_dataset.query('Player == "CONNOR MCDAVID"')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player</th>
      <th>Age</th>
      <th>Tm</th>
      <th>Pos</th>
      <th>GP</th>
      <th>G</th>
      <th>A</th>
      <th>PTS</th>
      <th>+/-</th>
      <th>PIM</th>
      <th>PS</th>
      <th>EV_G</th>
      <th>PP_G</th>
      <th>SH_G</th>
      <th>GW_G</th>
      <th>EV_A</th>
      <th>PP_A</th>
      <th>SH_A</th>
      <th>S</th>
      <th>S%</th>
      <th>TOI</th>
      <th>ATOI</th>
      <th>BLK</th>
      <th>HIT</th>
      <th>FOW</th>
      <th>FOL</th>
      <th>CF</th>
      <th>CA</th>
      <th>CF%</th>
      <th>CF% rel</th>
      <th>FF</th>
      <th>FA</th>
      <th>FF%</th>
      <th>FF% rel</th>
      <th>oiSH%</th>
      <th>oiSV%</th>
      <th>PDO</th>
      <th>oZS%</th>
      <th>dZS%</th>
      <th>TOI/60</th>
      <th>TOI(EV)</th>
      <th>TK</th>
      <th>GV</th>
      <th>SAtt.</th>
      <th>Thru%</th>
      <th>Shift</th>
      <th>TOI_EVEN</th>
      <th>CF% Rel_EVEN</th>
      <th>GF/60_EVEN</th>
      <th>GA/60_EVEN</th>
      <th>TOI_PP</th>
      <th>CF% Rel_PP</th>
      <th>GF/60_PP</th>
      <th>GA/60_PP</th>
      <th>TOI_SH</th>
      <th>CF% Rel_SH</th>
      <th>GF/60_SH</th>
      <th>GA/60_SH</th>
      <th>FTSY_PTS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>720</th>
      <td>CONNOR MCDAVID</td>
      <td>19.0</td>
      <td>EDM</td>
      <td>F</td>
      <td>45.0</td>
      <td>16.0</td>
      <td>32.0</td>
      <td>48.0</td>
      <td>-1.0</td>
      <td>18.0</td>
      <td>5.8</td>
      <td>13.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>21.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>105.0</td>
      <td>15.2</td>
      <td>850.0</td>
      <td>1133.0</td>
      <td>10.0</td>
      <td>12.0</td>
      <td>249.0</td>
      <td>355.0</td>
      <td>709.0</td>
      <td>629.0</td>
      <td>53.0</td>
      <td>6.2</td>
      <td>529.0</td>
      <td>472.0</td>
      <td>52.8</td>
      <td>6.9</td>
      <td>10.5</td>
      <td>87.7</td>
      <td>98.2</td>
      <td>53.5</td>
      <td>46.5</td>
      <td>1133.0</td>
      <td>896.0</td>
      <td>33.0</td>
      <td>17.0</td>
      <td>156.0</td>
      <td>67.3</td>
      <td>42.0</td>
      <td>908.0</td>
      <td>6.2</td>
      <td>3.7</td>
      <td>3.7</td>
      <td>177.0</td>
      <td>5.7</td>
      <td>8.3</td>
      <td>0.5</td>
      <td>46.0</td>
      <td>7.1</td>
      <td>0.0</td>
      <td>3.8</td>
      <td>128.7</td>
    </tr>
    <tr>
      <th>1315</th>
      <td>CONNOR MCDAVID</td>
      <td>20.0</td>
      <td>EDM</td>
      <td>F</td>
      <td>82.0</td>
      <td>30.0</td>
      <td>70.0</td>
      <td>100.0</td>
      <td>27.0</td>
      <td>26.0</td>
      <td>12.8</td>
      <td>26.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>45.0</td>
      <td>24.0</td>
      <td>1.0</td>
      <td>251.0</td>
      <td>12.0</td>
      <td>1733.0</td>
      <td>1268.0</td>
      <td>29.0</td>
      <td>34.0</td>
      <td>348.0</td>
      <td>458.0</td>
      <td>1479.0</td>
      <td>1272.0</td>
      <td>53.8</td>
      <td>5.6</td>
      <td>1112.0</td>
      <td>917.0</td>
      <td>54.8</td>
      <td>5.6</td>
      <td>10.9</td>
      <td>91.0</td>
      <td>101.9</td>
      <td>56.2</td>
      <td>43.8</td>
      <td>1268.0</td>
      <td>1031.0</td>
      <td>76.0</td>
      <td>54.0</td>
      <td>420.0</td>
      <td>59.8</td>
      <td>51.0</td>
      <td>1037.0</td>
      <td>5.6</td>
      <td>3.7</td>
      <td>2.6</td>
      <td>182.0</td>
      <td>-1.9</td>
      <td>9.8</td>
      <td>0.9</td>
      <td>48.0</td>
      <td>-2.6</td>
      <td>1.8</td>
      <td>7.3</td>
      <td>269.3</td>
    </tr>
    <tr>
      <th>1912</th>
      <td>CONNOR MCDAVID</td>
      <td>21.0</td>
      <td>EDM</td>
      <td>F</td>
      <td>82.0</td>
      <td>41.0</td>
      <td>67.0</td>
      <td>108.0</td>
      <td>20.0</td>
      <td>26.0</td>
      <td>13.1</td>
      <td>35.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>49.0</td>
      <td>15.0</td>
      <td>3.0</td>
      <td>274.0</td>
      <td>15.0</td>
      <td>1767.0</td>
      <td>1293.0</td>
      <td>46.0</td>
      <td>28.0</td>
      <td>376.0</td>
      <td>533.0</td>
      <td>1643.0</td>
      <td>1442.0</td>
      <td>53.3</td>
      <td>3.8</td>
      <td>1259.0</td>
      <td>1066.0</td>
      <td>54.2</td>
      <td>4.4</td>
      <td>10.5</td>
      <td>90.5</td>
      <td>100.9</td>
      <td>55.6</td>
      <td>44.4</td>
      <td>1292.0</td>
      <td>1038.0</td>
      <td>111.0</td>
      <td>67.0</td>
      <td>433.0</td>
      <td>63.5</td>
      <td>58.0</td>
      <td>1049.0</td>
      <td>3.8</td>
      <td>4.0</td>
      <td>3.2</td>
      <td>177.0</td>
      <td>1.3</td>
      <td>5.3</td>
      <td>1.0</td>
      <td>65.0</td>
      <td>5.8</td>
      <td>3.0</td>
      <td>6.1</td>
      <td>303.4</td>
    </tr>
    <tr>
      <th>2490</th>
      <td>CONNOR MCDAVID</td>
      <td>22.0</td>
      <td>EDM</td>
      <td>F</td>
      <td>78.0</td>
      <td>41.0</td>
      <td>75.0</td>
      <td>116.0</td>
      <td>3.0</td>
      <td>20.0</td>
      <td>13.0</td>
      <td>31.0</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>9.0</td>
      <td>50.0</td>
      <td>24.0</td>
      <td>1.0</td>
      <td>240.0</td>
      <td>17.1</td>
      <td>1781.0</td>
      <td>1370.0</td>
      <td>30.0</td>
      <td>39.0</td>
      <td>428.0</td>
      <td>490.0</td>
      <td>1466.0</td>
      <td>1436.0</td>
      <td>50.5</td>
      <td>3.8</td>
      <td>1110.0</td>
      <td>1109.0</td>
      <td>50.0</td>
      <td>3.3</td>
      <td>12.2</td>
      <td>88.5</td>
      <td>100.7</td>
      <td>57.2</td>
      <td>42.8</td>
      <td>1370.0</td>
      <td>1127.0</td>
      <td>99.0</td>
      <td>89.0</td>
      <td>422.0</td>
      <td>56.6</td>
      <td>55.0</td>
      <td>1127.0</td>
      <td>3.8</td>
      <td>4.0</td>
      <td>3.8</td>
      <td>205.0</td>
      <td>8.1</td>
      <td>9.4</td>
      <td>0.7</td>
      <td>37.0</td>
      <td>13.7</td>
      <td>3.7</td>
      <td>12.3</td>
      <td>317.9</td>
    </tr>
    <tr>
      <th>3070</th>
      <td>CONNOR MCDAVID</td>
      <td>23.0</td>
      <td>EDM</td>
      <td>F</td>
      <td>64.0</td>
      <td>34.0</td>
      <td>63.0</td>
      <td>97.0</td>
      <td>-6.0</td>
      <td>28.0</td>
      <td>10.9</td>
      <td>23.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>31.0</td>
      <td>32.0</td>
      <td>0.0</td>
      <td>212.0</td>
      <td>16.0</td>
      <td>1399.0</td>
      <td>1312.0</td>
      <td>18.0</td>
      <td>37.0</td>
      <td>320.0</td>
      <td>350.0</td>
      <td>1049.0</td>
      <td>1118.0</td>
      <td>48.4</td>
      <td>1.0</td>
      <td>819.0</td>
      <td>872.0</td>
      <td>48.4</td>
      <td>-0.1</td>
      <td>11.4</td>
      <td>89.3</td>
      <td>100.7</td>
      <td>57.5</td>
      <td>42.5</td>
      <td>1312.0</td>
      <td>1051.0</td>
      <td>53.0</td>
      <td>75.0</td>
      <td>339.0</td>
      <td>62.5</td>
      <td>52.0</td>
      <td>1072.0</td>
      <td>1.0</td>
      <td>3.7</td>
      <td>3.7</td>
      <td>233.0</td>
      <td>-2.3</td>
      <td>12.4</td>
      <td>1.9</td>
      <td>6.0</td>
      <td>-10.5</td>
      <td>0.0</td>
      <td>2.7</td>
      <td>267.5</td>
    </tr>
    <tr>
      <th>3681</th>
      <td>CONNOR MCDAVID</td>
      <td>24.0</td>
      <td>EDM</td>
      <td>F</td>
      <td>56.0</td>
      <td>33.0</td>
      <td>72.0</td>
      <td>105.0</td>
      <td>21.0</td>
      <td>20.0</td>
      <td>13.0</td>
      <td>24.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>44.0</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>200.0</td>
      <td>16.5</td>
      <td>1241.0</td>
      <td>1329.0</td>
      <td>24.0</td>
      <td>61.0</td>
      <td>316.0</td>
      <td>322.0</td>
      <td>1066.0</td>
      <td>865.0</td>
      <td>55.2</td>
      <td>10.5</td>
      <td>810.0</td>
      <td>667.0</td>
      <td>54.8</td>
      <td>9.8</td>
      <td>12.1</td>
      <td>89.1</td>
      <td>101.2</td>
      <td>62.4</td>
      <td>37.6</td>
      <td>1329.0</td>
      <td>1069.0</td>
      <td>36.0</td>
      <td>47.0</td>
      <td>314.0</td>
      <td>63.7</td>
      <td>57.0</td>
      <td>1071.0</td>
      <td>10.5</td>
      <td>4.5</td>
      <td>3.2</td>
      <td>251.0</td>
      <td>6.9</td>
      <td>11.9</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>39.1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>282.9</td>
    </tr>
    <tr>
      <th>4305</th>
      <td>CONNOR MCDAVID</td>
      <td>25.0</td>
      <td>EDM</td>
      <td>F</td>
      <td>80.0</td>
      <td>44.0</td>
      <td>79.0</td>
      <td>123.0</td>
      <td>28.0</td>
      <td>45.0</td>
      <td>14.0</td>
      <td>34.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>44.0</td>
      <td>34.0</td>
      <td>1.0</td>
      <td>314.0</td>
      <td>14.0</td>
      <td>1765.0</td>
      <td>1324.0</td>
      <td>26.0</td>
      <td>75.0</td>
      <td>562.0</td>
      <td>484.0</td>
      <td>1670.0</td>
      <td>1265.0</td>
      <td>56.9</td>
      <td>7.2</td>
      <td>1278.0</td>
      <td>987.0</td>
      <td>56.4</td>
      <td>7.9</td>
      <td>10.1</td>
      <td>91.0</td>
      <td>101.1</td>
      <td>56.3</td>
      <td>43.7</td>
      <td>1324.0</td>
      <td>1088.0</td>
      <td>73.0</td>
      <td>68.0</td>
      <td>481.0</td>
      <td>65.5</td>
      <td>59.0</td>
      <td>1088.0</td>
      <td>7.2</td>
      <td>4.0</td>
      <td>2.7</td>
      <td>223.0</td>
      <td>5.1</td>
      <td>10.7</td>
      <td>0.8</td>
      <td>12.0</td>
      <td>6.2</td>
      <td>3.7</td>
      <td>3.7</td>
      <td>344.1</td>
    </tr>
  </tbody>
</table>
</div>



# Preprocess Data

The X dataframe is the independant variable and y is the dependent variable.


```python
dataset = data.complete_dataset

# Inputs
X = dataset.drop(['Player', 'Tm'], axis=1)

# Output
y = dataset['FTSY_PTS']
```

## Preprocess X

### Drop Variables associated with FTSY_PTS

The goal of this model is to not rely on the variables, which are directly connected to the production of points (fantasy system). Thus, I removed the variables which are connected to the fantasy points formula: goals, assists, power play goals, power play assists, short handed goals, short handed assists, even goals, even assists, points share, game winning goals, blocks, hits, points (nhl system).


```python
to_drop = [
    'G', 'A', 'PP_G','SH_G', 'GW_G', 'EV_G', 'PS',
    'PP_A', 'SH_A', 'BLK','HIT', 'EV_A', 'PTS','S%'
]
X = X.drop(to_drop, axis=1)
```

### Get Dummies For Pos Column

Since, we cannot input categorical variables into the model, I created columns which determines the player's position as a 1 or 0. We drop the first column after creating dummies, to achieve this and to avoid any problems. The value than I dropped will get absorbed into the constant.


```python
posd = pd.get_dummies(X.Pos, drop_first=True)
X = pd.concat([X, posd],axis=1).drop(['Pos'], axis=1)
```

### Add Polynomial Features


```python
fig, ax = plt.subplots(5,5, figsize=(10,10))

cols = X.columns
col_idx = 0
for i1 in range(5):
    for i2 in range(5):
        ax[i1,i2].scatter(X[cols[col_idx]], y)
        ax[i1,i2].set_title(cols[col_idx])
        col_idx+=1
        
fig.tight_layout()
```


    
![png](media/output_29_0.png)
    


By looking at the scatter plots, it seems like a polynomial will fit better in some cases. I'll add second degree polynomial features to the data.


```python
degree = 2
for col in X:
    X[col+f'_{degree}'] = X[col] ** degree
```

### Drop Highly Correlated Columns

To avoid bad p-values off the start, I'm removing highly correlated variables (correlation coefficient over 0.8).


```python
sns.heatmap(X.corr().abs())
```




    <AxesSubplot:>




    
![png](media/output_34_1.png)
    



```python
# Drop highly correlated values
cor_matrix = X.corr().abs()
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] >= 0.8)]
X = X.drop(to_drop, axis=1)

print('Dropped: \n', to_drop)
```

    Dropped: 
     ['TOI', 'FOL', 'CF', 'CA', 'FF', 'FA', 'FF%', 'FF% rel', 'dZS%', 'TOI/60', 'TOI(EV)', 'GV', 'SAtt.', 'TOI_EVEN', 'CF% Rel_EVEN', 'GF/60_EVEN', 'GA/60_EVEN', 'FTSY_PTS', 'Age_2', 'GP_2', 'PIM_2', 'S_2', 'TOI_2', 'ATOI_2', 'FOW_2', 'FOL_2', 'CF_2', 'CA_2', 'CF%_2', 'FF_2', 'FA_2', 'FF%_2', 'FF% rel_2', 'oiSH%_2', 'oiSV%_2', 'PDO_2', 'oZS%_2', 'dZS%_2', 'TOI/60_2', 'TOI(EV)_2', 'TK_2', 'GV_2', 'SAtt._2', 'Thru%_2', 'Shift_2', 'TOI_EVEN_2', 'CF% Rel_EVEN_2', 'GF/60_EVEN_2', 'GA/60_EVEN_2', 'TOI_PP_2', 'CF% Rel_PP_2', 'GA/60_PP_2', 'TOI_SH_2', 'CF% Rel_SH_2', 'GF/60_SH_2', 'FTSY_PTS_2', 'F_2']


    /tmp/ipykernel_56729/591815908.py:3: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))


### Normalize Data

There are big differences in the data, in terms of scale and ranges. To combat this, I decided to normalize the data


```python
X_norm = preprocessing.normalize(X)
X = pd.DataFrame(X_norm, columns =list(X.columns))
```

## Preprocess Y


```python
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
ax1.hist(y)
ax1.set_title('Not Log Treated')
ax2.hist(np.log(y))
ax2.set_title('Log Treated')
```




    Text(0.5, 1.0, 'Log Treated')




    
![png](media/output_40_1.png)
    


After looking at the distribution of dependent variable, it is clear that the mean is skewed to one side. To help achieve a distribution which is closer to a normal distribution, I decided to use np.log.


```python
y = np.log(y)
```

# OLS Model

## Split Data


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Fit Model

Adding a constant to X will help the fit.


```python
model_ols = sm.OLS(y_train, sm.add_constant(X_train)).fit(cov_type='HC3')
model_ols.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>FTSY_PTS</td>     <th>  R-squared:         </th> <td>   0.847</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.846</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   621.0</td>
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 27 Feb 2023</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>19:48:40</td>     <th>  Log-Likelihood:    </th> <td> -1583.2</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  3904</td>      <th>  AIC:               </th> <td>   3226.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  3874</td>      <th>  BIC:               </th> <td>   3414.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    29</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>         <td>HC3</td>       <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>      <td>    3.8543</td> <td>    0.258</td> <td>   14.933</td> <td> 0.000</td> <td>    3.348</td> <td>    4.360</td>
</tr>
<tr>
  <th>Age</th>        <td>    2.8389</td> <td>    1.705</td> <td>    1.665</td> <td> 0.096</td> <td>   -0.502</td> <td>    6.180</td>
</tr>
<tr>
  <th>GP</th>         <td>   19.1086</td> <td>    0.909</td> <td>   21.020</td> <td> 0.000</td> <td>   17.327</td> <td>   20.890</td>
</tr>
<tr>
  <th>+/-</th>        <td>    9.8630</td> <td>    2.386</td> <td>    4.133</td> <td> 0.000</td> <td>    5.186</td> <td>   14.540</td>
</tr>
<tr>
  <th>PIM</th>        <td>    0.9812</td> <td>    0.244</td> <td>    4.025</td> <td> 0.000</td> <td>    0.503</td> <td>    1.459</td>
</tr>
<tr>
  <th>S</th>          <td>    4.1820</td> <td>    0.372</td> <td>   11.249</td> <td> 0.000</td> <td>    3.453</td> <td>    4.911</td>
</tr>
<tr>
  <th>ATOI</th>       <td>    0.8932</td> <td>    0.351</td> <td>    2.548</td> <td> 0.011</td> <td>    0.206</td> <td>    1.580</td>
</tr>
<tr>
  <th>FOW</th>        <td>    0.1893</td> <td>    0.074</td> <td>    2.550</td> <td> 0.011</td> <td>    0.044</td> <td>    0.335</td>
</tr>
<tr>
  <th>CF%</th>        <td>   -7.3045</td> <td>    2.883</td> <td>   -2.533</td> <td> 0.011</td> <td>  -12.956</td> <td>   -1.653</td>
</tr>
<tr>
  <th>CF% rel</th>    <td>    1.8274</td> <td>    4.587</td> <td>    0.398</td> <td> 0.690</td> <td>   -7.164</td> <td>   10.819</td>
</tr>
<tr>
  <th>oiSH%</th>      <td> -198.8090</td> <td>  112.402</td> <td>   -1.769</td> <td> 0.077</td> <td> -419.114</td> <td>   21.496</td>
</tr>
<tr>
  <th>oiSV%</th>      <td> -242.2822</td> <td>  112.430</td> <td>   -2.155</td> <td> 0.031</td> <td> -462.641</td> <td>  -21.923</td>
</tr>
<tr>
  <th>PDO</th>        <td>  218.7649</td> <td>  112.564</td> <td>    1.943</td> <td> 0.052</td> <td>   -1.856</td> <td>  439.385</td>
</tr>
<tr>
  <th>oZS%</th>       <td>   -1.9812</td> <td>    1.366</td> <td>   -1.450</td> <td> 0.147</td> <td>   -4.658</td> <td>    0.696</td>
</tr>
<tr>
  <th>TK</th>         <td>    0.3948</td> <td>    0.528</td> <td>    0.748</td> <td> 0.454</td> <td>   -0.639</td> <td>    1.429</td>
</tr>
<tr>
  <th>Thru%</th>      <td>    2.3633</td> <td>    2.037</td> <td>    1.160</td> <td> 0.246</td> <td>   -1.629</td> <td>    6.356</td>
</tr>
<tr>
  <th>Shift</th>      <td>   -0.3463</td> <td>    0.891</td> <td>   -0.389</td> <td> 0.698</td> <td>   -2.093</td> <td>    1.401</td>
</tr>
<tr>
  <th>TOI_PP</th>     <td>    0.8897</td> <td>    0.304</td> <td>    2.926</td> <td> 0.003</td> <td>    0.294</td> <td>    1.486</td>
</tr>
<tr>
  <th>CF% Rel_PP</th> <td>   -0.3757</td> <td>    0.469</td> <td>   -0.802</td> <td> 0.423</td> <td>   -1.294</td> <td>    0.543</td>
</tr>
<tr>
  <th>GF/60_PP</th>   <td>   20.5895</td> <td>    4.398</td> <td>    4.682</td> <td> 0.000</td> <td>   11.970</td> <td>   29.209</td>
</tr>
<tr>
  <th>GA/60_PP</th>   <td>    1.5992</td> <td>    1.805</td> <td>    0.886</td> <td> 0.376</td> <td>   -1.939</td> <td>    5.138</td>
</tr>
<tr>
  <th>TOI_SH</th>     <td>   -0.6796</td> <td>    0.439</td> <td>   -1.549</td> <td> 0.121</td> <td>   -1.540</td> <td>    0.180</td>
</tr>
<tr>
  <th>CF% Rel_SH</th> <td>    0.1573</td> <td>    0.565</td> <td>    0.278</td> <td> 0.781</td> <td>   -0.950</td> <td>    1.265</td>
</tr>
<tr>
  <th>GF/60_SH</th>   <td>    0.1260</td> <td>    1.129</td> <td>    0.112</td> <td> 0.911</td> <td>   -2.086</td> <td>    2.338</td>
</tr>
<tr>
  <th>GA/60_SH</th>   <td>    8.1050</td> <td>    4.154</td> <td>    1.951</td> <td> 0.051</td> <td>   -0.037</td> <td>   16.247</td>
</tr>
<tr>
  <th>F</th>          <td>  246.5392</td> <td>   46.484</td> <td>    5.304</td> <td> 0.000</td> <td>  155.432</td> <td>  337.646</td>
</tr>
<tr>
  <th>+/-_2</th>      <td>    0.3347</td> <td>    0.078</td> <td>    4.265</td> <td> 0.000</td> <td>    0.181</td> <td>    0.489</td>
</tr>
<tr>
  <th>CF% rel_2</th>  <td>   -1.7176</td> <td>    0.507</td> <td>   -3.389</td> <td> 0.001</td> <td>   -2.711</td> <td>   -0.724</td>
</tr>
<tr>
  <th>GF/60_PP_2</th> <td>   -0.8096</td> <td>    0.269</td> <td>   -3.006</td> <td> 0.003</td> <td>   -1.337</td> <td>   -0.282</td>
</tr>
<tr>
  <th>GA/60_SH_2</th> <td>   -0.3778</td> <td>    0.247</td> <td>   -1.527</td> <td> 0.127</td> <td>   -0.863</td> <td>    0.107</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>2188.421</td> <th>  Durbin-Watson:     </th> <td>   1.988</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>43256.568</td>
</tr>
<tr>
  <th>Skew:</th>           <td>-2.252</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td>18.673</td>  <th>  Cond. No.          </th> <td>4.82e+04</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors are heteroscedasticity robust (HC3)<br/>[2] The condition number is large, 4.82e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
y_pred = model_ols.predict(sm.add_constant(X_test))

y_test_real = np.exp(y_test)
y_pred_real = np.exp(y_pred)

mae = mean_absolute_error(y_test_real, y_pred_real)
mse = mean_squared_error(y_test_real, y_pred_real)
rmse = mean_squared_error(y_test_real, y_pred_real, squared=False)

print(f"Mean Absolute Error: {round(mae, 2)}")
print(f"Mean Squared Error: {round(mse, 2)}")
print(f"Root Mean Squared Error: {round(rmse, 2)}")
```

    Mean Absolute Error: 16.64
    Mean Squared Error: 598.73
    Root Mean Squared Error: 24.47


# WLS Model

## Calculate Weights

To get the weights, I set fit a model using the residual and the fitted values. After that, the weight is simply the inverse of the cubed fitted values. The values are cubed because this give the best performing model according to the R-Squared, AIC and BIC.


```python
weights = 1 / smf.ols('model_ols.resid.abs() ~ model_ols.fittedvalues', data=X_train).fit().fittedvalues**3
```

## Fit Model


```python
model_wls = sm.WLS(y_train, sm.add_constant(X_train), weights=weights).fit(cov_type='HC3')
model_wls.summary()
```




<table class="simpletable">
<caption>WLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>FTSY_PTS</td>     <th>  R-squared:         </th> <td>   0.906</td>
</tr>
<tr>
  <th>Model:</th>                   <td>WLS</td>       <th>  Adj. R-squared:    </th> <td>   0.906</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   724.9</td>
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 27 Feb 2023</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>19:48:40</td>     <th>  Log-Likelihood:    </th> <td> -198.26</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  3904</td>      <th>  AIC:               </th> <td>   456.5</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  3874</td>      <th>  BIC:               </th> <td>   644.6</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    29</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>         <td>HC3</td>       <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>      <td>    4.4476</td> <td>    0.160</td> <td>   27.729</td> <td> 0.000</td> <td>    4.133</td> <td>    4.762</td>
</tr>
<tr>
  <th>Age</th>        <td>   -1.1316</td> <td>    1.138</td> <td>   -0.994</td> <td> 0.320</td> <td>   -3.362</td> <td>    1.099</td>
</tr>
<tr>
  <th>GP</th>         <td>   12.6257</td> <td>    0.446</td> <td>   28.278</td> <td> 0.000</td> <td>   11.751</td> <td>   13.501</td>
</tr>
<tr>
  <th>+/-</th>        <td>    6.1982</td> <td>    0.479</td> <td>   12.935</td> <td> 0.000</td> <td>    5.259</td> <td>    7.137</td>
</tr>
<tr>
  <th>PIM</th>        <td>    1.2805</td> <td>    0.215</td> <td>    5.949</td> <td> 0.000</td> <td>    0.859</td> <td>    1.702</td>
</tr>
<tr>
  <th>S</th>          <td>    2.5416</td> <td>    0.144</td> <td>   17.653</td> <td> 0.000</td> <td>    2.259</td> <td>    2.824</td>
</tr>
<tr>
  <th>ATOI</th>       <td>    0.9270</td> <td>    0.157</td> <td>    5.890</td> <td> 0.000</td> <td>    0.618</td> <td>    1.235</td>
</tr>
<tr>
  <th>FOW</th>        <td>    0.2192</td> <td>    0.048</td> <td>    4.615</td> <td> 0.000</td> <td>    0.126</td> <td>    0.312</td>
</tr>
<tr>
  <th>CF%</th>        <td>   -5.6097</td> <td>    2.022</td> <td>   -2.775</td> <td> 0.006</td> <td>   -9.573</td> <td>   -1.647</td>
</tr>
<tr>
  <th>CF% rel</th>    <td>    0.6084</td> <td>    2.380</td> <td>    0.256</td> <td> 0.798</td> <td>   -4.057</td> <td>    5.273</td>
</tr>
<tr>
  <th>oiSH%</th>      <td>  -42.4620</td> <td>  102.216</td> <td>   -0.415</td> <td> 0.678</td> <td> -242.802</td> <td>  157.878</td>
</tr>
<tr>
  <th>oiSV%</th>      <td> -108.1977</td> <td>  102.084</td> <td>   -1.060</td> <td> 0.289</td> <td> -308.279</td> <td>   91.883</td>
</tr>
<tr>
  <th>PDO</th>        <td>   81.3671</td> <td>  102.163</td> <td>    0.796</td> <td> 0.426</td> <td> -118.868</td> <td>  281.602</td>
</tr>
<tr>
  <th>oZS%</th>       <td>   -0.7007</td> <td>    0.836</td> <td>   -0.839</td> <td> 0.402</td> <td>   -2.339</td> <td>    0.937</td>
</tr>
<tr>
  <th>TK</th>         <td>    1.9801</td> <td>    0.421</td> <td>    4.701</td> <td> 0.000</td> <td>    1.155</td> <td>    2.806</td>
</tr>
<tr>
  <th>Thru%</th>      <td>   -1.0735</td> <td>    1.119</td> <td>   -0.959</td> <td> 0.337</td> <td>   -3.267</td> <td>    1.120</td>
</tr>
<tr>
  <th>Shift</th>      <td>   -0.0226</td> <td>    0.788</td> <td>   -0.029</td> <td> 0.977</td> <td>   -1.567</td> <td>    1.522</td>
</tr>
<tr>
  <th>TOI_PP</th>     <td>    1.5612</td> <td>    0.179</td> <td>    8.736</td> <td> 0.000</td> <td>    1.211</td> <td>    1.911</td>
</tr>
<tr>
  <th>CF% Rel_PP</th> <td>   -0.8214</td> <td>    0.367</td> <td>   -2.236</td> <td> 0.025</td> <td>   -1.541</td> <td>   -0.101</td>
</tr>
<tr>
  <th>GF/60_PP</th>   <td>   18.6592</td> <td>    3.455</td> <td>    5.401</td> <td> 0.000</td> <td>   11.888</td> <td>   25.430</td>
</tr>
<tr>
  <th>GA/60_PP</th>   <td>    3.3235</td> <td>    2.594</td> <td>    1.281</td> <td> 0.200</td> <td>   -1.761</td> <td>    8.408</td>
</tr>
<tr>
  <th>TOI_SH</th>     <td>   -0.2503</td> <td>    0.173</td> <td>   -1.448</td> <td> 0.148</td> <td>   -0.589</td> <td>    0.088</td>
</tr>
<tr>
  <th>CF% Rel_SH</th> <td>    0.1043</td> <td>    0.289</td> <td>    0.362</td> <td> 0.718</td> <td>   -0.461</td> <td>    0.670</td>
</tr>
<tr>
  <th>GF/60_SH</th>   <td>    0.3806</td> <td>    0.980</td> <td>    0.389</td> <td> 0.698</td> <td>   -1.539</td> <td>    2.301</td>
</tr>
<tr>
  <th>GA/60_SH</th>   <td>    6.4306</td> <td>    2.617</td> <td>    2.457</td> <td> 0.014</td> <td>    1.302</td> <td>   11.560</td>
</tr>
<tr>
  <th>F</th>          <td>  300.0510</td> <td>   23.741</td> <td>   12.639</td> <td> 0.000</td> <td>  253.520</td> <td>  346.582</td>
</tr>
<tr>
  <th>+/-_2</th>      <td>    0.1608</td> <td>    0.052</td> <td>    3.074</td> <td> 0.002</td> <td>    0.058</td> <td>    0.263</td>
</tr>
<tr>
  <th>CF% rel_2</th>  <td>    0.2119</td> <td>    0.235</td> <td>    0.903</td> <td> 0.367</td> <td>   -0.248</td> <td>    0.672</td>
</tr>
<tr>
  <th>GF/60_PP_2</th> <td>   -0.2668</td> <td>    0.213</td> <td>   -1.253</td> <td> 0.210</td> <td>   -0.684</td> <td>    0.150</td>
</tr>
<tr>
  <th>GA/60_SH_2</th> <td>   -0.1556</td> <td>    0.155</td> <td>   -1.007</td> <td> 0.314</td> <td>   -0.459</td> <td>    0.147</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>817.117</td> <th>  Durbin-Watson:     </th> <td>   1.997</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>4460.748</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.888</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.926</td>  <th>  Cond. No.          </th> <td>5.61e+04</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors are heteroscedasticity robust (HC3)<br/>[2] The condition number is large, 5.61e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



This model has very respectable r-squared, aic and bic values, but there are multiple variables which are statistically insignificant. This means that there are potentially random elements added to the model.


```python
y_pred = model_wls.predict(sm.add_constant(X_test))

y_test_real = np.exp(y_test)
y_pred_real = np.exp(y_pred)

mae = mean_absolute_error(y_test_real, y_pred_real)
mse = mean_squared_error(y_test_real, y_pred_real)
rmse = mean_squared_error(y_test_real, y_pred_real, squared=False)

print(f"Mean Absolute Error: {round(mae, 2)}")
print(f"Mean Squared Error: {round(mse, 2)}")
print(f"Root Mean Squared Error: {round(rmse, 2)}")
```

    Mean Absolute Error: 12.59
    Mean Squared Error: 299.05
    Root Mean Squared Error: 17.29


Based on the evaluation metrics, the model's predicted values for a certain target variable have an average absolute error of 12.59 units, an average squared error of 299.05 squared units, and an average error of 17.29 units, measured in the same units as the target variable.

# Optimized WLS *(Insignificant Values Removed)*

## Remove Insignificant Variables

To build the best model, I used the p-values from the previous WLS model to remove statistically insignificant variables. Also, I removed a couple variables manually (Shift, ATOI, TOI_SH, GF/60_PP_2, GA/60_PP) while adjusting the current model (Optimized WLS) based on the p-values.


```python
to_drop = (
    list(model_wls.pvalues[model_wls.pvalues > 0.05].index)
                           +
    ['Shift', 'ATOI', 'TOI_SH', 'GF/60_PP_2', 'GA/60_PP', 'CF% Rel_PP', 'GA/60_SH', '+/-_2', 'F']
)
X_opt = X.drop(to_drop, axis=1)


print('Dropped: \n', to_drop)
```

    Dropped: 
     ['Age', 'CF% rel', 'oiSH%', 'oiSV%', 'PDO', 'oZS%', 'Thru%', 'Shift', 'GA/60_PP', 'TOI_SH', 'CF% Rel_SH', 'GF/60_SH', 'CF% rel_2', 'GF/60_PP_2', 'GA/60_SH_2', 'Shift', 'ATOI', 'TOI_SH', 'GF/60_PP_2', 'GA/60_PP', 'CF% Rel_PP', 'GA/60_SH', '+/-_2', 'F']


## Split New Data

Splitting the new dataset into training and test datasets.


```python
X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size=0.2, random_state=42)
```

## Recalculate Weights

Since the data's shape changed, I recalculated the weights using a new OLS model.


```python
model_ols_opt = sm.OLS(y_train, sm.add_constant(X_train)).fit(cov_type='HC3')
weights = 1 / smf.ols('model_ols_opt.resid.abs() ~ model_ols_opt.fittedvalues', data=X_train).fit().fittedvalues**4
```


```python
model_opt = sm.WLS(y_train, sm.add_constant(X_train), weights=weights).fit(cov_type='HC3')
model_opt.summary()
```




<table class="simpletable">
<caption>WLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>FTSY_PTS</td>     <th>  R-squared:         </th> <td>   0.849</td>
</tr>
<tr>
  <th>Model:</th>                   <td>WLS</td>       <th>  Adj. R-squared:    </th> <td>   0.849</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1310.</td>
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 27 Feb 2023</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>19:48:40</td>     <th>  Log-Likelihood:    </th> <td> -1193.1</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  3904</td>      <th>  AIC:               </th> <td>   2406.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  3894</td>      <th>  BIC:               </th> <td>   2469.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     9</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>         <td>HC3</td>       <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>        <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>    <td>    4.6580</td> <td>    0.036</td> <td>  128.137</td> <td> 0.000</td> <td>    4.587</td> <td>    4.729</td>
</tr>
<tr>
  <th>GP</th>       <td>    8.2144</td> <td>    0.555</td> <td>   14.795</td> <td> 0.000</td> <td>    7.126</td> <td>    9.303</td>
</tr>
<tr>
  <th>+/-</th>      <td>   10.1939</td> <td>    0.606</td> <td>   16.828</td> <td> 0.000</td> <td>    9.007</td> <td>   11.381</td>
</tr>
<tr>
  <th>PIM</th>      <td>    1.5471</td> <td>    0.229</td> <td>    6.765</td> <td> 0.000</td> <td>    1.099</td> <td>    1.995</td>
</tr>
<tr>
  <th>S</th>        <td>    3.5920</td> <td>    0.146</td> <td>   24.535</td> <td> 0.000</td> <td>    3.305</td> <td>    3.879</td>
</tr>
<tr>
  <th>FOW</th>      <td>    0.2642</td> <td>    0.028</td> <td>    9.302</td> <td> 0.000</td> <td>    0.209</td> <td>    0.320</td>
</tr>
<tr>
  <th>CF%</th>      <td>  -32.0138</td> <td>    0.785</td> <td>  -40.780</td> <td> 0.000</td> <td>  -33.552</td> <td>  -30.475</td>
</tr>
<tr>
  <th>TK</th>       <td>    4.3143</td> <td>    0.648</td> <td>    6.657</td> <td> 0.000</td> <td>    3.044</td> <td>    5.584</td>
</tr>
<tr>
  <th>TOI_PP</th>   <td>    3.1684</td> <td>    0.154</td> <td>   20.518</td> <td> 0.000</td> <td>    2.866</td> <td>    3.471</td>
</tr>
<tr>
  <th>GF/60_PP</th> <td>   12.0139</td> <td>    2.025</td> <td>    5.932</td> <td> 0.000</td> <td>    8.044</td> <td>   15.984</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>1330.292</td> <th>  Durbin-Watson:     </th> <td>   1.957</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>12010.034</td>
</tr>
<tr>
  <th>Skew:</th>           <td>-1.361</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td>11.150</td>  <th>  Cond. No.          </th> <td>    462.</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors are heteroscedasticity robust (HC3)



After all the tweaks to the model, the polynomial features were filtered out, so it should be noted that they arent necessary. Considering the degree applied to the weight, that might have an effect on which variables are necessary.With an r-squared of 0.849, we can say that the model is acceptable. Finally, all the p-values are very close to zero. Also, the aic and bic values remain respectable (between those of the ols and the first wls model).


```python
y_pred = model_opt.predict(sm.add_constant(X_test))

y_test_real = np.exp(y_test)
y_pred_real = np.exp(y_pred)

mae = mean_absolute_error(y_test_real, y_pred_real)
mse = mean_squared_error(y_test_real, y_pred_real)
rmse = mean_squared_error(y_test_real, y_pred_real, squared=False)

print(f"Mean Absolute Error: {round(mae, 2)}")
print(f"Mean Squared Error: {round(mse, 2)}")
print(f"Root Mean Squared Error: {round(rmse, 2)}")
```

    Mean Absolute Error: 15.08
    Mean Squared Error: 438.61
    Root Mean Squared Error: 20.94


Based on the evaluation metrics, the model's predicted values for a certain target variable have an average absolute error of 15.08 units, an average squared error of 438.61 squared units, and an average error of 20.94 units, measured in the same units as the target variable.

# Model Choice


```python
p_wls = model_wls.pvalues.to_frame().reset_index().rename({'index': 'Models', 0:'WLS Model'}, axis=1).round(2)
p_opt = model_opt.pvalues.to_frame().reset_index().rename({'index': 'Models', 0:'OPT Model'}, axis=1).round(2)

p_table = p_wls.merge(p_opt, on='Models', how='left').sort_values(by='OPT Model').set_index('Models').T.reset_index()

fig, ax = plt.subplots(figsize=(30,0.3))
ax.table(cellText=p_table.values, colLabels=p_table.columns).set_fontsize(15)
ax.axis('off')
ax.set_title('P-Values')
```




    Text(0.5, 1.0, 'P-Values')




    
![png](media/output_74_1.png)
    


| Metrics  | WLS Model  | Optimized WLS Model  |
|---|---|---|
| MSE  | <span style="color: green;">12.59</span>  | <span style="color: red;">15.08</span>  | 
| RMSE  | <span style="color: green;">17.29</span> | <span style="color: red;">20.94</span>  | 
| R^2  | <span style="color: green;">0.906</span>  | <span style="color: red;">0.849</span>  | 
----------------------------
*Note: The WLS Model has a many values that are statistically insignificant

The high amount of statistically insignificant variables in the first WLS model indicates that those variables have a higher chance of "luck" affecting the model. Thus, even thought the model has a much higher r-squared, AIC and BIC, the optimized model captures the nature of the data and its relationships much better.

# Model Summary

__Explanatory Variable Glossary:__
| Variables | Description |
|-------------|--------------------|
| GP | Total amount of games played. |
| +/- | Difference between the amount scored by a team and by the opposing team while a player is on the ice. |
| PIM | Total amount of penalty minutes. |
| S | Total amount of shots on goal. |
| FOW | Total amount of faceoff wins. |
| CF% | Calculated as the sum of shots on goal, missed shots, and blocked shots over the shots against, missed shots against and blocked shots against at equal strength. |
| TK | Total amount of takeaways. |
| TOI_PP | Average time on ice while on power play. |
| GF/60_PP |Total amount of goals for while a player is on the ice, per 60 minutes of ice time during the power play. |

__Explanatory Variable Coefficients:__
|  Variables   |  Coefs    | Description      |
|--------|----------------|-------------------|
|const    |   4.66| The baseline amount of points is 4.66.  |
|GP        |  8.21| For every game played, the player gains 8.2 points. |
|+/-       |  10.19| For every +/-, the player is expected to score 10 more points. |
|PIM      |   1.55| For every PIM, the player is expected to score about 1.5 more points. |
|S        |   3.59| For every shot, the player is expected to score about 3.6 more points. |
|FOW      |   0.26| For every faceoff win, the player is expected to score about 0.3 more points. |
|CF%       |  -32.01|For every increase of CF%, the player is expected to score about 32 fewer points. |
|TK       |   4.31| For every takeaway, the player is expected to score about 4.3 more points.|
|TOI_PP    |  3.17| For every increase of time on ice during the power play, the player is expected to score about 3.2 more points.|
|GF/60_PP  |  12.01|For every increase of goals for per 60 minutes, the player is expected to score about 12 more points. |
-----
*The coefficents assume that every other variable is at 0.*
