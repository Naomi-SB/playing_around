import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


################################################################## THESE ARE THE PHYSICAL DATA FEATURES
features = ['climateregions__climateregion', 'elevation__elevation', 'lat', 'lon', 'startdate',
                 'contest-pevpr-sfc-gauss-14d__pevpr','contest-precip-14d__precip','contest-pres-sfc-gauss-14d__pres',
                 'contest-prwtr-eatm-14d__prwtr','contest-rhum-sig995-14d__rhum','contest-slp-14d__slp',
                 'contest-tmp2m-14d__tmp2m','contest-wind-h10-14d__wind-hgt-10','contest-wind-h100-14d__wind-hgt-100',
                 'contest-wind-h500-14d__wind-hgt-500','contest-wind-h850-14d__wind-hgt-850','contest-wind-uwnd-250-14d__wind-uwnd-250','contest-wind-uwnd-925-14d__wind-uwnd-925','contest-wind-vwnd-250-14d__wind-vwnd-250',
                 'contest-wind-vwnd-925-14d__wind-vwnd-925']



################################################################## DATA ACQUISITION
def get_explore_data():
    ''' 
    This function reads in a csv held in the same repository folder
    '''
    df = pd.read_csv('train_data.csv').drop('index', axis='columns')
    return df



################################################################## DATA PREPARATION
def prep_data(df, features=[]):
    '''
    This function pulls in the defined features as the only features 
    to be represented as columns in the resulting dataframe
    '''
    df['startdate'] = pd.to_datetime(df['startdate'])
    if len(features) == 0:
        return df
    else:
        return df[features]

def create_region_bins(df):
    '''
    This function creates a new column that holds
    three categorical variables dry, temperate, and continental
    that represents the bins we put the 15 original regions into
    based on the first letter of their Koppen-Geiger code
    '''
    df['region_bins'] = df.region.replace({'BWh' :'Dry', 'BWk' :'Dry', 'BSh':'Dry', 'BSk':'Dry',
                                        'Csa':'Temperate', 'Csb':'Temperate', 'Cfa':'Temperate', 'Cfb':'Temperate',
                                        'Dsb':'Continental', 'Dsc':'Continental', 'Dwa':'Continental', 'Dwb':'Continental', 'Dfa':'Continental', 'Dfb':'Continental', 'Dfc':'Continental'})

    return df


def create_elevation_bins(df):
    '''
    Function creates four bins of elevation based
    on mathematical quantiles.
    '''
    names = ['bottom_low', 'top_low', 'mid', 'high']
    df['elevation_range'] = pd.qcut(df['elevation'], 4, labels=names)
    
    return df

    
def create_season_bins(df):
    '''
    Creates a column for month and then bases that to create bins for seasons
    '''
    #create month column
    df['month']=df['startdate'].dt.month
    
    #define season groups
    season_groups = {
        "Autumn": [9,10,11],
        "Winter": [12,1,2],
        "Spring": [3,4,5],
        "Summer": [6,7,8],
    }
    #add season onto the df
    df["season"] = (
        df["month"]
        .apply(lambda x: [k for k in season_groups.keys() if x in season_groups[k]])
        .str[0]
        .fillna("Other")
    )
    return df

def rename_data(df):
    '''
    This function takes in the dataframe and returns all columns
    with only the listed columns names changed to be more readable.
    '''
    df=df.rename(columns={'climateregions__climateregion': 'region', 
                                      'elevation__elevation': 'elevation',
                                      'contest-pevpr-sfc-gauss-14d__pevpr':'potential_evap',
                                      'contest-precip-14d__precip':'precip',
                                      'contest-pres-sfc-gauss-14d__pres':'barometric_pressure',
                                      'contest-prwtr-eatm-14d__prwtr':'all_atmos_precip',
                                      'contest-rhum-sig995-14d__rhum':'relative_humidity',
                                      'contest-slp-14d__slp':'sea_level_press',
                                      'contest-tmp2m-14d__tmp2m':'mean_temp',
                                      'contest-wind-h10-14d__wind-hgt-10':'height_10_mb',
                                      'contest-wind-h100-14d__wind-hgt-100':'height_100_mb',
                                      'contest-wind-h500-14d__wind-hgt-500':'height_500_mb',
                                      'contest-wind-h850-14d__wind-hgt-850':'height_850_mb',
                                      'contest-wind-uwnd-250-14d__wind-uwnd-250':'zonal_wind_250mb',
                                      'contest-wind-uwnd-925-14d__wind-uwnd-925':'zonal_wind_925mb',
                                      'contest-wind-vwnd-250-14d__wind-vwnd-250':'long_wind_250mb',
                                      'contest-wind-vwnd-925-14d__wind-vwnd-925':'long_wind_925mb'
                                     })
    return df

def deal_with_nulls(df):
    '''
    Calculates nulls based on other features in the df
    and drops one that can't be computed
    '''
    #calculate the columns with nulls
    count_na_df = df.isna().sum()
    null_cols = list(count_na_df[count_na_df > 0].index)

    #means of each nmme measurement
    g_means =  ['nmme0-tmp2m-34w__nmme0mean', 
    'nmme-tmp2m-56w__nmmemean', 
    'nmme-prate-34w__nmmemean', 
    'nmme0-prate-56w__nmme0mean', 
    'nmme0-prate-34w__nmme0mean', 
    'nmme-prate-56w__nmmemean', 
    'nmme-tmp2m-34w__nmmemean']

    #measurements we have already
    g_1 = ['nmme0-tmp2m-34w__cancm30',
    'nmme0-tmp2m-34w__cancm40',
    'nmme0-tmp2m-34w__ccsm40',
    'nmme0-tmp2m-34w__cfsv20',
    'nmme0-tmp2m-34w__gfdlflora0',
    'nmme0-tmp2m-34w__gfdlflorb0',
    'nmme0-tmp2m-34w__gfdl0',
    'nmme0-tmp2m-34w__nasa0']

    g_2 = ['nmme-tmp2m-56w__cancm3',
    'nmme-tmp2m-56w__cancm4',
    'nmme-tmp2m-56w__ccsm4',
    'nmme-tmp2m-56w__cfsv2',
    'nmme-tmp2m-56w__gfdl',
    'nmme-tmp2m-56w__gfdlflora',
    'nmme-tmp2m-56w__gfdlflorb',
    'nmme-tmp2m-56w__nasa']

    g_3 = ['nmme-prate-34w__cancm3',
    'nmme-prate-34w__cancm4',
    'nmme-prate-34w__ccsm4',
    'nmme-prate-34w__cfsv2',
    'nmme-prate-34w__gfdl',
    'nmme-prate-34w__gfdlflora',
    'nmme-prate-34w__gfdlflorb',
    'nmme-prate-34w__nasa']

    g_4 = [ 'nmme0-prate-56w__cancm30',
    'nmme0-prate-56w__cancm40',
    'nmme0-prate-56w__ccsm40',
    'nmme0-prate-56w__cfsv20',
    'nmme0-prate-56w__gfdlflora0',
    'nmme0-prate-56w__gfdlflorb0',
    'nmme0-prate-56w__gfdl0',
    'nmme0-prate-56w__nasa0']

    g_5 = ['nmme0-prate-34w__cancm30',
    'nmme0-prate-34w__cancm40',
    'nmme0-prate-34w__ccsm40',
    'nmme0-prate-34w__cfsv20',
    'nmme0-prate-34w__gfdlflora0',
    'nmme0-prate-34w__gfdlflorb0',
    'nmme0-prate-34w__gfdl0',
    'nmme0-prate-34w__nasa0']

    g_6 = ['nmme-prate-56w__cancm3',
    'nmme-prate-56w__cancm4',
    'nmme-prate-56w__ccsm4',
    'nmme-prate-56w__cfsv2',
    'nmme-prate-56w__gfdl',
    'nmme-prate-56w__gfdlflora',
    'nmme-prate-56w__gfdlflorb',
    'nmme-prate-56w__nasa']

    g_7 = ['nmme-tmp2m-34w__cancm3',
    'nmme-tmp2m-34w__cancm4',
    'nmme-tmp2m-34w__ccsm4',
    'nmme-tmp2m-34w__cfsv2',
    'nmme-tmp2m-34w__gfdl',
    'nmme-tmp2m-34w__gfdlflora',
    'nmme-tmp2m-34w__gfdlflorb',
    'nmme-tmp2m-34w__nasa']

    #make a list of lists
    gs = [g_1, g_2, g_3, g_4, g_5, g_6, g_7]

    #go through and compute all features where there were nulls
    zip_cols = zip(null_cols, gs, g_means)
    for c, g, s in zip_cols:
        df[c] = (df[s]*9) - df[g].sum(1)

    #drop column we can't compute
    df = df.drop(columns='ccsm30')

    return df


################################# SUM OF PREPARATION 
def get_contest_data(df):
    '''
    This function takes in a dataframe and runs it 
    through the processes of superior preparation functions.
    '''
    df = prep_data(df, features=features)
    df = rename_data(df)
    df = create_elevation_bins(df)
    df = create_region_bins(df)
    df = create_season_bins(df)

    #df = df.drop(columns=['elevation','region']
    return df
    
################################################################## SPLITTING DATA
def split_data(df, test_size=0.15):
    '''
    Takes in a data frame and the train size
    It returns train, validate , and test data frames
    with validate being 0.05 bigger than test and train has the rest of the data.
    '''
    train, test = train_test_split(df, test_size = test_size , random_state=27)
    train, validate = train_test_split(train, test_size = (test_size + 0.05)/(1-test_size), random_state=27)
    
    return train, validate, test