#from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
#import streamlit as st
import geopandas as gpd
from matplotlib import pyplot as plt

#api_path = r'path to json Kaggle ApiKey'
#api = KaggleApi()
#api.authenticate()
#files = api.competition_download_files("Russia Real Estate 2018-2021") 

MOSCOW_REGION = 3
N_SAMPLE = 7

data = pd.read_csv('all_v2.csv')
data = data.rename(columns={'geo_lat':'lat', 'geo_lon':'lon'})
tg = data.loc[(55.5735 <= data['lat']) & (data['lat'] <= 55.9104) & (36.3789 <= data['lon']) & (data['lon'] <= 37.8597)].reset_index()

tg.to_csv('moscov.scv')
