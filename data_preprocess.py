import os
import sys
import math
import pandas as pd
import numpy as np

def get_dataset():

    # Import Dataset
    pre = os.path.dirname(os.path.abspath('__file__'))
    post = 'testPuneAQMNew_22.csv'
    full_path = os.path.join(pre,post)
    df = pd.read_csv(full_path)

    # One Hot Encode airQualityLevel_encod
    airQualityLevel_dict = {'SATISFACTORY': 1, 'MODERATE': 2, 'POOR': 3, 'VERY_POOR': 4, 'SEVERE': 5}
    df['airQualityLevel_encod'] = df.airQualityLevel.map(airQualityLevel_dict)
    df = df.drop(['airQualityLevel'], axis=1)

    # One Hot Encode airMajorPollutant_encod
    aqiMajorPollutant_dict = {'PM2.5': 1, 'CO': 2, 'NO2': 3}
    df['aqiMajorPollutant_encod'] = df.aqiMajorPollutant.map(aqiMajorPollutant_dict)
    df = df.drop(['aqiMajorPollutant'], axis=1)
    df = df.dropna(subset=['airTemperature.avgOverTime'])

    # Final Features
    features_considered = ['observationDateTime','airQualityLevel_encod', 'aqiMajorPollutant_encod','airQualityIndex','uv.avgOverTime',
                       'o3.avgOverTime','pm2p5.avgOverTime','co2.avgOverTime','pm10.avgOverTime','co.avgOverTime','no2.avgOverTime',
                       'airTemperature.avgOverTime','illuminance.avgOverTime','ambientNoise.avgOverTime','so2.avgOverTime',
                       'relativeHumidity.avgOverTime','atmosphericPressure.avgOverTime']
    features = df[features_considered]
    features.index = df['observationDateTime']
    dataset = features.values

    return dataset, features.index



def time_feature_to_time(time_feature):
    time = np.zeros_like(time_feature)
    for i, batch in enumerate(time_feature):
        time[i] = str(batch.split(' ')[1][3:5])
    return time


def scale_data(dataset):
    '''
    scale the dataset to smaller values
    : param dataset:                 provide the dataset
    : return                         scaled data
    '''
    mean = dataset.mean(0)
    std = dataset.std(0)

    return (dataset - mean)/std
