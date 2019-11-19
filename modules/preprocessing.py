#!/usr/bin/env python

import numpy as np
import pandas as pd
import socket as sk
import struct as st
import datetime as dt
import ipaddress as ip
from time import perf_counter

# # Configurations

# finding the bad guys, dah!

# slowloris 
slowloris_low  = st.unpack('!I', sk.inet_aton('10.128.0.1'))[0]
slowloris_high = st.unpack('!I', sk.inet_aton('10.128.0.50'))[0]

# slowhttptest
slowhttptest_low  = st.unpack('!I', sk.inet_aton('10.128.0.50'))[0]
slowhttptest_high = st.unpack('!I', sk.inet_aton('10.128.0.100'))[0]

# slowloris_ng
slowloris_ng_low  = st.unpack('!I', sk.inet_aton('10.128.0.100'))[0]
slowloris_ng_high = st.unpack('!I', sk.inet_aton('10.128.0.150'))[0]

# defining the TCP flags
tcp_flags = [2, 4, 16, 17, 18, 20, 24, 25, 82, 144, 152, 194]


def prequelProcessing(dataset):
    
    # Casting IP to a single integer
    dataset['source_ip'] = dataset.source_ip.apply(lambda x: st.unpack('!I', sk.inet_aton(x))[0])
    dataset['dest_ip'] = dataset.dest_ip.apply(lambda x: st.unpack('!I', sk.inet_aton(x))[0])
    
    # Casting Hexa to decimal base
    dataset['tcp_flag'] = dataset.tcp_flag.apply(lambda x: int(x, 16))
    
    # Parsing string to datetime object
    dataset['date'] = dataset['date'] + ' ' + dataset['time']
    dataset['date'] = pd.to_datetime(dataset['date'], format='%Y%m%d %H:%M:%S', utc=True)
    
    # Getting rid of useless columns
    dataset.drop(columns=['data', 'time'], inplace=True)
    
    return dataset

def features(grouped_data):
    
    number_requisitions = np.sum(grouped_data['dest_port'] == 80) + np.sum(grouped_data['dest_port'] == 443)
    number_different_destinations = len(np.unique(grouped_data['dest_ip']))
    mean_frame_length = grouped_data['frame_length'].mean()
    
    data = {
            'number_requisitions'           : [number_requisitions], 
            'number_different_destinations' : [number_different_destinations], 
            'mean_frame_length'             : [mean_frame_length]
           }

    for flag in tcp_flags:
        data['flag_' + str(flag)] = [np.sum(grouped_data['tcp_flag'] == flag)]
    
    return pd.DataFrame(data)


def turnToPercentil(dataset, summary, column_name):

    for i in range(len(summary.index)):
        
        if summary[column_name][i] > 0:

            data_percentil = dataset.loc[summary.index[i], column_name] / summary[column_name][i]
            dataset.loc[summary.index[i], column_name] = data_percentil.values


def normalizationPerTimePeriod(dataset):
    
    summary = dataset.groupby('date').sum()
    
    column_names= dataset.columns.values
    column_names = np.delete(column_names, 2)
    
    for column in column_names:
        
        turnToPercentil(dataset, summary, column)


def generateLabelColumn(grouped):
    
    # setting all IPs with none intruser type
    grouped['y'] = 0
    
    # resetting the index 
    dataset = grouped.reset_index()
    
    # finding the bad guys
    slowloris    = (dataset.source_ip >= slowloris_low) & (dataset.source_ip < slowloris_high)
    slowhttptest = (dataset.source_ip >= slowhttptest_low) & (dataset.source_ip < slowhttptest_high)
    slowloris_ng = (dataset.source_ip >= slowloris_ng_low) & (dataset.source_ip < slowloris_ng_high)

    # and labeling them
    dataset.loc[slowloris, 'y']    = 1
    dataset.loc[slowhttptest, 'y'] = 2
    dataset.loc[slowloris_ng, 'y'] = 3
    
    # resuming the original index
    dataset.set_index(['date', 'source_ip'], inplace=True)
    
    # getting rid of useless columns
    dataset.drop(columns=['level_2'], inplace=True)
    
    return dataset

def preprocessing(packages, frequency):

    print('Applying prequel processing')
    
    dataset = prequelProcessing(packages)
    
    print('Generating the features')

    grouped = dataset.groupby([
            # groupping the data per a specific time frequency
            pd.Grouper(key='date', freq=frequency), 
            # groupping the remaining data by the IPs
            pd.Grouper(key='source_ip')
            # Applying the function who will create the news features
            ]).apply(features)

    print('Normalizing the features')
    
    # normalizing the data
    normalizationPerTimePeriod(grouped)

    print('Creating the true label array')
    
    # generating the true label array
    dataset = generateLabelColumn(grouped)
    
    return dataset
