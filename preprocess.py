# import relevant modules
import os
import boto3
import pathlib
import argparse
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger()

if __name__=='__main__':
    # start preprocessing
    logger.debug('Starting preprocessing...')
    
    # get arguments supplied to function
    logger.debug('Initialisng data pathway')
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()
    
    # downloading data
    logger.debug('Downloading data')
    BASE_DIR = "/opt/ml/processing"
    pathlib.Path(f"{BASE_DIR}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data # get input data from input data parameter
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    # reading data
    logger.debug('Reading raw data')
    fn = f"{BASE_DIR}/data/airline_tickets.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)
    raw_data = pd.read_csv(fn)

    # perform transformations
    logging.debug('Starting transfomations')
    raw_data = raw_data.sample(1)
    
    # seperate strings into muliple columns
    raw_data[['Airline', 'Flight code', 'Class']] = raw_data['Airline-Class'].str.split(pat=r'\n', n=-1, expand=True)
    raw_data[['Departure time', 'Departure city']] = raw_data['Departure Time'].str.split(pat=r'\n', n=-1, expand=True)
    raw_data[['Arrival time', 'Arrival city']] = raw_data['Arrival Time'].str.split(pat=r'\n', n=-1, expand=True)

    raw_data = raw_data.drop(['Arrival Time', 'Departure Time', 'Airline-Class'], axis=1)

    # convert time to string without ':'
    raw_data['Arrival time'] = raw_data['Arrival time'].str.split(pat=':')
    raw_data['Arrival time'] = raw_data['Arrival time'].str.join('')

    raw_data['Departure time'] = raw_data['Departure time'].str.split(pat=':')
    raw_data['Departure time'] = raw_data['Departure time'].str.join('')

    arrival_time = pd.cut(
        pd.to_numeric(raw_data['Arrival time']),  
        pd.to_numeric(['0600','1200','1800','2400']),
        labels=['Morning','Afternoon','Evening']
    ).astype(str).replace('nan','Midnight')

    departure_time = pd.cut(
        pd.to_numeric(raw_data['Departure time']),  
        pd.to_numeric(['0600','1200','1800','2400']),
        labels=['Morning','Afternoon','Evening']
    ).astype(str).replace('nan','Midnight')

    raw_data['Arrival time'] = arrival_time
    raw_data['Departure time'] = departure_time

    # apply ordinal encoding
    # basic ordinal encoding to class variable
    unique_class = pd.unique(raw_data['Class']).tolist()

    class_encoding = {}
    for i in range(len(unique_class)):
        class_encoding[unique_class[i]] = i

    raw_data['Class'] = raw_data['Class'].map(class_encoding)

    # basic encoding to departure time and arrival time
    time_encoding = {
        'Morning': 1,
        'Afternoon': 2,
        'Evening': 3,
        'Midnight': 4
    }

    raw_data['Departure time'] = raw_data['Departure time'].map(time_encoding)
    raw_data['Arrival time'] = raw_data['Arrival time'].map(time_encoding)

    # apply ordinal encoding to airline
    unique_airline = pd.unique(raw_data['Airline']).tolist()

    airline_encoding = {}
    for i in range(len(unique_airline)):
        airline_encoding[unique_airline[i]] = i + 1

    raw_data['Airline'] = raw_data['Airline'].map(airline_encoding)

    # apply ordinal encoding to departure city and arrival city
    unique_dcity = pd.unique(raw_data['Departure city']).tolist()
    unique_acity = pd.unique(raw_data['Arrival city']).tolist()

    dcity_encoding = {}
    for i in range(len(unique_dcity)):
        dcity_encoding[unique_dcity[i]] = i + 1

    acity_encoding = {}
    for i in range(len(unique_acity)):
        acity_encoding[unique_acity[i]] = i + 1

    raw_data['Departure city'] = raw_data['Departure city'].map(dcity_encoding)
    raw_data['Arrival city'] = raw_data['Arrival city'].map(acity_encoding)

    # convert total stops into integer (ignores via which airport)
    raw_data['Total Stops'] = raw_data['Total Stops'].str[0].replace('n', 0)

    # find difference between dates and convert dates
    booking_date = pd.to_datetime(raw_data['Date of Booking'], dayfirst=True)
    journey_date = pd.to_datetime(raw_data['Date of Journey'], dayfirst=True)

    raw_data['Date Difference'] = (journey_date - booking_date) / np.timedelta64(1, 'D')

    # find journey day and journey month
    raw_data['Journey day'] = journey_date.dt.day_of_week
    raw_data['Journey month'] = journey_date.dt.month

    # convert time into continuous variable
    duration_cte = raw_data['Duration'].str.split('h ', n=-1, expand=True)

    duration_cte[1] = pd.to_numeric(duration_cte[1].str.replace('m', ''))
    duration_cte[0] = pd.to_numeric(duration_cte[0])

    raw_data['Duration'] = duration_cte[0] + duration_cte[1]/60

    # drop unnecessary columns
    raw_data = raw_data.drop(['Date of Booking', 'Date of Journey', 'Flight code'], axis=1)

    # move price to the front
    price = pd.to_numeric(raw_data['Price'].str.replace(',', ''))
    processed_data = raw_data.drop(['Price'], axis=1)
    processed_data.insert(0, 'Price', price)
    
    # complete transformation steps
    logger.debug('Completed transformations')
    
    # split data for training, validation, and testing
    logger.debug('Splitting dataset')
    train, test, validation = np.split(processed_data, [int(0.7 * len(processed_data)), int(0.85 * len(processed_data))])
    
    # Saving dataset
    logger.debug('Saving dataset')
    pd.DataFrame(train).to_csv(f"{BASE_DIR}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(f"{BASE_DIR}/validation/validation.csv", header=False, index=False)
    pd.DataFrame(test).to_csv(f"{BASE_DIR}/test/test.csv", header=False, index=False)
