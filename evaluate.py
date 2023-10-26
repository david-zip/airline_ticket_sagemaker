import os
import logging
import pickle
import tarfile
import pathlib
import pandas as pd
import numpy as np
import logging
import json

from sklearn.metrics import mean_squared_error

logger = logging.getLogger()

if __name__=='__main__':
    logger.debug('Starting evaluation...')
    model_path = '/opt/ml/processing/model/model.tar.gz'
    with tarfile.open(model_path) as tar:
        tar.extractall(path='..')

    logger.debug('Loading xgboost model')
    model = pickle.load(open('xgboost-model', 'rb'))

    logger.debug('Reading test data')
    test_path = '/opt/ml/processing/test/test.csv'
    df = pd.read_csv(test_path, header=None)
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    X_test = df.values

    logger.info('Performing predictions against test data.')
    predictions = model.predict(X_test)

    logger.debug('Calculating mean squared error.')
    mse = mean_squared_error(y_test, predictions)
    std = np.std(y_test - predictions)
    report_dict = {
        'regression_metrics': {
            'mse': {
                'value': mse,
                'standard_deviation': std
            },
        },
    }

    output_dir = '/opt/ml/processing/evaluation'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info('Writing out evaluation report with mse: %f', mse)
    evaluation_path = f'{output_dir}/evaluation.json'
    with open(evaluation_path, 'w') as f:
        f.write(json.dumps(report_dict))
