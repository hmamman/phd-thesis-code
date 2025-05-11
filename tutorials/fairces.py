import math
import os
import sys

import time

import joblib
import numpy as np

# Get the absolute path to the directory where fairces.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Two levels up from fairces.py
sys.path.append(os.path.join(base_path, "../"))

from tutorials.fairses import FairSES
from tutorials.algorithms.ces import CES
from utils.helpers import get_experiment_params


class FairCES (FairSES):
    def __init__(self, config, model, sensitive_param, population_size=200, threshold=0):
        super().__init__(config, model, sensitive_param, population_size, threshold)

        self.setup = CES(
            mu=self.population_size,
            lambda_=2,
            sigma=0.1,
            bounds=np.array(self.config.input_bounds),
            fitness_func=self.evaluate_disc
        )
        self.approach_name = f"FairCES"


if __name__ == '__main__':
    config, sensitive_name, sensitive_param, classifier_name, max_allowed_time = get_experiment_params()

    print(f'Dataset: {config.dataset_name}')
    print(f'Classifier: {classifier_name}')
    print(f'Sensitive name: {sensitive_name}')
    print('')

    if classifier_name == 'dnn':
        import tensorflow as tf

        classifier_path = f'models/{config.dataset_name}/dnn_slfc.keras'
        model = tf.keras.models.load_model(classifier_path)
    else:

        classifier_path = f'models/{config.dataset_name}/{classifier_name}.pkl'
        model = joblib.load(classifier_path)

    fairces = FairCES(
        config=config,
        model=model,
        sensitive_param=sensitive_param
    )

    fairces.run(max_allowed_time=max_allowed_time)
