import math
import os
import sys

import time

import joblib
import numpy as np

# Get the absolute path to the directory where fairpes.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Two levels up from fairpes.py
sys.path.append(os.path.join(base_path, "../"))

from tutorials.fairses import FairSES
from tutorials.algorithms.pes import PES
from utils.helpers import get_experiment_params


class FairPES (FairSES):
    def __init__(self, config, model, sensitive_param, population_size=200, threshold=0):
        super().__init__(config, model, sensitive_param, population_size, threshold)

        self.setup = PES(
            mu=self.population_size,
            lambda_=2,
            sigma=0.1,
            bounds=np.array(self.config.input_bounds),
            fitness_func=self.evaluate_disc
        )
        self.approach_name = f"FairPES"


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

    fairpes = FairPES(
        config=config,
        model=model,
        sensitive_param=sensitive_param
    )

    fairpes.run(max_allowed_time=max_allowed_time)
