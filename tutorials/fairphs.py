import os
import sys

import joblib
import numpy as np


# Get the absolute path to the directory where fairphs.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Two levels up from fairphs.py
sys.path.append(os.path.join(base_path, "../"))

from tutorials.algorithms.phs import PHS
from utils.helpers import get_experiment_params
from tutorials.fairses import FairSES


class FairPHS(FairSES):
    def __init__(self, config, model, sensitive_param, population_size=200, threshold=0):
        super().__init__(config, model, sensitive_param, population_size, threshold)
        self.approach_name = "FairPHS"

        self.setup = PHS(
            mu=self.population_size,
            bounds=np.array(self.config.input_bounds),
            fitness_func=self.evaluate_disc,
        )


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

    fairphs = FairPHS(
        config=config,
        model=model,
        sensitive_param=sensitive_param
    )

    fairphs.run(max_allowed_time=max_allowed_time)
