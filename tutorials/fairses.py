import math
import os
import sys

import time

import joblib
import numpy as np


# Get the absolute path to the directory where faires.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Two levels up from faires.py
sys.path.append(os.path.join(base_path, "../"))

from tutorials.algorithms.ses import SES
from utils.helpers import get_experiment_params, generate_report
from utils.dnn_wrapper import dnn_model_wrapper


class FairSES:
    def __init__(self, config, model, sensitive_param, population_size=200, threshold=0):
        self.binary_threshold = threshold
        self.config = config
        self.threshold = threshold
        self.population_size = population_size

        self.tot_inputs = set()
        self.disc_inputs = set()
        self.disc_inputs_list = []

        self.input_bounds = np.array(self.config.input_bounds)
        self.sensitive_param = sensitive_param
        self.model = dnn_model_wrapper(model=model)

        self.start_time = time.time()
        self.time_to_1000_disc = -1
        self.total_generated = 0
        self.cumulative_efficiency = []
        self.inference_count = 0

        self.setup = SES(
            mu=self.population_size,
            lambda_=2,
            sigma=0.1,
            bounds=np.array(self.config.input_bounds),
            fitness_func=self.evaluate_disc
        )
        self.approach_name = f"FairSES"


    def evaluate_disc(self, inp):
        inp0 = np.array([int(x) for x in inp])
        inp1 = np.array([int(x) for x in inp])

        inp0 = inp0.reshape(1, -1)
        inp1 = inp1.reshape(1, -1)

        self.tot_inputs.add(tuple(map(tuple, inp0)))
        self.total_generated += 1

        inp0 = np.asarray(inp0).reshape((1, -1))
        out0 = self.model.predict(inp0)
        self.inference_count += 1

        for val in range(self.config.input_bounds[self.sensitive_param - 1][0],
                         self.config.input_bounds[self.sensitive_param - 1][1] + 1):
            if val != inp0[0][self.sensitive_param - 1]:
                inp1[0][self.sensitive_param - 1] = val
                out1 = self.model.predict(inp1)
                self.inference_count += 1

                if abs(out0 - out1) > self.threshold and (tuple(map(tuple, inp0)) not in self.disc_inputs):
                    self.disc_inputs.add(tuple(map(tuple, list(inp0))))
                    self.disc_inputs_list.append(inp0.tolist()[0])

                    self.set_time_to_1000_disc()

                    return 1
        return 0

    def evaluate_disc2(self, inp):
        # Convert input to numpy array
        inp0 = np.array([int(x) for x in inp]).reshape(1, -1)

        # Track this input in our set of total inputs
        self.tot_inputs.add(tuple(map(tuple, inp0)))
        self.total_generated += 1

        # Get the possible values for the sensitive parameter
        sensitive_param_values = range(
            self.config.input_bounds[self.sensitive_param - 1][0],
            self.config.input_bounds[self.sensitive_param - 1][1] + 1
        )

        # Create an array of variations - one for each possible value of the sensitive parameter
        variations = []
        for val in sensitive_param_values:
            if val != inp0[0][self.sensitive_param - 1]:
                inp_variant = inp0.copy()
                inp_variant[0][self.sensitive_param - 1] = val
                variations.append(inp_variant)

        # If no variations were created, return 0
        if not variations:
            return 0

        # Combine original input with all variations into a single batch
        all_inputs = np.vstack([inp0] + variations)

        # Make a single inference call for all inputs
        all_outputs = self.model.predict(all_inputs)
        self.inference_count += 1  # Only one model call regardless of number of variations

        # Extract the output for the original input
        original_output = all_outputs[0]

        # Vectorized comparison with all variant outputs
        differences = np.abs(all_outputs[1:] - original_output) > self.threshold
        diff_count = np.sum(differences)

        # If we found discriminatory inputs, add them to our tracking structures
        if diff_count > 0:
            self.disc_inputs.add(tuple(map(tuple, inp0)))
            self.disc_inputs_list.append(inp0.tolist()[0])
            self.set_time_to_1000_disc()

        return int(diff_count)

    def update_cumulative_efficiency(self, iteration):
        """
        Update the cumulative efficiency data if the current number of total inputs
        meets the tracking criteria (first input or every tracking_interval inputs).
        """
        total_inputs = len(self.tot_inputs)
        total_disc = len(self.disc_inputs)
        self.cumulative_efficiency.append([time.time() - self.start_time, iteration, total_inputs, total_disc])

    def set_time_to_1000_disc(self):
        disc_inputs_count = len(self.disc_inputs)

        if disc_inputs_count >= 1000 and self.time_to_1000_disc == -1:
            self.time_to_1000_disc = time.time() - self.start_time
            print(f"\nTime to generate 1000 discriminatory inputs: {self.time_to_1000_disc:.2f} seconds")

    def run(self, max_samples=1000 * 1000, max_allowed_time=3600):
        self.start_time = time.time()

        count = 300

        max_evolution = math.ceil(max_samples / self.population_size)

        for i in range(max_evolution):
            self.setup.run()
            self.update_cumulative_efficiency(1 + i)

            use_time = time.time() - self.start_time
            if use_time >= count:
                count += 300
                self.report(elapsed_time=use_time, is_log=True)

            if count >= max_allowed_time or self.total_generated >= max_samples:
                break

        elapsed_time = time.time() - self.start_time
        # self.setup.plot_evolution()
        #
        self.report(elapsed_time=elapsed_time, is_log=False)

    def report(self, elapsed_time, is_log: bool):
        additional_data = {'inference_count': self.inference_count}
        generate_report(
            approach_name=f'{self.approach_name}',
            dataset_name=self.config.dataset_name,
            classifier_name=self.model.__class__.__name__,
            sensitive_name=self.config.sens_name[self.sensitive_param],
            tot_inputs=self.tot_inputs,
            disc_inputs=self.disc_inputs,
            total_generated_inputs=self.total_generated,
            elapsed_time=elapsed_time,
            time_to_1000_disc=self.time_to_1000_disc,
            cumulative_efficiency=self.cumulative_efficiency,
            is_log=is_log,
            **additional_data,
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

    fairses = FairSES(
        config=config,
        model=model,
        sensitive_param=sensitive_param
    )

    fairses.run(max_allowed_time=max_allowed_time)
