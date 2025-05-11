import os
import sys

import numpy as np
import joblib
import time
import random
from scipy.optimize import basinhopping

# Get the absolute path to the directory where aequitas.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Two levels up from aequitas.py
sys.path.append(os.path.join(base_path, "../../"))

from utils.dnn_wrapper import dnn_model_wrapper
from utils.helpers import generate_report, get_experiment_params

class Aequitas:
    def __init__(self, config, model, sensitive_param, max_time_allowed=3600, threshold=0):
        self.config = config
        self.init_prob = 0.5
        self.params = config.params
        self.direction_probability = [self.init_prob] * self.params
        self.direction_probability_change_size = 0.001
        self.param_probability = [1.0 / self.params] * self.params
        self.param_probability_change_size = 0.001
        self.local_disc_inputs = set()
        self.local_disc_inputs_list = []
        self.global_disc_inputs = set()
        self.global_disc_inputs_list = []
        self.tot_inputs = set()
        self.local_iteration_limit = 1000
        self.global_iteration_limit = 1000
        self.model = dnn_model_wrapper(model)
        self.sensitive_param = sensitive_param
        self.threshold = threshold
        self.input_bounds = config.input_bounds
        self.perturbation_unit = 1
        self.max_time_allowed = max_time_allowed

        self.time_to_1000_disc = -1
        self.total_generated = 0
        self.cumulative_efficiency = []
        self.start_time = time.time()
        self.last_evaluated_input = None

    def normalise_probability(self):
        probability_sum = sum(self.param_probability)
        self.param_probability = [float(prob) / float(probability_sum) for prob in self.param_probability]

    class LocalPerturbation:
        def __init__(self, parent, stepsize=1):
            self.parent = parent
            self.stepsize = stepsize

        def __call__(self, x):
            s = self.stepsize
            param_choice = np.random.choice(range(self.parent.params), p=self.parent.param_probability)
            act = [-1, 1]
            direction_choice = np.random.choice(act, p=[self.parent.direction_probability[param_choice],
                                                        (1 - self.parent.direction_probability[param_choice])])

            if (x[param_choice] == self.parent.input_bounds[param_choice][0]) or (
                    x[param_choice] == self.parent.input_bounds[param_choice][1]):
                direction_choice = np.random.choice(act)

            x[param_choice] = x[param_choice] + (direction_choice * self.parent.perturbation_unit)

            x[param_choice] = max(self.parent.input_bounds[param_choice][0], x[param_choice])
            x[param_choice] = min(self.parent.input_bounds[param_choice][1], x[param_choice])

            ei = self.parent.evaluate_input(x)

            if (ei and direction_choice == -1) or (not ei and direction_choice == 1):
                self.parent.direction_probability[param_choice] = min(
                    self.parent.direction_probability[param_choice] + (
                            self.parent.direction_probability_change_size * self.parent.perturbation_unit), 1)
            elif (not ei and direction_choice == -1) or (ei and direction_choice == 1):
                self.parent.direction_probability[param_choice] = max(
                    self.parent.direction_probability[param_choice] - (
                            self.parent.direction_probability_change_size * self.parent.perturbation_unit), 0)

            if ei:
                self.parent.param_probability[param_choice] = self.parent.param_probability[
                                                                  param_choice] + self.parent.param_probability_change_size
                self.parent.normalise_probability()
            else:
                self.parent.param_probability[param_choice] = max(
                    self.parent.param_probability[param_choice] - self.parent.param_probability_change_size, 0)
                self.parent.normalise_probability()

            return x

    class GlobalDiscovery:
        def __init__(self, parent, stepsize=1):
            self.parent = parent
            self.stepsize = stepsize

        def __call__(self, x):
            for i in range(self.parent.params):
                random.seed(time.time())
                x[i] = random.randint(self.parent.input_bounds[i][0], self.parent.input_bounds[i][1])

            x[self.parent.sensitive_param - 1] = 0
            return x

    def evaluate_input(self, inp):
        inp0 = np.array([int(k) for k in inp])
        inp1 = np.array([int(k) for k in inp])
        original_sensitive_value = inp0[self.sensitive_param - 1]

        inp0 = inp0.reshape(1, -1)
        out0 = self.make_prediction(inp0)

        for i in range(self.input_bounds[self.sensitive_param - 1][0],
                       self.input_bounds[self.sensitive_param - 1][1] + 1):
            if original_sensitive_value != i:
                inp1[self.sensitive_param - 1] = i
                out1 = self.make_prediction(inp1.reshape(1, -1))
                if abs(out1 - out0) > self.threshold:
                    return True
        return False

    def evaluate_global(self, inp):
        inp0 = np.array([int(k) for k in inp])
        inp1 = np.array([int(k) for k in inp])

        original_sensitive_value = inp0[self.sensitive_param - 1]

        inp0 = inp0.reshape(1, -1)

        self.tot_inputs.add(tuple(map(tuple, inp0)))
        self.total_generated += 1

        if tuple(map(tuple, inp0)) in self.global_disc_inputs:
            return 0

        out0 = self.make_prediction(inp0)

        for i in range(self.input_bounds[self.sensitive_param - 1][0],
                       self.input_bounds[self.sensitive_param - 1][1] + 1):
            if original_sensitive_value != i:
                inp1[self.sensitive_param - 1] = i
                out1 = self.make_prediction(inp1.reshape(1, -1))
                if abs(out1 - out0) > self.threshold:
                    self.global_disc_inputs.add(tuple(map(tuple, inp0)))
                    self.global_disc_inputs_list.append(inp0.tolist()[0])

                    self.set_time_to_1000_disc()

                    return abs(out0 - out1)
        return 0

    def evaluate_local(self, inp):
        inp0 = np.array([int(k) for k in inp])
        inp1 = np.array([int(k) for k in inp])

        original_sensitive_value = inp0[self.sensitive_param - 1]

        inp0 = inp0.reshape(1, -1)
        inp_tuple = tuple(map(tuple, inp0))

        self.tot_inputs.add(inp_tuple)
        # Check if the input is different from the last evaluated input
        if not inp_tuple == self.last_evaluated_input:
            self.total_generated += 1

        # Update last_evaluated_input
        self.last_evaluated_input = inp_tuple
        self.total_generated += 1

        if (tuple(map(tuple, inp0)) in self.local_disc_inputs) or tuple(map(tuple, inp0)) in self.global_disc_inputs:
            return 0

        out0 = self.make_prediction(inp0)

        for i in range(self.input_bounds[self.sensitive_param - 1][0],
                       self.input_bounds[self.sensitive_param - 1][1] + 1):
            if original_sensitive_value != i:
                inp1[self.sensitive_param - 1] = i
                out1 = self.make_prediction(inp1.reshape(1, -1))
                if abs(out1 - out0) > self.threshold:
                    self.local_disc_inputs.add(tuple(map(tuple, inp0)))
                    self.local_disc_inputs_list.append(inp0.tolist()[0])

                    self.set_time_to_1000_disc()

                    return abs(out0 - out1)
        return 0

    def make_prediction(self, inp):
        output = self.model.predict(inp)[0]
        return (output > 0.5).astype(int)

    def update_cumulative_efficiency(self, iteration):
        """
                Update the cumulative efficiency data if the current number of total inputs
                meets the tracking criteria (first input or every tracking_interval inputs).
                """
        total_inputs = len(self.tot_inputs)
        total_disc = len(self.local_disc_inputs) + len(self.global_disc_inputs)
        self.cumulative_efficiency.append([time.time() - self.start_time, iteration, total_inputs, total_disc])

    def set_time_to_1000_disc(self):
        disc_inputs_count = len(self.global_disc_inputs) + len(self.local_disc_inputs)
        if disc_inputs_count >= 1000 and self.time_to_1000_disc == -1:
            self.time_to_1000_disc = time.time() - self.start_time
            print(f"\nTime to generate 1000 discriminatory inputs: {self.time_to_1000_disc:.2f} seconds")

    def run(self, max_global=1000, max_local=1000, max_allowed_time=3600):
        self.start_time = time.time()
        self.global_iteration_limit = max_global
        self.local_iteration_limit = max_local

        minimizer = {"method": "L-BFGS-B"}

        global_discovery = self.GlobalDiscovery(self)
        local_perturbation = self.LocalPerturbation(self)

        print("Search started")

        count = 300

        basinhopping(self.evaluate_global, self.config.initial_input, stepsize=1.0, take_step=global_discovery,
                     minimizer_kwargs=minimizer, niter=self.global_iteration_limit)
        print(f'Total global generation: {len(self.global_disc_inputs)}')

        for i in range(len(self.global_disc_inputs_list)):
            inp = self.global_disc_inputs_list[i]
            basinhopping(self.evaluate_local, inp, stepsize=1.0, take_step=local_perturbation,
                         minimizer_kwargs=minimizer, niter=self.local_iteration_limit)
            self.update_cumulative_efficiency(i)
            end = time.time()
            use_time = end - self.start_time
            if use_time >= count:
                count += 300
                self.report(elapsed_time=use_time, is_log=True)

            if count >= max_allowed_time or self.total_generated >= max_global*max_local:
                break

        elapsed_time = time.time() - self.start_time

        self.report(elapsed_time=elapsed_time, is_log=False)

    def report(self, elapsed_time, is_log: bool):
        disc_inputs = self.local_disc_inputs | self.global_disc_inputs

        generate_report(
            approach_name='AEQUITAS',
            dataset_name=self.config.dataset_name,
            classifier_name=self.model.__class__.__name__,
            sensitive_name=self.config.sens_name[self.sensitive_param],
            tot_inputs=self.tot_inputs,
            disc_inputs=disc_inputs,
            total_generated_inputs=self.total_generated,
            elapsed_time=elapsed_time,
            time_to_1000_disc=self.time_to_1000_disc,
            cumulative_efficiency=self.cumulative_efficiency,
            is_log=is_log,
        )


if __name__ == '__main__':
    config, sensitive_name, sensitive_param, classifier_name, max_allowed_time = get_experiment_params()

    print(f'Dataset: {config.dataset_name}')
    print(f'Classifier: {classifier_name}')
    print(f'Sensitive name: {sensitive_name}')
    print('')

    if classifier_name == 'dnn':
        import tensorflow

        classifier_path = f'models/{config.dataset_name}/{classifier_name}_slfc.keras'
        model = tensorflow.keras.models.load_model(classifier_path)
    else:

        classifier_path = f'models/{config.dataset_name}/{classifier_name}.pkl'
        model = joblib.load(classifier_path)

    aequitas = Aequitas(
        config=config,
        model=model,
        sensitive_param=sensitive_param,
    )

    aequitas.run(max_allowed_time=max_allowed_time)
