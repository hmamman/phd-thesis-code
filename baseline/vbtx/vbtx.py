import os
import sys
import time
import random

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Get the absolute path to the directory where vbtx.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Two levels up from vbtx.py
sys.path.append(os.path.join(base_path, "../../"))

from baseline.vbtx.XORSampler import XORSampler
from baseline.vbtx.SearchTree import Tree2SMT

from utils.helpers import generate_report, get_experiment_params

from sklearn.base import BaseEstimator, ClassifierMixin


class Sequential(BaseEstimator, ClassifierMixin):
    def __str__(self):
        return 'Sequential'
    
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        import tensorflow as tf
        inp = tf.convert_to_tensor(X, dtype=tf.float32)
        prob = self.model(inp, training=False)  # Model output

        # Check if the output is a single-column (binary classification)
        if prob.shape[1] == 1:
            # Convert single-column probabilities to two columns [P(class 0), P(class 1)]
            prob = tf.concat([1 - prob, prob], axis=1)
        return prob.numpy()

    def predict(self, X):
        prob = self.predict_proba(X)
        return prob.argmax(axis=1)

def m(model):
    if hasattr(model, 'layers'):
        model = Sequential(model=model)
    return model


class VBTX:
    def __str__(self):
        return 'VBTX'

    def __init__(self, config, model, sensitive_param):
        self.start_time = time.time()
        self.config = config
        self.sensitive_param = sensitive_param
        self.model = m(model)
        self.preds = self.model.predict

        self.disc_inputs = set()
        self.disc_inputs_list = []
        self.tot_inputs = set()
        self.total_generated = 0
        self.elapsed_time = 0
        self.time_to_1000_disc = -1
        self.cumulative_efficiency = []

        # VBTX specific initialization
        self.tree2smt = Tree2SMT(
            feature_names=self.config.feature_name,
            class_name=self.config.output_feature,
            protected_att=[self.sensitive_param - 1],
            vbtx_ver="improved"
        )
        self.train_data = []
        self.disc_data = []
        self.test_data = []
        self.no_train_data_sample = 5000  # Default value, can be adjusted
        self.vbtx_ver = "improved"
        self.no_test = 0
        self.no_disc = 0

    def create_train_data(self, num):
        """Generate initial training data"""
        self.train_data = []
        for _ in range(num):
            temp = []
            for i in range(len(self.config.input_bounds)):
                bound = self.config.input_bounds[i]
                temp.append(random.randint(bound[0], bound[1]))
            temp.append(int(self.preds(np.array([temp]))[0]))
            self.train_data.append(temp)

    def train_approximate_DT(self):
        """Train decision tree approximation of the model"""
        X = [item[:-1] for item in self.train_data]
        Y = [item[-1] for item in self.train_data]
        clf = DecisionTreeClassifier(criterion="entropy")
        return clf.fit(X, Y)

    def check_disc(self, testdata):
        """Check for discriminatory instances"""
        no_test = len(testdata) // 2
        X = [item[:-1] for item in testdata]
        Y = [int(item[-1]) for item in testdata]
        real_Y = self.preds(np.array(X))

        train_data_count = 0

        for i in range(0, no_test * 2, 2):
            equal1 = Y[i] == real_Y[i]
            equal2 = Y[i + 1] == real_Y[i + 1]

            if not equal1:
                self.train_data.append(X[i] + [real_Y[i]])
                train_data_count += 1
            if not equal2:
                self.train_data.append(X[i + 1] + [real_Y[i + 1]])
                train_data_count += 1

            if equal1 and equal2:

                if tuple(X[i]) not in self.disc_inputs:
                    self.disc_inputs.add(tuple(X[i]))
                    self.disc_inputs_list.append(X[i])
                self.set_time_to_1000_disc()

            # Update total inputs tracking
            self.tot_inputs.add(tuple(X[i]))
            self.total_generated += 1

        return train_data_count

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

    def run(self, max_test_data=1000*1000, max_allowed_time=300):
        """Main testing loop matching SG's interface"""
        self.start_time = time.time()
        restart_flag = True
        self.no_test = 0
        no_new_train_count = 0
        loop = 0
        count = 300

        while True:
            end = time.time()
            use_time = end - self.start_time
            if use_time >= count:
                count += 300
                self.report(elapsed_time=use_time, is_log=True)

            if count >= max_allowed_time:
                break

            if self.total_generated >= max_test_data:
                break

            # Step 1: Create/update approximation model
            if restart_flag:
                self.create_train_data(self.no_train_data_sample)
                restart_flag = False
            DT = self.train_approximate_DT()

            # Step 2: Convert DT to SMT formula
            smt_str = self.tree2smt.dt_to_smt(DT)
            param_xor = self.tree2smt.get_parm_xor()

            # Step 3: Generate run cases
            sampler = XORSampler(
                smt_str=smt_str,
                param_xor=param_xor,
                max_loop=1000,
                max_path=50,
                no_of_xor=5,
                need_only_one_sol=False,
                need_change_s=True,
                need_blocking=False,
                class_list=[self.config.feature_name[-1]],
                protected_list=[self.config.feature_name[self.sensitive_param - 1]]
            )

            satFlag, test_data = sampler.sample()

            if satFlag:
                # Step 4 & 5: Execute tests and update training data
                self.no_test += len(test_data) // 2
                self.test_data += test_data
                # self.total_generated += len(test_data)

                if self.check_disc(test_data) == 0:
                    no_new_train_count += 1
                    if no_new_train_count >= 5:
                        restart_flag = True
                        no_new_train_count = 0
                else:
                    no_new_train_count = 0
            else:
                restart_flag = True
            
            self.update_cumulative_efficiency(loop)

            loop += 1

        elapsed_time = time.time() - self.start_time
        self.report(elapsed_time=elapsed_time, is_log=False)

    def report(self, elapsed_time, is_log: bool):
        """Generate report matching SG's format"""

        from utils.helpers import generate_report
        generate_report(
            approach_name='VBTX',
            dataset_name=self.config.dataset_name,
            classifier_name=self.model.__class__.__name__,
            sensitive_name=self.config.sens_name[self.sensitive_param],
            tot_inputs=self.tot_inputs,
            disc_inputs=self.disc_inputs,
            total_generated_inputs=self.total_generated,
            elapsed_time=elapsed_time,
            time_to_1000_disc=self.time_to_1000_disc,
            cumulative_efficiency=self.cumulative_efficiency,
            is_log=is_log
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

    vbtx = VBTX(
        config=config,
        model=model,
        sensitive_param=sensitive_param,
    )

    vbtx.run(max_allowed_time=max_allowed_time)