import time
import joblib
import numpy as np
import tensorflow as tf
import sys, os

from keras import Model
from scipy.optimize import basinhopping
from sklearn.cluster import KMeans
import copy

# Adjust the import path as needed
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, "../../"))
from utils.helpers import get_data, generate_report, get_experiment_params


class ADF:
    def __init__(self, config, model, sensitive_param, cluster_num=4, max_global=1000, max_local=1000, max_iter=40):

        # Set GPU configuration
        self.approach_name = 'ADF'
        self.start_time = time.time()
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Ensure GPU 2 is available
        np.seterr(all='ignore')

        self.config = config
        self.input_bounds = self.config.input_bounds
        self.sensitive_param = sensitive_param
        self.model = model
        self.cluster_num = cluster_num
        self.max_global = max_global
        self.max_local = max_local
        self.max_iter = max_iter
        self.perturbation_size = 1

        # Initialize testing results storage
        self.tot_inputs = set()
        self.global_disc_inputs = set()
        self.global_disc_inputs_list = []
        self.local_disc_inputs = set()
        self.local_disc_inputs_list = []

        # Clustering model and dataset details
        self.input_bounds = self.config.input_bounds
        self.dataset_name = self.config.dataset_name

        self.time_to_1000_disc = -1
        self.total_generated = 0
        self.cumulative_efficiency = []
        self.tracking_interval = 1000
        self.last_evaluated_input = None

        # Build the model and prepare weights
        self.build_dnn_model()

    def build_dnn_model(self):
        # Get data
        data = get_data(self.config.dataset_name)
        self.X, self.Y, self.input_shape, self.nb_classes = data()

        # Create model inputs
        self.x = tf.keras.Input(shape=self.input_shape[1:])
        self.nx = tf.keras.Input(shape=self.input_shape[1:])

        if not self.model.built:
            self.model.build(input_shape=(None, self.input_shape[1]))

    @tf.function
    def compute_gradients(self, x, nx):
        """Compute gradients using gradient tape"""
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(nx)

            # Predictions and losses
            preds_x = self.model(x, training=False)
            preds_nx = self.model(nx, training=False)
            loss = tf.reduce_mean(tf.square(preds_x - preds_nx))

        # Gradients
        x_grad = tape.gradient(loss, x)
        nx_grad = tape.gradient(loss, nx)
        del tape
        return x_grad, nx_grad

    def cluster(self):
        """
        Construct the K-means clustering model to increase the complexity of discrimination
        :return: the K_means clustering model
        """
        path = 'clusters/' + self.config.dataset_name + '.pkl'
        if os.path.exists(path):
            clf = joblib.load(path)
        else:
            clf = KMeans(n_clusters=self.cluster_num, random_state=2019).fit(self.X)
            joblib.dump(clf, path)
        return clf

    def clip(self, inp):
        """
        Clip the generating instance with each feature to make sure it is valid
        :param inp: generating instance
        :return: a valid generating instance
        """
        for i in range(len(inp)):
            inp[i] = max(inp[i], self.input_bounds[i][0])
            inp[i] = min(inp[i], self.input_bounds[i][1])
        return inp

    def check_for_error_condition(self, inp):
        """
        Check whether the test case is an individual discriminatory instance.
        :param inp: The input instance to check
        :return: True if the instance is discriminatory, False otherwise
        """
        inp = inp.astype('int')
        inp_tuple = tuple(inp)

        self.tot_inputs.add(inp_tuple)
        # Check if the input is different from the last evaluated input
        if not inp_tuple == self.last_evaluated_input:
            self.total_generated += 1

        # Update last_evaluated_input
        self.last_evaluated_input = inp_tuple

        _, label = self.make_prediction(np.array([inp]))

        for val in range(self.config.input_bounds[self.sensitive_param - 1][0],
                         self.config.input_bounds[self.sensitive_param - 1][1] + 1):

            if val != inp[self.sensitive_param - 1]:
                tnew = inp.copy()
                tnew[self.sensitive_param - 1] = val
                _, label_new = self.make_prediction(np.array([tnew]))
                if label_new != label:
                    return True

        return False

    def make_prediction(self, inp):
        inp = tf.convert_to_tensor(inp, dtype=tf.float32)
        prob = self.model(inp, training=False)[0]
        label = tf.cast(prob > 0.5, tf.int32).numpy()
        return prob, label

    def seed_test_input(self, clusters, limit):
        """Select seed inputs for testing"""
        i = 0
        rows = []
        max_size = max([len(c[0]) for c in clusters])
        while i < max_size:
            if len(rows) == limit:
                break
            for c in clusters:
                if i >= len(c[0]):
                    continue
                index = c[0][i]
                row = self.X[index:index + 1]
                rows.append(row)
                if len(rows) == limit:
                    break
            i += 1
        return np.vstack(rows)  # Ensure correct shape

    def evaluate_local(self, inp):
        inp0 = inp.astype('int')

        result = self.check_for_error_condition(inp0)

        if result and tuple(inp0) not in self.global_disc_inputs and tuple(inp0) not in self.local_disc_inputs:
            self.local_disc_inputs.add(tuple(inp0))
            self.local_disc_inputs_list.append(inp0.tolist())
            self.set_time_to_1000_disc()
        return 1 if result else 0

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

    def run(self, max_allowed_time=3600):
        """Run the fairness testing process"""
        self.start_time = time.time()

        # Perform clustering
        clf = self.cluster()
        clusters = [np.where(clf.labels_ == i) for i in range(self.cluster_num)]

        # Select seed inputs
        inputs = self.seed_test_input(clusters, min(self.max_global, len(self.X)))

        """Run global and local search"""
        sd = 0  # Random seed offset

        count = 300

        for num in range(len(inputs)):
            index = num  # Assuming 'inputs' is already the data, not indices
            sample = inputs[index:index + 1]  # Shape (1, features)
            memory1 = sample[0] * 0
            memory2 = sample[0] * 0 + 1
            memory3 = sample[0] * 0 - 1

            # start global perturbation
            for iter in range(self.max_iter + 1):
                prob, label = self.make_prediction(sample)
                max_diff = 0
                n_value = -1
                # search the instance with maximum probability difference for global perturbation
                for i in range(self.input_bounds[self.sensitive_param - 1][0],
                               self.input_bounds[self.sensitive_param - 1][1] + 1):
                    if i != sample[0][self.sensitive_param - 1]:
                        n_sample = sample.copy()
                        n_sample[0][self.sensitive_param - 1] = i
                        n_prob, n_label = self.make_prediction(n_sample)
                        if label != n_label:
                            n_value = i
                            break
                        else:
                            prob_diff = np.abs(prob - n_prob).sum()
                            if prob_diff > max_diff:
                                max_diff = prob_diff
                                n_value = i

                temp = copy.deepcopy(sample[0].astype('int').tolist())
                # temp = temp[:self.sensitive_param - 1] + temp[self.sensitive_param:]

                self.tot_inputs.add(tuple(temp))
                self.total_generated += 1

                # if get an individual discriminatory instance
                if label != n_label and (tuple(temp) not in self.global_disc_inputs) and (
                        tuple(temp) not in self.local_disc_inputs):
                    self.global_disc_inputs_list.append(temp)
                    self.global_disc_inputs.add(tuple(temp))
                    self.set_time_to_1000_disc()
                    # start local perturbation
                    local_perturbation = LocalPerturbation(
                        n_value=n_value,
                        parent=self
                    )
                    # Local perturbation
                    minimizer = {"method": "L-BFGS-B"}

                    basinhopping(self.evaluate_local, temp, stepsize=1.0,
                                 take_step=local_perturbation,
                                 minimizer_kwargs=minimizer,
                                 niter=self.max_local)
                    # # Perform local perturbation
                    # for _ in range(self.max_local):
                    #     local_perturbation(temp)
                    break

                # Compute gradients using GradientTape
                x_tensor = tf.convert_to_tensor(sample, dtype=tf.float32)
                n_sample = np.copy(sample)
                n_sample[0][self.sensitive_param - 1] = n_value
                nx_tensor = tf.convert_to_tensor(n_sample, dtype=tf.float32)

                s_grad, n_grad = self.compute_gradients(x_tensor, nx_tensor)
                sn_grad = s_grad + n_grad

                s_grad = tf.sign(s_grad).numpy()[0]
                n_grad = tf.sign(n_grad).numpy()[0]
                sn_grad = tf.sign(sn_grad).numpy()[0]

                # find the feature with same impact
                if np.all(s_grad == 0):
                    g_diff = n_grad
                elif np.all(n_grad == 0):
                    g_diff = s_grad
                else:
                    g_diff = (s_grad == n_grad).astype(float)

                g_diff[self.sensitive_param - 1] = 0
                if np.all(g_diff == 0):
                    g_diff = sn_grad
                    g_diff[self.sensitive_param - 1] = 0
                if np.all(s_grad == 0) or np.array_equal(memory1, memory3):
                    np.random.seed(seed=2020 + sd)
                    sd += 1
                    delta = self.perturbation_size
                    s_grad = np.random.randint(-delta, delta + 1, size=s_grad.shape)

                g_diff = np.ones_like(g_diff)
                g_diff[self.sensitive_param - 1] = 0
                cal_grad = s_grad * g_diff  # g_diff:
                memory1 = memory2
                memory2 = memory3
                memory3 = cal_grad
                sample[0] = self.clip(sample[0] + self.perturbation_size * cal_grad).astype("int")
                if iter == self.max_iter:
                    break

            self.update_cumulative_efficiency(num)

            end = time.time()
            use_time = end - self.start_time
            if use_time >= count:
                count += 300
                self.report(elapsed_time=use_time, is_log=True)

            if use_time >= max_allowed_time or self.total_generated >= self.max_global * self.max_local:
                print('Max time or generation limit exceeded!')
                break

        elapsed_time = time.time() - self.start_time

        self.report(elapsed_time=elapsed_time, is_log=False)

    def report(self, elapsed_time, is_log: bool):

        disc_inputs = self.local_disc_inputs | self.global_disc_inputs

        generate_report(
            approach_name=self.approach_name,
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


class LocalPerturbation:
    def __init__(self, parent, n_value):
        self.parent = parent
        self.n_value = n_value

    def __call__(self, x):
        """Perform local perturbation"""
        s = np.random.choice([1.0, -1.0]) * self.parent.perturbation_size

        n_x = x.copy()
        n_x[self.parent.sensitive_param - 1] = self.n_value

        x_tensor = tf.convert_to_tensor([x], dtype=tf.float32)
        nx_tensor = tf.convert_to_tensor([n_x], dtype=tf.float32)

        ind_grad, n_ind_grad = self.parent.compute_gradients(x_tensor, nx_tensor)

        ind_grad = ind_grad.numpy()[0]
        n_ind_grad = n_ind_grad.numpy()[0]

        if (np.all(ind_grad == 0) and np.all(n_ind_grad == 0)):
            probs = 1.0 / (len(x) - 1) * np.ones(len(x))
            probs[self.parent.sensitive_param - 1] = 0
        else:
            grad_sum = 1.0 / (np.abs(ind_grad) + np.abs(n_ind_grad))
            grad_sum[self.parent.sensitive_param - 1] = 0
            probs = grad_sum / np.sum(grad_sum)

        probs = probs / probs.sum()
        index = np.random.choice(range(len(x)), p=probs)

        local_cal_grad = np.zeros(len(x))
        local_cal_grad[index] = 1.0

        x = self.parent.clip(x + s * local_cal_grad).astype("int")
        return x

if __name__ == '__main__':
    config, sensitive_name, sensitive_param, classifier_name, max_allowed_time = get_experiment_params()

    print(f'Dataset: {config.dataset_name}')
    print(f'Classifier: {classifier_name}')
    print(f'Sensitive name: {sensitive_name}')
    print('')

    classifier_path = f'models/{config.dataset_name}/{classifier_name}_slfc.keras'
    model = tf.keras.models.load_model(classifier_path)

    adf = ADF(
        config=config,
        model=model,
        sensitive_param=sensitive_param,
    )

    adf.run(max_allowed_time=max_allowed_time)
