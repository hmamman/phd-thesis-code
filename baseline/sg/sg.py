import time
import joblib
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from queue import PriorityQueue
from z3 import *
from lime import lime_tabular

# Get the absolute path to the directory where sg.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Two levels up from sg.py
sys.path.append(os.path.join(base_path, "../../"))

from utils.helpers import get_data, generate_report, get_experiment_params, cluster
from utils.dnn_wrapper import dnn_model_wrapper


class SG:
    def __str__(self):
        return 'SG'

    def __init__(self, config, model, sensitive_param):
        self.start_time = time.time()
        self.config = config
        self.sensitive_param = sensitive_param
        self.arguments = self.gen_arguments()
        self.rank1, self.rank2, self.rank3 = 5, 1, 10
        self.T1 = 0.3
        self.global_disc_inputs = set()
        self.global_disc_inputs_list = []
        self.local_disc_inputs = set()
        self.local_disc_inputs_list = []
        self.tot_inputs = set()

        self.model = dnn_model_wrapper(model)

        self.elapsed_time = 0
        self.time_to_1000_disc = -1
        self.total_generated = 0
        self.cumulative_efficiency = []

        self.load_data()

    def load_data(self):
        data = get_data(self.config.dataset_name)
        self.X, self.Y, self.input_shape, self.nb_classes = data()

    def gen_arguments(self):
        return [Int(name) for name in self.config.feature_name]

    def global_discovery(self, iteration):
        samples = []
        for _ in range(iteration):
            x = np.array([random.randint(bound[0], bound[1]) for bound in self.config.input_bounds])
            samples.append(x)
        return np.array(samples)

    def cluster(self, cluster_num):
        """
        Construct the K-means clustering model to increase the complexity of discrimination
        :return: the K_means clustering model
        """
        path = 'clusters/' + self.config.dataset_name + '.pkl'
        if os.path.exists(path):
            clf = joblib.load(path)
        else:
            clf = KMeans(n_clusters=cluster_num, random_state=2019).fit(self.X)
            joblib.dump(clf, path)
        return clf

    def seed_test_input(self, cluster_num, limit):
        clf = self.cluster(cluster_num)
        clusters = [np.where(clf.labels_ == i) for i in range(cluster_num)]
        rows = []
        max_size = max(len(c[0]) for c in clusters)
        for i in range(max_size):
            if len(rows) == limit:
                break
            for c in clusters:
                if i < len(c[0]):
                    rows.append(c[0][i])
        return np.array(rows)

    def getPath(self, input, X):
        explainer = lime_tabular.LimeTabularExplainer(
            X,
            feature_names=self.config.feature_name,
            class_names='y',
            categorical_features=self.config.categorical_features,
            discretize_continuous=True
        )
        _, g_data = explainer._LimeTabularExplainer__data_inverse(input, num_samples=5000)
        g_labels = self.make_prediction(g_data)
        tree = DecisionTreeClassifier(random_state=2019)
        tree.fit(g_data, g_labels)
        path_index = tree.decision_path(np.array([input])).indices
        path = []
        for i in range(len(path_index)):
            node = path_index[i]
            i = i + 1
            f = tree.tree_.feature[node]
            if f != -2:
                left_count = tree.tree_.n_node_samples[tree.tree_.children_left[node]]
                right_count = tree.tree_.n_node_samples[tree.tree_.children_right[node]]
                left_confidence = 1.0 * left_count / (left_count + right_count)
                right_confidence = 1.0 - left_confidence
                if tree.tree_.children_left[node] == path_index[i]:
                    path.append([f, "<=", tree.tree_.threshold[node], left_confidence])
                else:
                    path.append([f, ">", tree.tree_.threshold[node], right_confidence])
        return path

    def check_for_error_condition(self, t):
        label = self.make_prediction(np.array([t]))[0]
        for val in range(self.config.input_bounds[self.sensitive_param - 1][0],
                         self.config.input_bounds[self.sensitive_param - 1][1] + 1):
            if val != t[self.sensitive_param - 1]:
                tnew = copy.deepcopy(t)
                tnew[self.sensitive_param - 1] = val
                label_new = self.make_prediction(np.array([tnew]))[0]
                if label_new != label:
                    return True
        return False

    def global_solve(self, path_constraint, t):
        s = Solver()
        for c in path_constraint:
            s.add(self.arguments[c[0]] >= self.config.input_bounds[c[0]][0])
            s.add(self.arguments[c[0]] <= self.config.input_bounds[c[0]][1])
            if c[1] == "<=":
                s.add(self.arguments[c[0]] <= c[2])
            else:
                s.add(self.arguments[c[0]] > c[2])

        if s.check() == sat:
            m = s.model()
            tnew = copy.deepcopy(t)
            for i in range(len(self.arguments)):
                if m[self.arguments[i]] is not None:
                    tnew[i] = int(m[self.arguments[i]].as_long())
            return tnew.astype('int').tolist()
        return None

    def average_confidence(self, path_constraint):
        return np.mean(np.array(path_constraint)[:, 3].astype(float))

    def local_solve(self, path_constraint, t, index):
        c = path_constraint[index]
        s = Solver()
        s.add(self.arguments[c[0]] >= self.config.input_bounds[c[0]][0])
        s.add(self.arguments[c[0]] <= self.config.input_bounds[c[0]][1])
        for i in range(len(path_constraint)):
            if path_constraint[i][0] == c[0]:
                if path_constraint[i][1] == "<=":
                    s.add(self.arguments[path_constraint[i][0]] <= path_constraint[i][2])
                else:
                    s.add(self.arguments[path_constraint[i][0]] > path_constraint[i][2])

        if s.check() == sat:
            m = s.model()
            tnew = copy.deepcopy(t)
            tnew[c[0]] = int(m[self.arguments[c[0]]].as_long())
            return tnew.astype('int').tolist()
        return None

    def make_prediction(self, inp):
        if hasattr(self.model, 'layers'):
            inp = tf.convert_to_tensor(inp, dtype=tf.float32)
            prob = self.model(inp, training=False)
            return tf.cast(prob > 0.5, tf.int32).numpy()

        output = self.model.predict(inp)

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

    def run(self, cluster_num=4, limit=1000, max_allowed_time=300):
        self.start_time = time.time()

        inputs = self.seed_test_input(cluster_num, limit)
        X = self.X.copy()

        q = PriorityQueue()
        for inp in inputs[::-1]:
            q.put((self.rank1, X[inp].tolist()))

        visited_path = []
        count = 300
        loop = 0
        while self.total_generated < limit * limit and not q.empty():
            end = time.time()
            use_time = end - self.start_time
            if use_time >= count:
                count += 300
                self.report(elapsed_time=use_time, is_log=True)

            if count >= max_allowed_time:
                break

            t_rank, t = q.get()
            t = np.array(t)
            found = self.check_for_error_condition(t)
            p = self.getPath(t, X)
            temp = copy.deepcopy(t.tolist())
            temp = temp[:self.sensitive_param - 1] + temp[self.sensitive_param:]

            self.tot_inputs.add(tuple(temp))

            self.total_generated += 1

            if found:
                if (tuple(temp) not in self.global_disc_inputs) and (tuple(temp) not in self.local_disc_inputs):
                    if t_rank > 2:
                        self.global_disc_inputs.add(tuple(temp))
                        self.global_disc_inputs_list.append(temp)
                    else:
                        self.local_disc_inputs.add(tuple(temp))
                        self.local_disc_inputs_list.append(temp)

                self.set_time_to_1000_disc()

                # Local search
                for i in range(len(p)):
                    path_constraint = copy.deepcopy(p)
                    c = path_constraint[i]
                    if c[0] == self.sensitive_param - 1:
                        continue
                    c[1] = ">" if c[1] == "<=" else "<="
                    c[3] = 1.0 - c[3]

                    if path_constraint not in visited_path:
                        visited_path.append(path_constraint)
                        input = self.local_solve(path_constraint, t, i)
                        if input is not None:
                            r = self.average_confidence(path_constraint)
                            q.put((self.rank2 + r, input))

            # Global search
            prefix_pred = []
            for c in p:
                if c[0] == self.sensitive_param - 1:
                    continue
                if c[3] < self.T1:
                    break

                n_c = copy.deepcopy(c)
                n_c[1] = ">" if n_c[1] == "<=" else "<="
                n_c[3] = 1.0 - c[3]
                path_constraint = prefix_pred + [n_c]

                if path_constraint not in visited_path:
                    visited_path.append(path_constraint)
                    input = self.global_solve(path_constraint, t)
                    if input is not None:
                        r = self.average_confidence(path_constraint)
                        q.put((self.rank3 - r, input))

                prefix_pred.append(c)
            self.update_cumulative_efficiency(loop)
            loop += 1

        elapsed_time = time.time() - self.start_time

        self.report(elapsed_time=elapsed_time, is_log=False)

    def report(self, elapsed_time, is_log: bool):
        disc_inputs = self.local_disc_inputs | self.global_disc_inputs

        generate_report(
            approach_name='SG',
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
        import tensorflow as tf

        classifier_path = f'models/{config.dataset_name}/{classifier_name}_slfc.keras'
        model = tf.keras.models.load_model(classifier_path)
    else:

        classifier_path = f'models/{config.dataset_name}/{classifier_name}.pkl'
        model = joblib.load(classifier_path)

    sg = SG(
        config=config,
        model=model,
        sensitive_param=sensitive_param
    )

    sg.run(max_allowed_time=max_allowed_time)
