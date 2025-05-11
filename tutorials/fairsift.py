import os
import sys

import joblib
import numpy as np


# Get the absolute path to the directory where fairsphs.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Two levels up from fairsphs.py
sys.path.append(os.path.join(base_path, "../"))

from tutorials.algorithms.sift import SIFT
from utils.helpers import get_experiment_params
from tutorials.fairses import FairSES


#Batch Inference Evaluation
class FairSIFT(FairSES):
    def __init__(self, config, model, sensitive_param, population_size=200, threshold=0):
        super().__init__(config, model, sensitive_param, population_size, threshold)
        self.approach_name = "FairSIFT"

        self.setup = SIFT(
            mu=self.population_size,
            bounds=np.array(self.config.input_bounds),
            fitness_func=self.batch_evaluate_disc
        )

    def batch_evaluate_disc(self, offsprings):
        # Precompute sensitive values during initialization
        if not hasattr(self, 'sensitive_metadata'):
            self.sensitive_param_index = self.sensitive_param - 1
            bounds = self.config.input_bounds[self.sensitive_param_index]
            self.all_sensitive_values = np.arange(bounds[0], bounds[1] + 1)
            self.num_sensitive_values = len(self.all_sensitive_values)
            self.sensitive_axes = (self.sensitive_param_index,)

        # Convert all offsprings to numpy arrays and hashable tuples
        offsprings_array = np.asarray(offsprings, dtype=int)
        offsprings_tuples = [tuple(c) for c in offsprings_array]
        num_offsprings = len(offsprings)

        # Update global tracking
        self.tot_inputs.update(offsprings_tuples)
        self.total_generated += num_offsprings

        # Identify candidates needing evaluation
        needs_eval = np.array([
            t not in self.disc_inputs
            for t in offsprings_tuples
        ], dtype=bool)

        if not np.any(needs_eval):
            return np.zeros(num_offsprings, dtype=int)

        # Prepare batch inputs for candidates requiring evaluation
        eval_candidates = offsprings_array[needs_eval]
        original_sensitives = eval_candidates[:, self.sensitive_param_index]

        # Generate all sensitive variations using broadcasting
        batch_per_candidate = self.num_sensitive_values
        batch_inputs = np.repeat(eval_candidates, batch_per_candidate, axis=0)

        # Create indices for sensitive value replacement
        replace_indices = np.tile(self.all_sensitive_values, len(eval_candidates))
        batch_inputs[:, self.sensitive_param_index] = replace_indices

        # Get original indices for each candidate's variations
        original_mask = (replace_indices == original_sensitives.repeat(batch_per_candidate))
        original_indices = np.where(original_mask)[0]

        # Perform single batch prediction
        all_outputs = self.model.predict(batch_inputs)
        self.inference_count += 1

        # Extract original outputs using advanced indexing
        original_outputs = all_outputs[original_indices]

        # Compare variations with originals using boolean matrix
        output_matrix = all_outputs.reshape(len(eval_candidates), batch_per_candidate)
        is_different = (output_matrix != original_outputs[:, np.newaxis]).any(axis=1)

        # Update discriminatory inputs
        discriminatory_candidates = eval_candidates[is_different]
        for candidate in discriminatory_candidates:
            candidate_tuple = tuple(candidate)
            self.disc_inputs.add(candidate_tuple)
            self.disc_inputs_list.append(candidate.tolist())

        # Set time if any discrimination found
        if len(discriminatory_candidates) > 0:
            self.set_time_to_1000_disc()

        # Build final results array
        results = np.zeros(num_offsprings, dtype=int)
        eval_indices = np.where(needs_eval)[0]
        results[eval_indices] = is_different.astype(int)

        return results


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

    fairsift = FairSIFT(
        config=config,
        model=model,
        sensitive_param=sensitive_param
    )

    fairsift.run(max_allowed_time=max_allowed_time)
