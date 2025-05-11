# Code to integrate PFI's adaptive feature selection into the PathSearcher class from aft.py
# Here is a draft implementation to achieve Pattern-Enhanced Path Exploration by combining AFT and PFI functionalities.
import os
import sys

import numpy as np
# Get the absolute path to the directory where vbtx.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Two levels up from vbtx.py
sys.path.append(os.path.join(base_path, "../../"))
from baseline.aft.PathSearcher import PathSearcher


class HybridPathSearcher(PathSearcher):
    def __init__(self, DT, CuT, protected_list_no, data_range, protected_value_comb, sensitive_indices, IntervalP=None):
        super().__init__(DT, CuT, protected_list_no, data_range, protected_value_comb, IntervalP)

        # Initialize attributes from PFI for adaptive feature selection
        self.feature_importance = np.ones(len(data_range))  # Assume each feature has initial equal importance
        self.recent_disc_patterns = []  # List to keep track of recent discriminatory patterns
        self.max_patterns = 50  # Limit to the number of patterns to track
        self.sensitive_indices = sensitive_indices  # Sensitive attribute indices for focused adaptation

    def adaptive_feature_selection(self):
        """
        Adaptively select features based on importance scores and recent patterns.
        Adjusts importance scores to prioritize features involved in recent discriminatory patterns.
        """
        for pattern in self.recent_disc_patterns:
            for idx in self.sensitive_indices:
                self.feature_importance[idx] += 0.1  # Increment importance if feature is part of a recent pattern

        # Normalize feature importance scores to avoid extreme values
        total_importance = np.sum(self.feature_importance)
        if total_importance > 0:
            self.feature_importance /= total_importance

    def add_discriminatory_pattern(self, pattern):
        """
        Add a new discriminatory pattern to recent history and update feature importance.
        """
        if len(self.recent_disc_patterns) >= self.max_patterns:
            self.recent_disc_patterns.pop(0)  # Remove the oldest pattern if exceeding max allowed
        self.recent_disc_patterns.append(pattern)
        self.adaptive_feature_selection()

    def detect_disc_from_path_pair(self, pair, max_train_data_each_path, max_sample_each_path, check_type):
        """
        Override detect_disc_from_path_pair to integrate adaptive feature selection in path exploration.
        """
        super().detect_disc_from_path_pair(pair, max_train_data_each_path, max_sample_each_path, check_type)

        # After detecting discrimination in the path pair, update the feature importance if discrimination found
        if self.disc_data:  # If discriminatory data is found, record pattern and update
            new_pattern = {"paths": [pair["path1"], pair["path2"]], "sensitive_features": self.sensitive_indices}
            self.add_discriminatory_pattern(new_pattern)

    def sample(self, dt_search_mode="random+flip", check_type="themis", MaxTry=10000, MaxDiscPathPair=100,
               max_train_data_each_path=10, max_sample_each_path=100):
        """
        Enhanced sampling method with adaptive feature selection.
        """
        # Run the original path sampling with adaptive feature selection applied
        self.adaptive_feature_selection()
        return super().sample(dt_search_mode, check_type, MaxTry, MaxDiscPathPair, max_train_data_each_path,
                              max_sample_each_path)

# The HybridPathSearcher can now be instantiated and used in place of the original PathSearcher
# to implement Pattern-Enhanced Path Exploration for fairness testing.

