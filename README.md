# Search-Based Fairness Testing for Detecting Individual Discrimination in Machine Learning-Based Software
**Experiments Source Code**

## Installation

1. Download/Clone the repository:
   ```bash
   Download from: https://github.com/hmamman/phd-thesis-code/archive/refs/heads/main.zip
   Unzip and cd into the directory
   OR
   Clone from: https://github.com/hmamman/phd-thesis-code.git


2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
## Datasets and Protected Attributes
| Dataset | Protected Attribute | Index (Starts at 1) |
|---------|---------------------|---------------------|
| census  | sex                 | 9                   |
|         | age                 | 1                   |
|         | race                | 8                   |
| credit  | sex                 | 9                   |
|         | age                 | 13                  |
| bank    | age                 | 1                   |
|         | marital             | 3                   |
| compas  | sex                 | 1                   |
|         | age                 | 2                   |
|         | race                | 3                   |
| meps    | sex                 | 3                   |

Dataset and protected attribute names are case-sensitive.

## Running Fairness Testing

### Command-Line Arguments

The script accepts the following arguments:

- `--dataset_name`: (string) Name of the dataset to use in the experiment. The default is `'census'`.
  - Example: `--dataset_name census`

- `--sensitive_name`: (string) Name of the protected attribute for fairness testing (e.g., `sex`, `age`, `race`). The default is `'age'`.
  - Example: `--sensitive_name sex`

- `--classifier_name`: (string) Name of the classifier to use (e.g., `mlp`, `dt`, `rf`, ect.). The default is `'dt'`.
  - Example: `--classifier_name svm`

- `--max_allowed_time`: (integer) Maximum time in seconds for the experiment to run. The default is `3600` seconds (1 hour).
  - Example: `--max_allowed_time 3600`

### Example Usage

To run the any framework (e.g., fairses, fairpes, fairces, fairesteo, fairphs, fairsift):
```bash
python ./tutorial/fairsift.py --classifier_name dt --dataset_name census --sensitive_name age --max_allowed_time 3600
```

To run a specific baseline approach included in this repository (e.g., aequitas, sg, adf, neuronfair, expga, vbtx, aft):
```bash
python ./baseline/expga/expga.py --classifier_name dt --dataset_name census --sensitive_name age --max_allowed_time 3600
```

You can also run all experiments for an approach by running the main.py file:
```bash
python ./main.py --approach_name fairsift --max_allowed_time 3600 --max_iteration 1  
```

