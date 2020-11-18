# covid19-prediction
Virtual Environment Setup

- `python3 -m pip install --user virtualenv`
- `python3 -m venv <env_name>`
- `source <env_name>/bin/activate`
- `pip3 install -r requirements.txt`

# Model hierarchy

- Doing state specific models
- May eventually use GMM to cluster states together based on their similarities, and then create models for specific clusters.

# Pre-Processing Steps:

1. Dealing with NaNs effectively
  - Average
  - Median
  - Set to zeros
  - Drop those rows entirely
  - Can drop that feature for that particular state.
2. Normalize data:
  - Don't normalize: Deaths, Confirmed
  - Normalize everything else
  - States [Don't need to worry about this]
  - Dates
    - Split in day and month
    - Normalize each of them between [0,1]
3. Augment the dataset:
  - State-specific moving averages (3 day, 7 day, 10 day, 14 day)
  - Get population data per state, and then create:
    - Deaths/100K
    - Confirmed/100K
  - Find a way to merge graph data with train data
  - Need to split entire dataset per state
4. Feature selection and exploration
  - Gain a better understanding of the features 
    - If two features are highly correlated, the model may actually perform better by just using one them.
   
# Evaluation Metric

1. MAPE as our evaluation metric.
  - We want our model to beat the baseline score for both death and confirmed.
  
# Model Selection

1. Train, test and validation split
  - 7800 rows of data: Split into April-July as train, and August as validation
    - For august, we want to use moving averages as features instead of the ground truth labels.
    - How much moving average deviates from the ground truth
2. Graph search:
  - Dictionary of Model names: a dictionary of hyperparameter choices
    - Eg. {"LinearRegression": {"Regularization": [0.1, 0.2, 0.3], .....}
          {"NeuralNetwork": {"LearningRate": [], "Learning Rate Decay": []...}
3. List of models:
  - Linear Regression
  - Polynomial Regression
  - Decision Trees
  - Random Forest
  - SGDRegresson
  - Multilayer Perceptron Regressor
