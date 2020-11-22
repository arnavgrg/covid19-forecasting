from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import ParameterGrid
import pandas as pd

def define_models_params():
    models = {
        'LR': LinearRegression(),
        'SGD': SGDRegressor(),
        'DT': DecisionTreeRegressor(),
        'RF': RandomForestRegressor(),
        'MLP': MLPRegressor()
    }
    grid = {
        'LR': {'C': [0.01, 0.1, 1, 10]},
        'SGD': {'C': [0.01, 0.1, 1, 10]},
        'DT': {'max_depth': [10, 20, 30, None]},
        'RF': {'n_estimators': [50, 100], 'max_depth': [10, 20, 30, None]},
        'MLP': {'hidden_layer_sizes': [(100,), (100,100), (100, 100, 100)]}
    }
    return models, grid

models, grid = define_models_params()

merged_df = pd.read_csv('./data/merged_transformed.csv')
confirmed = merged_df['Confirmed']
deaths = merged_df['Deaths']
X = merged_df.drop(columns=['Confirmed', 'Deaths'])

mask = X['Month'] < 0.5
X_train, X_val = X[mask], X[~mask]
confirmed_train, confirmed_val = confirmed[mask], confirmed[~mask]
deaths_train, deaths_val = deaths[mask], deaths[~mask]

chosen_model = None
score = 0
results = []

# for model in models:
#     param_values = grid[model]
#     for p in ParameterGrid(param_values):
#         model.set_params(**p)

#         # Test confirmed
#         model.fit(X_train, confirmed_train)
#         confirmed_pred = model.predict(X_val)
#         confirmed_score = MAPE(confirmed_val, confirmed_pred)

#         # Test deaths
#         model.fit(X_train, deaths_train)
#         deaths_pred = model.predict(X_val)
#         deaths_score = MAPE(deaths_val, deaths_pred)

#         # Evaluate model
#         # TODO: decide how to aggregate score
#         curr_score = (confirmed_score + deaths_score) / float(2)
#         if curr_score > score:
#             score = curr_score
#             chosen_model = model

#         # Save results
#         results.append([curr_score, model])

# results = sorted(results, reverse=True)
# for i in range(len(results)):
#     print(str(results[i][0]), ':', str(results[i][1]))
