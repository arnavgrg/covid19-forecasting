from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import ParameterGrid
import pandas as pd
import numpy as np

def percentage_error(actual, predicted):
    '''
        Helper method for MAPE. Returns percentage error between
        predicted values and ground truth values.
    '''
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res


def MAPE(y_true, y_pred):
    '''
        Inputs:
            - y_true: ground truth values
            - y_pred: predicted values
        Returns:
            - Mean Absolute Percentage Error (MAPE), a value that lies in the range [0,100]
    '''
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100

def define_models_params(verbose=True, random_state=0):
    '''
        Inputs:
            - verbose: decide to display
            - random_state: for reproducibility
        Returns:
            - models: dictionary of models
            - grid: dictionary of hyper-parameters
    '''
    models = {
        'LR': LinearRegression(),
        'SGD': SGDRegressor(),
        'DT': DecisionTreeRegressor(),
        'RF': RandomForestRegressor(random_state=random_state)
    }
    grid = {
        'LR': {},
        'SGD': {'alpha': [0.0001, 0.001, 0.01], 'penalty': ['l1', 'l2']},
        'DT': {'max_depth': [15, 20, 25]},
        'RF': {'max_depth': [15, 20, 25]},
    }
    return models, grid

def param_search():
    '''
    Partitions off August data as validation, then returns best models for Confirmed and Deaths
    '''
    models, grid = define_models_params()

    # Import data
    merged_df = pd.read_csv('./data/merged_transformed.csv')
    confirmed = merged_df['Confirmed']
    deaths = merged_df['Deaths']
    X = merged_df.drop(columns=['Confirmed', 'Deaths'])

    # Separate into training and validation
    mask = X['Month'] < 0.5
    X_train, X_val = X[mask], X[~mask]
    confirmed_train, confirmed_val = confirmed[mask], confirmed[~mask]
    deaths_train, deaths_val = deaths[mask], deaths[~mask]

    # Setup for iteration
    chosen_confirmed, chosen_deaths = None, None
    confirmed_score, deaths_score = 100, 100
    results = {}
    results['confirmed_models'] = []
    results['confirmed_scores'] = []
    results['deaths_models'] = []
    results['deaths_scores'] = []

    for idx in models:
        model = models[idx]
        param_values = grid[idx]
        for p in ParameterGrid(param_values):
            model.set_params(**p)
            print(model)

            # Test confirmed
            model.fit(X_train, confirmed_train)
            confirmed_pred = model.predict(X_val)
            curr_confirmed_score = MAPE(confirmed_val, confirmed_pred)
            if curr_confirmed_score < confirmed_score:
                confirmed_score = curr_confirmed_score
                chosen_confirmed = model
            results['confirmed_scores'].append(confirmed_score)
            results['confirmed_models'].append(model)

            # Test deaths
            model.fit(X_train, deaths_train)
            deaths_pred = model.predict(X_val)
            curr_deaths_score = MAPE(deaths_val, deaths_pred)
            if curr_deaths_score < deaths_score:
                deaths_score = curr_deaths_score
                chosen_deaths = model
            results['deaths_scores'].append(deaths_score)
            results['deaths_models'].append(model)

    print()
    print('BEST MODEL')
    print('CONFIRMED: ', chosen_confirmed, confirmed_score)
    print('DEATHS: ', chosen_deaths, deaths_score)
    return chosen_confirmed, chosen_deaths

def main():
    confirmed_model, deaths_model = param_search()

    # Get train data
    train_df = pd.read_csv('./data/merged_transformed.csv')
    confirmed = train_df['Confirmed']
    deaths = train_df['Deaths']
    X_train = train_df.drop(columns=['Confirmed', 'Deaths'])

    # Get test data
    test_df = pd.read_csv('./data/test_transformed.csv')
    X_test = test_df.drop(columns=['Confirmed', 'Deaths', 'ForecastID'])

    # Fit models and predict
    confirmed_model.fit(X_train, confirmed)
    confirmed_pred = confirmed_model.predict(X_test)
    deaths_model.fit(X_train, deaths)
    deaths_pred = deaths_model.predict(X_test)

    # Create submission file
    submission = pd.DataFrame({'Confirmed': confirmed_pred, 'Deaths': deaths_pred})
    submission.index.name = 'ForecastID'

    # Rename csv
    submission.to_csv('./data/Submissions/first_grid_search.csv')

if __name__ == "__main__":
    main()