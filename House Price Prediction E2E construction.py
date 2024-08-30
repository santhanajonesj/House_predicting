# %%
from pathlib import Path
import re

import numpy as np 
from scipy import stats
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm.auto as tqdm

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# %%
df = pd.read_csv('Housing.csv')

# %%
# Initially declaring the target feature, as it will be utilised further down the line.
target = 'MEDV'

# %%
# There are multiple features that have a high distriubtion from the median, signaling a high right/left skew.
# This will be noted as unskew strategies will be applied later on.
df.describe()

# %%
# CHAS Feature is binary, however, it is not indicated as such in the original dataset
df.dtypes

# %%
# There are multiple feature that are missing data, as, fortunately, 
# all the features are numerical a median fill approach can be utilized.
df.info()

# %%
df.nunique()

# %%
def plot_hist(df):
    # Selecting numerical features, this is not required as all features
    # are numerical
    cols = (
        df
        .select_dtypes(include=[int, float])
        .columns
    )

    ncols = 2
    # Determining how many rows will be required based on the number of features
    # that are available; this process is agile and won't require modifications
    # if new features will be added
    nrows = np.ceil(len(cols) / ncols).astype(int)
    # The more features there are available the larger the figure will become
    vertical_figsize = 2 * nrows

    axs = plt.subplots(nrows, ncols, figsize=[10, vertical_figsize])[1].flatten()

    for col, ax in zip(cols, axs):
        df[col].plot.hist(title=col, ax=ax)
    plt.tight_layout()
    plt.show()

# %%
plot_hist(df)

# %%
#Processing
# As previously metnioned, NaN values will be substituted with median
df = df.fillna(df.median(axis=0))

# %%
df['CHAS'] = df['CHAS'].astype(bool)

# %%
#Feature Construction
df['CRIM_ZN'] = df['CRIM'] * df['ZN']
df['INDUS_CHAS'] = df['INDUS'] * df['CHAS']
df['NOX_DIS'] = df['NOX'] * df['DIS']
df['RM_AGE'] = df['RM'] * df['AGE']
df['RAD_TAX'] = df['RAD'] * df['TAX']
df['PTRATIO_B'] = df['PTRATIO'] * df['B']

# %%
#Unskewing data
skew_res = df.select_dtypes([int, float]).skew().abs().sort_values(ascending=False)
skew_cols = skew_res.loc[lambda x: (x>=1) & (x.index!=target)].index

print(skew_res)
print('-'*50)
print('Cols that are skewed:')
print(', '.join(skew_cols))

# %%
def best_transformation(data) -> tuple:
    functions = [np.log1p, np.sqrt, stats.yeojohnson]
    results = []
    
    for func in functions:
        transformed_data = func(data)
        if type(transformed_data) == tuple:
            vals, _ = transformed_data
            results.append(vals)
        else:
            results.append(transformed_data)
            
    abs_skew_results = [np.abs(stats.skew(val)) for val in results]
    lowest_skew_index = abs_skew_results.index(min(abs_skew_results))
    return functions[lowest_skew_index], results[lowest_skew_index]


# %%
def unskew(col):
    global best_transformation
    print('-'*100)
    col_skew = stats.skew(col)
    col_name = col.name
    print('{} skew is: {}'.format(col_name, col_skew))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[10, 4])
    col.plot.hist(color='red', alpha=.4, label='pre-skew', ax=ax1)

    if np.abs(col_skew) >= 1.:
        result_skew, data = best_transformation(col)
        new_col_skew = stats.skew(data)
        print(f'Best function {result_skew} and the skew results: {new_col_skew}')
        ax2.hist(data, label='post skew processing')
        plt.legend()
        plt.show()
        
        if np.abs(new_col_skew) >= 1.:
            print(f'Transformation was not successful for {col_name}, returning original data')
            return col 
        
        return data 
    
    plt.show()
    
df[skew_cols] = df[skew_cols].apply(unskew, axis=0);

# %%
#Feature Selection
#Selecting highest correlated data with the target

corr_ranking = (
    df
    .drop(target, axis=1)
    .corrwith(df[target])
    .abs()
    .sort_values(ascending=False)
)

# %%
_, ax = plt.subplots(figsize=[10, 10])
sns.heatmap(df.corr(), annot=True, cmap="rocket", cbar=False, ax=ax);

# %%
# Taking all the values above 0.3

threshold = .3
chosen_cols = corr_ranking[corr_ranking>=threshold]
print(chosen_cols)
chosen_cols = chosen_cols.index.to_list()

# %%
#Data Preparation
#Train test split

X = df[chosen_cols]
y = df[target]

# %%
X.shape, y.shape

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

# %%
#Scaling
X_train.dtypes

# %%
# Since CHAS did not pass correlation filtration there is no need to remove it
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

cols = X_train.select_dtypes([float, int]).columns.to_list()

X_train[cols] = scaler.fit_transform(X_train)
X_test[cols] = scaler.transform(X_test)

# %%
#Model Construction
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

# %%
n_features = X_train.shape[1]

# %%
random_forest_params = {
    'n_estimators': np.sort(np.random.default_rng().choice(500, size=10, replace=False)),
    'max_features': np.sort(np.random.default_rng().choice(n_features, size=5, replace=False)),
    'max_depth': [1, 5, 10],
}

# %%
xgb_params = {
    'objective': ['reg:squarederror'],
    'max_depth': [2, 5,],
    'min_child_weight': np.arange(1, 5, 2),
    'n_estimators': np.sort(np.random.default_rng().choice(500, size=3, replace=False)),
    'learning_rate': [1e-1, 1e-2,],
    'gamma': np.sort(np.random.default_rng().choice(20, size=3, replace=False)),
    'reg_lambda': [0, 1.0, 10.0],
    'scale_pos_weight': [1, 3, 5],
    'n_jobs': [-1],
}

# %%
best_mode_params = {
    LinearRegression(): {'fit_intercept': True},
    RandomForestRegressor(): {'max_depth': 10, 'max_features': 9, 'n_estimators': 378},
    XGBRegressor(): {'gamma': 18, 'learning_rate': 0.1, 'max_depth': 2, 'min_child_weight': 3, 'n_estimators': 461, 'n_jobs': -1, 'objective': 'reg:squarederror', 'reg_lambda': 0, 'scale_pos_weight': 1},
}

# %%
from sklearn.metrics import mean_squared_error, r2_score
b_models = []
model_results = []

for model in best_mode_params.keys():
    params = best_mode_params[model]
    model.set_params(**params)
    model.fit(X_train, y_train)    
    b_models.append(model)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    model_name = re.search(r'\w+', str(model))[0]
    results = pd.Series({'MSE': mse, 'RMSE': rmse, 'R2': r2}, name=model_name)
    model_results.append(results)

# %%
#Evaluation
pd.concat(model_results, axis=1)

# %%
#As visible from above, XGBoost outperformed RandomForest.

feature_imp = []
for model in b_models:
    try:
        model_name = re.search(r'\w+', str(model))[0]
        feature_imp.append(
            pd.Series(
                {
                    col: importance 
                    for col, importance in zip(cols, model.feature_importances_)
                },
                name = model_name
            )
        )
    except AttributeError:
        pass
    
pd.concat(feature_imp, axis=1).sort_values(by='XGBRegressor', ascending=False)


# %%
#Actual data v/s prediction data
xgb_model = b_models[2]
col = 'LSTAT'
y_pred = xgb_model.predict(X_test.sort_values(by=col))

(
    pd.concat([X_test[col], y_test], axis=1)
    .sort_values(by=col)
    .plot(x=col, y='MEDV', color='red', alpha=.4, label='Actual Data'),
)

plt.scatter(X_test[col].sort_values(), y_pred, label='Prediction Data')
plt.legend()
plt.show()

# %%
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

# %%
#Evaluation Pipeline¶
#Evaluation pipeline to automate the process of training and evaluation, instead of training and evaluating for every model

class Evaluation:
    def __init__(self, train, test):
        self.train = train
        self.test = test
        
    def evaluate(self, model, name):
        x, y = self.train
        y_pred = model.predict(x)
        mae = mean_absolute_error(y_pred, y)
        mse = mean_squared_error(y_pred, y)
        r2 = r2_score(y_pred, y)
        print(name, "\n", "-"*20)
        print("MAE: {}\nMSE: {}\nr2: {}".format(mae, mse, r2))
        
    def training(self, model, name):
        x, y = self.train
        model.fit(x, y)
        self.evaluate(model, name)
        return model

# %%
#Defining models, tuning their hyperparameters¶
lnr = LinearRegression()
rfr = RandomForestRegressor(n_estimators=150, max_depth=115, criterion='friedman_mse',
                           max_features='log2')
dtr = DecisionTreeRegressor(max_depth=110,criterion='friedman_mse')
svr = SVR(C=0.7)
abr = AdaBoostRegressor(n_estimators=50, learning_rate=0.5)
xgb = XGBRegressor(n_estimators=1000, max_depth=11, eta=0.31)

models = [lnr, rfr, dtr, svr, abr, xgb]
names = ['Linear Regression', 'Random Forest Regressor',
        'Decision Tree Regressor', 'SVR',
        'Ada Boost Regressor', 'XGBRegressor']

assesment = Evaluation((X_train, y_train), (X_test, y_test))

# %%
assesment_selected = Evaluation((X_train, y_train), (X_test, y_test))
selected_trained = []
for i, j in zip(models, names):
    selected_trained += [assesment_selected.training(i, j)]

# %%
#Feature Selection
corr_matrix = df.corr(method='pearson')
sns.heatmap(corr_matrix, annot=True)

# %%
#Threshold
selected = []
for i in corr_matrix.index[:-1]:
    if corr_matrix.loc[i, "MEDV"] > 0.4 or corr_matrix.loc[i, "MEDV"] < -0.4:
        selected += [i]

# %%
x_s = df.loc[:, selected].values
y_s = df.loc[:, 'MEDV'].values
X_train, X_test, ys_train, ys_test = train_test_split(x_s, y_s, random_state=42, test_size=0.2)

# %%
assesment_selected = Evaluation((X_train, y_train), (X_test, ys_test))
selected_trained = []
for i, j in zip(models, names):
    selected_trained += [assesment_selected.training(i, j)]


