import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import statsmodels.api as sm
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

import warnings
warnings.filterwarnings("ignore")
from ml_feature_engg import X_train, X_test, y_train, y_test, selected_col



def backward_stepwise_elimination_with_summary_pval(X_train, y_train):
    table = pd.DataFrame(columns=["AIC", "BIC", "Adj. R2", "P-Value"])
    p_val_table = pd.DataFrame(columns=["P-Value"])
    f_stat_table = pd.DataFrame(columns=["F-Statistic"])
    adj_r2_table = pd.DataFrame(columns=["Adj. R2"])
    removed_features = []
    counter = 0
    model = sm.OLS(y_train, X_train).fit()
    print(model.summary())

    while model.pvalues.max() >= 0.01:
        table.loc[model.pvalues.idxmax(), "AIC"] = model.aic
        table.loc[model.pvalues.idxmax(), "BIC"] = model.bic
        table.loc[model.pvalues.idxmax(), "Adj. R2"] = model.rsquared_adj
        table.loc[model.pvalues.idxmax(), "P-Value"] = model.pvalues.max()
        p_val_table.loc[model.pvalues.idxmax(), "P-Value"] = model.pvalues.max()
        f_stat_table.loc[model.pvalues.idxmax(), "F-Statistic"] = model.fvalue
        adj_r2_table.loc[model.pvalues.idxmax(), "Adj. R2"] = model.rsquared_adj
        X_train.drop(model.pvalues.idxmax(), axis=1, inplace=True)
        X_test.drop(model.pvalues.idxmax(), axis=1, inplace=True)
        print("Removed Feature:", model.pvalues.idxmax())
        removed_features.append(model.pvalues.idxmax())
        counter += 1
        print("No. of Features Removed:", counter)
        model = sm.OLS(y_train, X_train).fit()
        print(model.summary())

    print("Removed Features:", removed_features)
    return table, p_val_table, f_stat_table, adj_r2_table


table, t_stat, f_stat, adj_r2  = backward_stepwise_elimination_with_summary_pval(X_train, y_train)
print(table)
print(t_stat)
print(f_stat)
print(adj_r2)

# Fit the model with the selected features
lin_reg_model = sm.OLS(y_train, X_train).fit()
print(lin_reg_model.summary())

confidence_intervals = lin_reg_model.conf_int()
print("Confidence Intervals:\n", confidence_intervals)


def backward_stepwise_summary_graph(table):
    fig, ax = plt.subplots(1, 3, figsize=(30, 15))

    # Define the tick locations
    tick_locations = range(0, len(table.index), 15)

    # Plot AIC
    ax[0].plot(table["AIC"], marker="o")
    ax[0].set_title("AIC")
    ax[0].set_xlabel("Step")
    ax[0].set_xticks(tick_locations)
    ax[0].set_xticklabels([table.index[i] for i in tick_locations], rotation=45)

    # Plot BIC
    ax[1].plot(table["BIC"], marker="o")
    ax[1].set_title("BIC")
    ax[1].set_xlabel("Step")
    ax[1].set_xticks(tick_locations)
    ax[1].set_xticklabels([table.index[i] for i in tick_locations], rotation=45)

    # Plot Adjusted R2
    ax[2].plot(table["Adj. R2"], marker="o")
    ax[2].set_title("Adjusted R2")
    ax[2].set_xlabel("Step")
    ax[2].set_xticks(tick_locations)
    ax[2].set_xticklabels([table.index[i] for i in tick_locations], rotation=45)

    for axis in ax:
        axis.tick_params(axis='x', which='major', labelsize=8)
        axis.grid(True)

    plt.tight_layout()
    plt.show()


backward_stepwise_summary_graph(table)

# Prediciton of dependent variable
y_pred = lin_reg_model.predict(X_test)

# Plot the predicted values against the actual values (diffent colors) with line of best fit
plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred, color='blue')
plt.plot(y_test, y_test, color='red')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

# Do predictions on the test set and evaluate the model
print(f"MSE:{mse(y_test, y_pred):.2f}")
print(f"MAE:{mae(y_test, y_pred):.2f}")
print(f"RMSE:{np.sqrt(mse(y_test, y_pred)): .2f}")
print(f"R2:{lin_reg_model.rsquared:.2f}")
print(f"Adjusted R2:{lin_reg_model.rsquared_adj:.2f}")