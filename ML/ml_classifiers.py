# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import RocCurveDisplay, accuracy_score, recall_score, roc_curve, roc_auc_score, confusion_matrix, \
    ConfusionMatrixDisplay, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, StackingClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")
from ml_feature_engg import X_train, X_test, y_train, y_test, selected_col, X, y
X = X[selected_col]
X_train = X_train[selected_col]
X_test = X_test[selected_col]
np.set_printoptions(precision=2)
def getMetricsTable(metrics):
    Table = pd.DataFrame(columns=['Model', 'Precision', 'Recal', 'Specificity', 'F-score', 'Confusion Matrix'])
    for model in metrics:
        Table.loc[len(Table)] = model
    x = PrettyTable()
    x.title = f"Classifier Comparison"
    x.field_names = Table.columns

    for index, row in Table.iterrows():
        x.add_row(row)

    print(x)
def getStratifiedTable(metrics):
    Table = pd.DataFrame(columns=['Model', 'Accuracies', 'Max Accuracy', 'Min Accuracy', 'Avg Accuracy'])
    for model in metrics:
        Table.loc[len(Table)] = model
    x = PrettyTable()
    x.title = f"Stratified K-fold Comparison"
    x.field_names = Table.columns

    for index, row in Table.iterrows():
        x.add_row(row)

    print(x)

Metrics = []
Stratified = []



# ========================================
# Decision Tree
# ========================================
dt_model = DecisionTreeClassifier(random_state=5805)
dt_model.fit(X_train, y_train)

# %%
y_pred_DF = dt_model.predict(X_test)
y_proba_DF = dt_model.predict_proba(X_test)[::, 1]
accuracy = accuracy_score(y_test, y_pred_DF)
print("Accuracy of Basic Tree:", round(accuracy, 2))

# %%
Metrics.insert(0,['Basic Decision Tree',
                        precision_score(y_test, y_pred_DF),
                        recall_score(y_test, y_pred_DF),
                        recall_score(y_test, y_pred_DF, pos_label=0),
                        f1_score(y_test, y_pred_DF),
                        confusion_matrix(y_test, y_pred_DF)])

# %%
getMetricsTable(Metrics)

# %%
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5805)
stratified_scores_dt = []
dt_model = DecisionTreeClassifier(random_state=5805)

for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    dt_model.fit(X_train_fold, y_train_fold)
    stratified_scores_dt.append(dt_model.score(X_test_fold, y_test_fold))

Stratified.insert(0,['Basic Decision Tree',
                        stratified_scores_dt,
                        max(stratified_scores_dt)*100,
                        min(stratified_scores_dt)*100,
                        (sum(stratified_scores_dt)/len(stratified_scores_dt))*100,
                        ])

getStratifiedTable(Stratified)

# %%

important_features_dt = dt_model.feature_importances_
feature_labels = X_train.columns
sorted_indices = (-important_features_dt).argsort()
sorted_feature_labels = feature_labels[sorted_indices]
print("Important features:", sorted_feature_labels)

# %%
labels = ['Feature Importance']
plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_feature_labels)), important_features_dt[sorted_indices], align="center", label=labels[0])
plt.yticks(range(len(sorted_feature_labels)), sorted_feature_labels)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Decision Tree Feature Importance")
plt.grid(True)
plt.show()

# ========================================
# PrePrunning
# ========================================
param_grid_dt = {
    'max_depth': [2, 3, 5, 7, 11],
    'min_samples_split': [2, 3, 5, 7, 11],
    'min_samples_leaf': [2, 3, 5, 7, 11],
    'max_features': ['sqrt', 'log2'],
    'splitter': ['best', 'random'],
    'criterion': ['gini', 'entropy', 'log_loss']
}
clf = GridSearchCV(dt_model, param_grid_dt, cv=5)
clf.fit(X_train, y_train)
print("BEST PARAMS FROM GRID SEARCH", clf.best_params_)


preTre = DecisionTreeClassifier(criterion=clf.best_params_['criterion'],
                                        max_depth=clf.best_params_['max_depth'], splitter=clf.best_params_['splitter'],
                                        max_features=clf.best_params_['max_features'],
                                        min_samples_split=clf.best_params_['min_samples_split'],
                                        min_samples_leaf=clf.best_params_['min_samples_leaf'], random_state=5805)
preTre.fit(X_train, y_train)

# %%
y_pred_pre = preTre.predict(X_test)
y_proba_pre = preTre.predict_proba(X_test)[::, 1]
accuracy_pre = accuracy_score(y_test, y_pred_pre)
recall_pre = recall_score(y_test, y_pred_pre)
print("Accuracy after Pre Prunning:", round(accuracy_pre, 2))

# %%
fpr_pre, tpr_pre, thresholds_pre = roc_curve(y_test, y_proba_pre)
auc_pre = roc_auc_score(y_test, y_proba_pre)

plt.plot(fpr_pre, tpr_pre, label='ROC, auc')
plt.fill_between(fpr_pre, tpr_pre, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc_pre, ha='right', fontsize=12, weight='bold', color='blue')
plt.title("ROC Curve for Pre Prunned Tree")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# %%
Metrics.insert(0,['Pre Prunning Decision Tree',
                        precision_score(y_test, y_pred_pre),
                        recall_score(y_test, y_pred_pre),
                        recall_score(y_test, y_pred_pre, pos_label=0),
                        f1_score(y_test, y_pred_pre),
                        confusion_matrix(y_test, y_pred_pre)])

# %%
getMetricsTable(Metrics)

# %%
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5805)
stratified_scores = []
dt_model = preTre

for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    dt_model.fit(X_train_fold, y_train_fold)
    stratified_scores.append(dt_model.score(X_test_fold, y_test_fold))

Stratified.insert(0,['Pre Prunned Decision Tree',
                        stratified_scores,
                        max(stratified_scores)*100,
                        min(stratified_scores)*100,
                        (sum(stratified_scores)/len(stratified_scores_dt))*100,
                        ])

getStratifiedTable(Stratified)


plot_tree(preTre,rounded=True,filled=True)
plt.show()

# ========================================
# Postprunning
# ========================================
path = dt_model.cost_complexity_pruning_path(X_train, y_train)
alphas = path['ccp_alphas']
print(len(alphas))
accuracy_train, accuracy_test = [], []

for alpha in alphas[:100]:
    alphaTree = DecisionTreeClassifier(ccp_alpha=alpha, random_state=5805)
    alphaTree.fit(X_train, y_train)

    y_train_pred = alphaTree.predict(X_train)
    y_test_pred = alphaTree.predict(X_test)

    accuracy_train.append(accuracy_score(y_train, y_train_pred))
    accuracy_test.append(accuracy_score(y_test, y_test_pred))

max_alpha = alphas[accuracy_test.index(max(accuracy_test))]

# %%
max_alpha

# %%
plt.plot(alphas[:100], accuracy_train, label='Train Accuracy', drawstyle='steps-post', marker='o')
plt.plot(alphas[:100], accuracy_test, label='Test Accuracy', drawstyle='steps-post', marker='o')
plt.grid()
# max_value = max(plt.yticks())
plt.annotate("Maximum", (3, max(accuracy_test)), xycoords="data")
plt.show()
print("Optimum Alpha", round(max_alpha, 2))

# %%
postTree = DecisionTreeClassifier(ccp_alpha=max_alpha, random_state=5805)
postTree.fit(X_train, y_train)

# %%
y_pred_post = postTree.predict(X_test)
y_proba_post = postTree.predict_proba(X_test)[::, 1]

accuracy_post = accuracy_score(y_test, y_pred_post)
recall_post = recall_score(y_test, y_pred_post)
print("Accuracy after post prunning:", round(accuracy_post, 2))

# %%
plot_tree(postTree,rounded=True,filled=True)
plt.show()

# %%
Metrics.insert(0,['Post Prunning Decision Tree',
                        precision_score(y_test, y_pred_post),
                        recall_score(y_test, y_pred_post),
                        recall_score(y_test, y_pred_post, pos_label=0),
                        f1_score(y_test, y_pred_post),
                        confusion_matrix(y_test, y_pred_post)])

# %%
getMetricsTable(Metrics)

# %%
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5805)
stratified_scores = []
dt_model = postTree

for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    dt_model.fit(X_train_fold, y_train_fold)
    stratified_scores.append(dt_model.score(X_test_fold, y_test_fold))

Stratified.insert(0,['Post Prunned Decision Tree',
                        stratified_scores,
                        max(stratified_scores)*100,
                        min(stratified_scores)*100,
                        (sum(stratified_scores)/len(stratified_scores_dt))*100,
                        ])

getStratifiedTable(Stratified)

# %%
fpr_postprun, tpr_postprun, thresholds_postprun = roc_curve(y_test, y_proba_post)
auc_post = roc_auc_score(y_test, y_proba_post)

plt.plot(fpr_postprun, tpr_postprun, label='ROC, auc')
plt.fill_between(fpr_postprun, tpr_postprun, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc_post, ha='right', fontsize=12, weight='bold', color='blue')
plt.title("ROC Curve for Post Prunned Tree")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# # ========================================
# # KNN
# # ========================================
X_train_knn = np.ascontiguousarray(X_train)
X_test_knn = np.ascontiguousarray(X_test)

knn = KNeighborsClassifier()
knn.fit(X_train_knn, y_train)

y_pred_knn = knn.predict(X_test_knn)
scores = cross_val_score(knn, X_train_knn, y_train, cv=5, scoring='accuracy')
#print('Cross Validation Accuracy Scores:', scores)

fpr_knn, tpr_knn, _ = roc_curve(y_test,y_pred_knn)
auc_knn = roc_auc_score(y_test,y_pred_knn)

plt.plot(fpr_knn, tpr_knn, label='ROC, auc')
plt.fill_between(fpr_knn, tpr_knn, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc_knn, ha='right', fontsize=12, weight='bold', color='blue')
plt.title("ROC Curve for Pre Prunned Tree")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()


Metrics.insert(0,['KNN',
                        precision_score(y_test, y_pred_knn),
                        recall_score(y_test, y_pred_knn),
                        recall_score(y_test, y_pred_knn, pos_label=0),
                        f1_score(y_test, y_pred_knn),
                        confusion_matrix(y_test, y_pred_knn)])

# %%
getMetricsTable(Metrics)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5805)
stratified_scores_dt = []
dt_model = knn

for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    dt_model.fit(X_train_fold, y_train_fold)
    stratified_scores_dt.append(dt_model.score(X_test_fold, y_test_fold))

Stratified.insert(0,['KNN',
                        stratified_scores_dt,
                        max(stratified_scores_dt)*100,
                        min(stratified_scores_dt)*100,
                        (sum(stratified_scores_dt)/len(stratified_scores_dt))*100,
                        ])

getStratifiedTable(Stratified)

# # ========================================
# # KNN Grid Search
# # ========================================
knn = KNeighborsClassifier()
k_range = list(range(1, 15))
param_grid = dict(n_neighbors=k_range)

clf = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=True, verbose=1, n_jobs=-1)
clf.fit(X_train_knn, y_train)

print("Best Estimator: \n", clf.best_estimator_)
print("Best Score: \n", clf.best_score_)
print("Best Paramters: \n", clf.best_params_)

best_knn = clf.best_estimator_
best_knn.fit(X_train_knn, y_train)

y_pred_knn_gs = best_knn.predict(X_test_knn)

scores = cross_val_score(best_knn, X_train_knn, y_train, cv=5, scoring='accuracy')
#print('Cross Validation Accuracy Scores:', scores)

fpr_knn_gs, tpr_knn_gs, _ = roc_curve(y_test,y_pred_knn_gs)
auc_knn_gs = roc_auc_score(y_test,y_pred_knn_gs)

plt.plot(fpr_knn_gs, tpr_knn_gs, label='ROC, auc')
plt.fill_between(fpr_knn_gs, tpr_knn_gs, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc_knn_gs, ha='right', fontsize=12, weight='bold', color='blue')
plt.title("ROC Curve for Pre Prunned Tree")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

Metrics.insert(0,['KNN Grid Search',
                        precision_score(y_test, y_pred_knn_gs),
                        recall_score(y_test, y_pred_knn_gs),
                        recall_score(y_test, y_pred_knn_gs, pos_label=0),
                        f1_score(y_test, y_pred_knn_gs),
                        confusion_matrix(y_test, y_pred_knn_gs)])

# %%
getMetricsTable(Metrics)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5805)
stratified_scores_dt = []
dt_model = best_knn

for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    dt_model.fit(X_train_fold, y_train_fold)
    stratified_scores_dt.append(dt_model.score(X_test_fold, y_test_fold))

Stratified.insert(0,['KNN grid search',
                        stratified_scores_dt,
                        max(stratified_scores_dt)*100,
                        min(stratified_scores_dt)*100,
                        (sum(stratified_scores_dt)/len(stratified_scores_dt))*100,
                        ])

getStratifiedTable(Stratified)

# ========================================
# Logistic Regression
# ========================================
logisticModel = LogisticRegression(random_state=5805, max_iter=1000)
logisticModel.fit(X_train, y_train)

# %%
y_pred_log = logisticModel.predict(X_test)
y_proba_log = logisticModel.predict_proba(X_test)[::, 1]

accuracy_log = accuracy_score(y_test, y_pred_log)
recall_log = recall_score(y_test, y_pred_log)
print("Accuracy Logistic Regression:", round(accuracy_log, 2))

# %%
Metrics.insert(0,['Logistic Regression',
                        precision_score(y_test, y_pred_log),
                        recall_score(y_test, y_pred_log),
                        recall_score(y_test, y_pred_log, pos_label=0),
                        f1_score(y_test, y_pred_log),
                        confusion_matrix(y_test, y_pred_log)])

# %%
getMetricsTable(Metrics)

# %%
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5805)
stratified_scores = []
model = logisticModel

for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train_fold, y_train_fold)
    stratified_scores.append(model.score(X_test_fold, y_test_fold))

Stratified.insert(0,['Logistics Regression',
                        stratified_scores,
                        max(stratified_scores)*100,
                        min(stratified_scores)*100,
                        (sum(stratified_scores)/len(stratified_scores_dt))*100,
                        ])

getStratifiedTable(Stratified)

# %%
fpr_log, tpr_log, thresholds_log = roc_curve(y_test, y_proba_log)
auc_log = roc_auc_score(y_test, y_proba_log)

plt.plot(fpr_log, tpr_log, label='ROC, auc')
plt.fill_between(fpr_log, tpr_log, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc_log, ha='right', fontsize=12, weight='bold', color='blue')
plt.title("ROC Curve for Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# ========================================
# Logistic Regression GridSearch
# ========================================
param_grid_log = {
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],
    'penalty': ['none', 'l1', 'l2', 'elasticnet'],
    'C': np.logspace(-3, 3, 7)
}
search = GridSearchCV(LogisticRegression(), param_grid=param_grid_log, scoring='accuracy', n_jobs=-1, cv=5)

result = search.fit(X_train, y_train)
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# %%
logisticModel_grid = LogisticRegression(C=result.best_params_['C'], penalty=result.best_params_['penalty'],
                                        solver=result.best_params_['solver'], random_state=5805, max_iter=1000)
logisticModel_grid.fit(X_train, y_train)

# %%
y_pred_log_grid = logisticModel_grid.predict(X_test)
y_proba_log_grid = logisticModel_grid.predict_proba(X_test)[::, 1]

accuracy_log_grid = accuracy_score(y_test, y_pred_log_grid)
recall_log_grid = recall_score(y_test, y_pred_log_grid)
print("Accuracy Logistic Regression:", round(accuracy_log_grid, 2))

# %%
Metrics.insert(0,['Logistic Regression GridSearch',
                        precision_score(y_test, y_pred_log_grid),
                        recall_score(y_test, y_pred_log_grid),
                        recall_score(y_test, y_pred_log_grid, pos_label=0),
                        f1_score(y_test, y_pred_log_grid),
                        confusion_matrix(y_test, y_pred_log_grid)])

# %%
getMetricsTable(Metrics)

# %%
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5805)
stratified_scores = []
model = logisticModel_grid

for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train_fold, y_train_fold)
    stratified_scores.append(model.score(X_test_fold, y_test_fold))

Stratified.insert(0,['Logistic Regression GridSearch',
                        stratified_scores,
                        max(stratified_scores)*100,
                        min(stratified_scores)*100,
                        (sum(stratified_scores)/len(stratified_scores_dt))*100,
                        ])

getStratifiedTable(Stratified)

# %%
fpr_log_grid, tpr_log_grid, thresholds_log_grid = roc_curve(y_test, y_proba_log_grid)
auc_log_grid = roc_auc_score(y_test, y_proba_log_grid)

plt.plot(fpr_log_grid, tpr_log_grid, label='ROC, auc')
plt.fill_between(fpr_log_grid, tpr_log_grid, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc_log_grid, ha='right', fontsize=12, weight='bold', color='blue')
plt.title("ROC Curve for Logistic Regression Grid Search")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# ========================================
# Naive Bayes
# ========================================
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

# %%
y_pred_NB = naive_bayes.predict(X_test)
y_proba_NB = naive_bayes.predict_proba(X_test)[::, 1]
print("Naive Bayes score: ", naive_bayes.score(X_test, y_test))

# %%
Metrics.insert(0,['Naive Bayes',
                        precision_score(y_test, y_pred_NB),
                        recall_score(y_test, y_pred_NB),
                        recall_score(y_test, y_pred_NB, pos_label=0),
                        f1_score(y_test, y_pred_NB),
                        confusion_matrix(y_test, y_pred_NB)])

# %%
getMetricsTable(Metrics)

# %%
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5805)
stratified_scores = []
model = naive_bayes

for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train_fold, y_train_fold)
    stratified_scores.append(model.score(X_test_fold, y_test_fold))

Stratified.insert(0,['Naive Bayes',
                        stratified_scores,
                        max(stratified_scores)*100,
                        min(stratified_scores)*100,
                        (sum(stratified_scores)/len(stratified_scores_dt))*100,
                        ])

getStratifiedTable(Stratified)

# %%
fpr_NB, tpr_NB, thresholds_NB = roc_curve(y_test, y_proba_NB)
auc_NB = roc_auc_score(y_test, y_proba_NB)

plt.plot(fpr_NB, tpr_NB, label='ROC, auc')
plt.fill_between(fpr_NB, tpr_NB, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc_NB, ha='right', fontsize=12, weight='bold', color='blue')
plt.title("ROC Curve for Naive Bayes")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# ========================================
# Naive Bayes Grid Search
# ========================================
param_grid_naive_bayes = {'var_smoothing': np.logspace(0, -9, num=100)}
gs_naive_bayes = GridSearchCV(estimator=GaussianNB(),
                     param_grid=param_grid_naive_bayes,
                     cv=5)
gs_naive_bayes.fit(X_train, y_train)

# %%
y_pred_NB_grid = gs_naive_bayes.predict(X_test)
y_proba_NB_grid = gs_naive_bayes.predict_proba(X_test)[::, 1]
print("Naive Bayes score Grid Search: ", gs_naive_bayes.score(X_test, y_test))

# %%
Metrics.insert(0,['Naive Bayes Grid Search',
                        precision_score(y_test, y_pred_NB_grid),
                        recall_score(y_test, y_pred_NB_grid),
                        recall_score(y_test, y_pred_NB_grid, pos_label=0),
                        f1_score(y_test, y_pred_NB_grid),
                        confusion_matrix(y_test, y_pred_NB_grid)])

# %%
getMetricsTable(Metrics)

# %%
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5805)
stratified_scores = []
model = gs_naive_bayes

for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train_fold, y_train_fold)
    stratified_scores.append(model.score(X_test_fold, y_test_fold))

Stratified.insert(0,['Naive Bayes GridSearch',
                        stratified_scores,
                        max(stratified_scores)*100,
                        min(stratified_scores)*100,
                        (sum(stratified_scores)/len(stratified_scores_dt))*100,
                        ])

getStratifiedTable(Stratified)

# %%
fpr_NB_grid, tpr_NB_grid, thresholds_NB_grid = roc_curve(y_test, y_proba_NB_grid)
auc_NB_grid = roc_auc_score(y_test, y_proba_NB_grid)

plt.plot(fpr_NB_grid, tpr_NB_grid, label='ROC, auc')
plt.fill_between(fpr_NB_grid, tpr_NB_grid, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc_NB_grid, ha='right', fontsize=12, weight='bold', color='blue')
plt.title("ROC Curve for Naive Bayes Grid Search")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# ========================================
# Random Forest
# ========================================
rf_model = RandomForestClassifier(random_state=5805)
rf_model.fit(X_train, y_train)

# %%
y_pred_RF = rf_model.predict(X_test)
y_proba_RF = rf_model.predict_proba(X_test)[::, 1]

accuracy_log = accuracy_score(y_test, y_pred_RF)
recall_log = recall_score(y_test, y_pred_RF)
print("Accuracy Random Forest:", round(accuracy_log, 2))

# %%
Metrics.insert(0,['Random Forest',
                        precision_score(y_test, y_pred_RF),
                        recall_score(y_test, y_pred_RF),
                        recall_score(y_test, y_pred_RF, pos_label=0),
                        f1_score(y_test, y_pred_RF),
                        confusion_matrix(y_test, y_pred_RF)])

# %%
getMetricsTable(Metrics)

# %%
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5805)
stratified_scores = []
model = rf_model

for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train_fold, y_train_fold)
    stratified_scores.append(model.score(X_test_fold, y_test_fold))

Stratified.insert(0,['Random Forest',
                        stratified_scores,
                        max(stratified_scores)*100,
                        min(stratified_scores)*100,
                        (sum(stratified_scores)/len(stratified_scores_dt))*100,
                        ])

getStratifiedTable(Stratified)

# %%
fpr_RF, tpr_RF, thresholds_RF = roc_curve(y_test, y_proba_RF)
auc_RF = roc_auc_score(y_test, y_proba_RF)

plt.plot(fpr_RF, tpr_RF, label='ROC, auc')
plt.fill_between(fpr_RF, tpr_RF, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc_RF, ha='right', fontsize=12, weight='bold', color='blue')
plt.title("ROC Curve for Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# ========================================
# Random Forest Grid Search
# ========================================
param_grid_rf = {
    'max_depth': [5, 10, 20],
    'min_samples_leaf': [5, 10, 20],
    'n_estimators': [10, 25, 30, ]
}
grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(),
                              param_grid=param_grid_rf,
                              cv=5)
grid_search_rf.fit(X_train, y_train)

# %%
y_pred_RF_grid = grid_search_rf.predict(X_test)
y_proba_RF_grid = grid_search_rf.predict_proba(X_test)[::, 1]
print("Random Forest Grid Search score: ", grid_search_rf.score(X_test, y_test))

# %%
Metrics.insert(0,['Random Forest Grid Search',
                        precision_score(y_test, y_pred_RF_grid),
                        recall_score(y_test, y_pred_RF_grid),
                        recall_score(y_test, y_pred_RF_grid, pos_label=0),
                        f1_score(y_test, y_pred_RF_grid),
                        confusion_matrix(y_test, y_pred_RF_grid)])

# %%
getMetricsTable(Metrics)

# %%
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5805)
stratified_scores = []
model = grid_search_rf

for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train_fold, y_train_fold)
    stratified_scores.append(model.score(X_test_fold, y_test_fold))

Stratified.insert(0,['Random Forest GridSearch',
                        stratified_scores,
                        max(stratified_scores)*100,
                        min(stratified_scores)*100,
                        (sum(stratified_scores)/len(stratified_scores_dt))*100,
                        ])

getStratifiedTable(Stratified)

# %%
fpr_RF_grid, tpr_RF_grid, thresholds_RF_grid = roc_curve(y_test, y_proba_RF_grid)
auc_RF_grid = roc_auc_score(y_test, y_proba_RF_grid)

plt.plot(fpr_RF_grid, tpr_RF_grid, label='ROC, auc')
plt.fill_between(fpr_RF_grid, tpr_RF_grid, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc_RF_grid, ha='right', fontsize=12, weight='bold', color='blue')
plt.title("ROC Curve for Random Forest Grid Search")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# ========================================
# Random Forest Bagging
# ========================================
bagging = BaggingClassifier()
bagging.fit(X_train, y_train)

# %%
y_pred_bagging = bagging.predict(X_test)
y_proba_bagging = rf_model.predict_proba(X_test)[::, 1]

accuracy_post = accuracy_score(y_test, y_pred_bagging)
recall_post = recall_score(y_test, y_pred_bagging)
print("Accuracy Random Forest Bagging:", round(accuracy_post, 2))

# %%
Metrics.insert(0,['Random Forest Bagging',
                        precision_score(y_test, y_pred_bagging),
                        recall_score(y_test, y_pred_bagging),
                        recall_score(y_test, y_pred_bagging, pos_label=0),
                        f1_score(y_test, y_pred_bagging),
                        confusion_matrix(y_test, y_pred_bagging)])

# %%
getMetricsTable(Metrics)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5805)
stratified_scores = []
model = bagging

for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train_fold, y_train_fold)
    stratified_scores.append(model.score(X_test_fold, y_test_fold))

Stratified.insert(0,['Random Forest Bagging',
                        stratified_scores,
                        max(stratified_scores)*100,
                        min(stratified_scores)*100,
                        (sum(stratified_scores)/len(stratified_scores_dt))*100,
                        ])

getStratifiedTable(Stratified)

# %%
fpr_RF_bagging, tpr_RF_bagging, thresholds_RF_bagging = roc_curve(y_test, y_proba_bagging)
auc_RF_bagging = roc_auc_score(y_test, y_proba_bagging)

plt.plot(fpr_RF_bagging, tpr_RF_bagging, label='ROC, auc')
plt.fill_between(fpr_RF_bagging, tpr_RF_bagging, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc_RF_bagging, ha='right', fontsize=12, weight='bold', color='blue')
plt.title("ROC Curve for Random Forest Bagging")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# # %%
estimators = [('BC', BaggingClassifier()),
              ('ABC', AdaBoostClassifier()),
              ('GBC', GradientBoostingClassifier())]

# ========================================
#  Stacking
# ========================================
stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking.fit(X_train, y_train)

# %%
y_pred_stacking = stacking.predict(X_test)
y_proba_stacking = stacking.predict_proba(X_test)[::, 1]

accuracy_post = accuracy_score(y_test, y_pred_stacking)
recall_post = recall_score(y_test, y_pred_stacking)
print("Accuracy Random Forest Stacking:", round(accuracy_post, 2))

# %%
Metrics.insert(0,['Random Forest Stacking',
                        precision_score(y_test, y_pred_stacking),
                        recall_score(y_test, y_pred_stacking),
                        recall_score(y_test, y_pred_stacking, pos_label=0),
                        f1_score(y_test, y_pred_stacking),
                        confusion_matrix(y_test, y_pred_stacking)])

# %%
getMetricsTable(Metrics)

# %%
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5805)
stratified_scores = []
model = stacking

for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train_fold, y_train_fold)
    stratified_scores.append(model.score(X_test_fold, y_test_fold))

Stratified.insert(0,['Random Forest Stacking',
                        stratified_scores,
                        max(stratified_scores)*100,
                        min(stratified_scores)*100,
                        (sum(stratified_scores)/len(stratified_scores_dt))*100,
                        ])

getStratifiedTable(Stratified)

fpr_RF_stacking, tpr_RF_stacking, thresholds_RF_stacking = roc_curve(y_test, y_proba_stacking)
auc_RF_stacking = roc_auc_score(y_test, y_proba_stacking)

plt.plot(fpr_RF_stacking, tpr_RF_stacking, label='ROC, auc')
plt.fill_between(fpr_RF_stacking, tpr_RF_stacking, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc_RF_stacking, ha='right', fontsize=12, weight='bold', color='blue')
plt.title("ROC Curve for Random Forest Stacking")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# ========================================
# Random Forest Gradient Boosting
# ========================================
gb = GradientBoostingClassifier(n_estimators=100, random_state=5805)
gb.fit(X_train, y_train)

# %%
y_pred_gradient = gb.predict(X_test)
y_proba_gradient = gb.predict_proba(X_test)[::, 1]
accuracy_gradient = accuracy_score(y_test, y_pred_gradient)
recall_gradient = recall_score(y_test, y_pred_gradient)
print("Accuracy after gradient boosting:", round(accuracy_gradient, 2))

# %%
Metrics.insert(0,['Random Forest Gradient Boosting',
                        precision_score(y_test, y_pred_gradient),
                        recall_score(y_test, y_pred_gradient),
                        recall_score(y_test, y_pred_gradient, pos_label=0),
                        f1_score(y_test, y_pred_gradient),
                        confusion_matrix(y_test, y_pred_gradient)])

# %%
getMetricsTable(Metrics)

fpr_RF_gradient, tpr_RF_gradient, thresholds_RF_gradient = roc_curve(y_test, y_proba_gradient)
auc_RF_gradient = roc_auc_score(y_test, y_proba_gradient)

plt.plot(fpr_RF_gradient, tpr_RF_gradient, label='ROC, auc')
plt.fill_between(fpr_RF_gradient, tpr_RF_gradient, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc_RF_gradient, ha='right', fontsize=12, weight='bold', color='blue')
plt.title("ROC Curve for Random Forest Gradient Boost")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# %%
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5805)
stratified_scores = []
model = gb

for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train_fold, y_train_fold)
    stratified_scores.append(model.score(X_test_fold, y_test_fold))

Stratified.insert(0,['Random Forest Gradient Boost',
                        stratified_scores,
                        max(stratified_scores)*100,
                        min(stratified_scores)*100,
                        (sum(stratified_scores)/len(stratified_scores_dt))*100,
                        ])

getStratifiedTable(Stratified)

# %%
ada = AdaBoostClassifier(n_estimators=100, random_state=5805)
ada.fit(X_train, y_train)

# %%
y_pred_ada = ada.predict(X_test)
y_proba_ada = ada.predict_proba(X_test)[::, 1]

accuracy_Ada = accuracy_score(y_test, y_pred_ada)
recall_Ada = recall_score(y_test, y_pred_ada)
print("Accuracy after Ada Boost:", round(accuracy_Ada, 2))

# %%
Metrics.insert(0,['Random Forest Ada Boosting',
                        precision_score(y_test, y_pred_ada),
                        recall_score(y_test, y_pred_ada),
                        recall_score(y_test, y_pred_ada, pos_label=0),
                        f1_score(y_test, y_pred_ada),
                        confusion_matrix(y_test, y_pred_ada)])

# %%
getMetricsTable(Metrics)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5805)
stratified_scores = []
model = ada

for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train_fold, y_train_fold)
    stratified_scores.append(model.score(X_test_fold, y_test_fold))

Stratified.insert(0,['Random Forest Ada Boost',
                        stratified_scores,
                        max(stratified_scores)*100,
                        min(stratified_scores)*100,
                        (sum(stratified_scores)/len(stratified_scores_dt))*100,
                        ])

getStratifiedTable(Stratified)

# %%
fpr_Ada, tpr_Ada, thresholds_Ada = roc_curve(y_test, y_proba_ada)
auc_Ada = roc_auc_score(y_test, y_proba_ada)

plt.plot(fpr_Ada, tpr_Ada, label='ROC, auc')
plt.fill_between(fpr_Ada, tpr_Ada, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc_Ada, ha='right', fontsize=12, weight='bold', color='blue')
plt.title("ROC Curve for Ada Boost")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# ========================================
# Multi Layer Perceptron
# ========================================
mlp = MLPClassifier(hidden_layer_sizes=(5, 2),
                        max_iter=300, activation='relu',
                        solver='adam')
mlp.fit(X_train, y_train)

# %%
y_pred_mlp = mlp.predict(X_test)
y_proba_mlp = mlp.predict_proba(X_test)[::, 1]

print('Accuracy for MultiLayer Perceptron: {:.2f}'.format(accuracy_score(y_test, y_pred_mlp)))

# %%
plt.plot(mlp.loss_curve_)
plt.title("Loss Curve", fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

# %%
Metrics.insert(0,['Multi Layer Perceptron',
                        precision_score(y_test, y_pred_mlp),
                        recall_score(y_test, y_pred_mlp),
                        recall_score(y_test, y_pred_mlp, pos_label=0),
                        f1_score(y_test, y_pred_mlp),
                        confusion_matrix(y_test, y_pred_mlp)])

# %%
getMetricsTable(Metrics)

# %%
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5805)
stratified_scores = []
model = mlp

for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train_fold, y_train_fold)
    stratified_scores.append(model.score(X_test_fold, y_test_fold))

Stratified.insert(0,['Multi Layer Perceptron',
                        stratified_scores,
                        max(stratified_scores)*100,
                        min(stratified_scores)*100,
                        (sum(stratified_scores)/len(stratified_scores_dt))*100,
                        ])

getStratifiedTable(Stratified)

# %%
fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_test, y_proba_mlp)
auc_mlp = roc_auc_score(y_test, y_proba_mlp)

plt.plot(fpr_mlp, tpr_mlp, label='ROC, auc')
plt.fill_between(fpr_mlp, tpr_mlp, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc_mlp, ha='right', fontsize=12, weight='bold', color='blue')
plt.title("ROC Curve for Multi Layer Perceptron")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# ========================================
# Multi Layer Perceptron grid search
# ========================================
param_grid_mlp = {
    'hidden_layer_sizes': [(2, 8, 16), (5, 10, 15)],
    'max_iter': [5, 10, 15],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
}

# %%
grid_mlp = GridSearchCV(mlp, param_grid_mlp, n_jobs=-1, cv=5)
grid_mlp.fit(X_train, y_train)

print(grid_mlp.best_params_)

# %%
y_pred_mlp_grid = grid_mlp.predict(X_test)
y_proba_mlp_grid = grid_mlp.predict_proba(X_test)[::, 1]

print('MLP  grid Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred_mlp_grid)))

# %%
Metrics.insert(0,['Multi Layer Perceptron Grid Search',
                        precision_score(y_test, y_pred_mlp_grid),
                        recall_score(y_test, y_pred_mlp_grid),
                        recall_score(y_test, y_pred_mlp_grid, pos_label=0),
                        f1_score(y_test, y_pred_mlp_grid),
                        confusion_matrix(y_test, y_pred_mlp_grid)])

# %%
getMetricsTable(Metrics)

# %%
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5805)
stratified_scores = []
model = grid_mlp

for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train_fold, y_train_fold)
    stratified_scores.append(model.score(X_test_fold, y_test_fold))

Stratified.insert(0,['MultiLayer Perceptron GridSearch',
                        stratified_scores,
                        max(stratified_scores)*100,
                        min(stratified_scores)*100,
                        (sum(stratified_scores)/len(stratified_scores_dt))*100,
                        ])

getStratifiedTable(Stratified)

# %%
fpr_mlp_grid, tpr_mlp_grid, thresholds_mlp_grid = roc_curve(y_test, y_proba_mlp_grid)
auc_mlp_grid = roc_auc_score(y_test, y_proba_mlp_grid)

plt.plot(fpr_mlp_grid, tpr_mlp_grid, label='ROC, auc')
plt.fill_between(fpr_mlp_grid, tpr_mlp_grid, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc_mlp_grid , ha='right', fontsize=12, weight='bold', color='blue')
plt.title("ROC Curve for Multilayer Perceptron Grid Search")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# ========================================
# Support Vector Machine: Linear
# ========================================
SVM_linear = SVC(kernel='linear', random_state=5805, probability=True)
SVM_linear.fit(X_train, y_train)

# %%
y_pred_svm_lin = SVM_linear.predict(X_test)
y_proba_svm_lin = SVM_linear.predict_proba(X_test)[::, 1]

accuracy = SVM_linear.score(X_test, y_test)
print('Linear SVM Accuracy:', accuracy)

# %%
Metrics.insert(0,['Linear SVM',
                        precision_score(y_test, y_pred_svm_lin),
                        recall_score(y_test, y_pred_svm_lin),
                        recall_score(y_test, y_pred_svm_lin, pos_label=0),
                        f1_score(y_test, y_pred_svm_lin),
                        confusion_matrix(y_test, y_pred_svm_lin)])

# %%
getMetricsTable(Metrics)

# %%
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5805)
stratified_scores = []
model = SVM_linear

for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train_fold, y_train_fold)
    stratified_scores.append(model.score(X_test_fold, y_test_fold))

Stratified.insert(0,['Linear SVM',
                        stratified_scores,
                        max(stratified_scores)*100,
                        min(stratified_scores)*100,
                        (sum(stratified_scores)/len(stratified_scores_dt))*100,
                        ])

getStratifiedTable(Stratified)

# %%
fpr_svm_lin, tpr_svm_lin, thresholds_svm_lin = roc_curve(y_test, y_proba_svm_lin)
auc_svm_lin = roc_auc_score(y_test, y_proba_svm_lin)

plt.plot(fpr_svm_lin, tpr_svm_lin, label='ROC, auc')
plt.fill_between(fpr_svm_lin, tpr_svm_lin, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc_svm_lin, ha='right', fontsize=12, weight='bold', color='blue')
plt.title("ROC Curve for Linear SVM")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# =============================================
# Support Vector Machine: Linear Grid Search
# =============================================
param_grid_svm_lin = {'C': [0.1, 0.5, 1], 'gamma': [0.1, 0.01], 'kernel': ['poly', 'sigmoid']}
SVM_linear_grid = GridSearchCV(SVC(probability=True), param_grid_svm_lin)
SVM_linear_grid.fit(X_train, y_train)

print(SVM_linear_grid.best_params_)

# %%
y_pred_svm_grid = SVM_linear_grid.predict(X_test)
y_proba_svm_grid = SVM_linear_grid.predict_proba(X_test)[::, 1]

accuracy_svm_grid = SVM_linear_grid.score(X_test, y_test)
print('SVM Grid Search Accuracy:', accuracy_svm_grid)

# %%
Metrics.insert(0,['SVM Grid Search',
                        precision_score(y_test, y_pred_svm_grid),
                        recall_score(y_test, y_pred_svm_grid),
                        recall_score(y_test, y_pred_svm_grid, pos_label=0),
                        f1_score(y_test, y_pred_svm_grid),
                        confusion_matrix(y_test, y_pred_svm_grid)])

# %%
getMetricsTable(Metrics)

# %%
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5805)
stratified_scores = []
model = SVM_linear_grid

for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train_fold, y_train_fold)
    stratified_scores.append(model.score(X_test_fold, y_test_fold))

Stratified.insert(0,['Linear SVM GridSearch',
                        stratified_scores,
                        max(stratified_scores)*100,
                        min(stratified_scores)*100,
                        (sum(stratified_scores)/len(stratified_scores_dt))*100,
                        ])

getStratifiedTable(Stratified)

# %%
fpr_svm_grid, tpr_svm_grid, thresholds_svm_grid = roc_curve(y_test, y_proba_svm_grid)
auc_svm_grid = roc_auc_score(y_test, y_proba_svm_grid)

plt.plot(fpr_svm_grid, tpr_svm_grid, label='ROC, auc')
plt.fill_between(fpr_svm_grid, tpr_svm_grid, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc_svm_grid, ha='right', fontsize=12, weight='bold', color='blue')
plt.title("ROC Curve for Linear SVM grid search")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# ========================================
# Support Vector Machine: RBF
# ========================================
SVM_rbf = SVC(kernel='rbf', random_state=5805, probability=True)
SVM_rbf.fit(X_train, y_train)

# %%
y_pred_svm_rbf = SVM_rbf.predict(X_test)
y_proba_svm_rbf = SVM_rbf.predict_proba(X_test)[::, 1]

accuracy_svm_rbf = SVM_rbf.score(X_test, y_test)
print('SVM rbf Accuracy:', accuracy_svm_rbf)

# %%
Metrics.insert(0,['SVM rbf',
                        precision_score(y_test, y_pred_svm_rbf),
                        recall_score(y_test, y_pred_svm_rbf),
                        recall_score(y_test, y_pred_svm_rbf, pos_label=0),
                        f1_score(y_test, y_pred_svm_rbf),
                        confusion_matrix(y_test, y_pred_svm_rbf)])

# %%
getMetricsTable(Metrics)

# %%
fpr_svm_rbf, tpr_svm_rbf, thresholds_svm_rbf = roc_curve(y_test, y_proba_svm_rbf)
auc_svm_rbf = roc_auc_score(y_test, y_proba_svm_rbf)

plt.plot(fpr_svm_rbf, tpr_svm_rbf, label='ROC, auc')
plt.fill_between(fpr_svm_rbf, tpr_svm_rbf, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc_svm_rbf, ha='right', fontsize=12, weight='bold', color='blue')
plt.title("ROC Curve for RBF SVM")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# ========================================
# Support Vector Machine: Polynomial
# ========================================
SVM_poly = SVC(kernel='poly', random_state=5805, probability=True)
SVM_poly.fit(X_train, y_train)

# %%
y_pred_svm_poly = SVM_poly.predict(X_test)
y_proba_svm_poly = SVM_poly.predict_proba(X_test)[::, 1]

accuracy_svm_poly = SVM_poly.score(X_test, y_test)
print('Accuracy:', accuracy_svm_poly)

# %%
Metrics.insert(0,['SVM Linear Poly',
                        precision_score(y_test, y_pred_svm_poly),
                        recall_score(y_test, y_pred_svm_poly),
                        recall_score(y_test, y_pred_svm_poly, pos_label=0),
                        f1_score(y_test, y_pred_svm_poly),
                        confusion_matrix(y_test, y_pred_svm_poly)])

# %%
getMetricsTable(Metrics)

# %%
fpr_svm_poly, tpr_svm_poly, thresholds_svm_poly = roc_curve(y_test, y_proba_svm_poly)
auc_svm_poly = roc_auc_score(y_test, y_proba_svm_poly)

plt.plot(fpr_svm_poly, tpr_svm_poly, label='ROC, auc')
plt.fill_between(fpr_svm_poly, tpr_svm_poly, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc_svm_poly, ha='right', fontsize=12, weight='bold', color='blue')
plt.title("ROC Curve for Polynomial SVM")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

