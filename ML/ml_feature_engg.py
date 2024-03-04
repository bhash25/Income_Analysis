import pandas as pd
import numpy as np
import seaborn as sns
from prettytable import PrettyTable
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)  # Show all columns
#pd.set_option('display.max_rows', None)  # Show all rows

df = pd.read_csv('adult.data')
print(df)
print(df.isna().sum().sum())
print(df.columns)

# ========================================
# Creating readable dataframe
# ========================================


col_change = {
    '39':'Age',
    ' State-gov':'workclass',
    ' 77516':'fnlwgt',
    ' Bachelors':'education',
    ' 13':'Education_num',
    ' Never-married':'marital-status',
    ' Adm-clerical':'occupation',
    ' Not-in-family':'relationship',
    ' White':'race',
    ' Male':'sex',
    ' 2174':'capital_gain',
    ' 0':'capital_loss',
    ' 40':'hours_per_week',
    ' United-States':'country',
    ' <=50K':'income'
}
df.rename(columns = col_change, inplace=True)


print(df.info)
print(df.describe())

# Mapping dictionary
# education_mapping = {
#     ' Bachelors': 'college',
#     ' HS-grad': 'high school',
#     ' 11th': 'high school',
#     ' Masters': 'post-grad',
#     ' 9th': 'high school',
#     ' Some-college': 'college',
#     ' Assoc-acdm': 'college',
#     ' Prof-school': 'post-grad',
#     ' 5th-6th': 'school',
#     ' 10th': 'high school',
#     ' 1st-4th': 'school',
#     ' Preschool': 'school',
#     ' 12th': 'high school'
# }
#
# # Apply the mapping to the 'education' column
# df['Education'] = df['education'].replace(education_mapping)
# df.drop('education', axis=1, inplace=True)
#
# # Mapping dictionary
# marital_status_mapping = {
#     ' Married-civ-spouse': 'married',
#     ' Divorced': 'divorced',
#     ' Married-spouse-absent': 'married',
#     ' Never-married': 'single',
#     ' Separated': 'divorced',
#     ' Married-AF-spouse': 'married',
#     ' Widowed': 'widowed'
# }
#
# # Apply the mapping to the 'marital_status' column
# df['Marital_Status'] = df['marital-status'].replace(marital_status_mapping)
# df.drop('marital-status', axis=1, inplace=True)
#
# # Mapping dictionary
# occupation_mapping = {
#     ' Exec-managerial': 'Professional',
#     ' Handlers-cleaners': 'Service',
#     ' Prof-specialty': 'Professional',
#     ' Other-service': 'Service',
#     ' Adm-clerical': 'Professional',
#     ' Sales': 'Professional',
#     ' Craft-repair': 'Service',
#     ' Tech-support': 'Professional',
#     ' ?': 'Unknown',
#     ' Protective-serv': 'Security',
#     ' Armed-Forces': 'Veteran',
#     ' Priv-house-serv': 'Security',
#     ' Machine-op-inspct': 'Service',
#     ' Transport-moving': 'Service',
#     ' Farming-fishing': 'Farmer'
#
# }
#
# # Apply the mapping to the 'occupation' column
# df['Occupation'] = df['occupation'].replace(occupation_mapping)
# df.drop('occupation', axis=1, inplace=True)
#
# # Mapping dictionary
# relationship_mapping = {
#     ' Husband': 'Spouse',
#     ' Not-in-family': 'Non-family',
#     ' Wife': 'Spouse',
#     ' Own-child': 'Parent',
#     ' Unmarried': 'Non-family',
#     ' Other-relative': 'Non-family'
# }
#
# # Apply a lambda function to transform the 'relationship' column
# df['Relationship'] = df['relationship'].apply(lambda x: relationship_mapping.get(x, x))
# df.drop('relationship', axis=1, inplace=True)
df.drop('Education_num', axis=1, inplace=True)
df['Country_USA'] = df['country'].apply(lambda x: 'USA' if x == 'United-States' else 'Other')
df.drop('country', axis=1, inplace=True)
df['income >50K'] = df['income'].apply(lambda x: 1 if x == ' >50K' else 0)
df.drop('income', axis=1, inplace=True)

# ========================================
# Count Plot
# ========================================
plt.figure()
sns.countplot(x='income >50K', data=df)
plt.xlabel("income")
plt.ylabel("Count")
plt.title("Count Plot of income")
plt.tight_layout()
plt.show()
def box_plot(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df_temp = df[(df > lower) & (df < upper)]
    return df_temp


# ========================================
# Box Plot
# ========================================
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

# Boxplot on the age
axes[0].boxplot(df['Age'], vert=False)
axes[0].set_title('Boxplot of Age')

# Boxplot on the second subplot
axes[1].boxplot(box_plot(df['Age']), vert=False)
axes[1].set_title('Boxplot of Age after removing outliers')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

# Boxplot on the hours
axes[0].boxplot(df['hours_per_week'], vert=False)
axes[0].set_title('Boxplot of hours')

# Boxplot on the second subplot
axes[1].boxplot(box_plot(df['hours_per_week']), vert=False)
axes[1].set_title('Boxplot of hours after removing outliers')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

# Boxplot on the fnlwgt
axes[0].boxplot(df['fnlwgt'], vert=False)
axes[0].set_title('Boxplot of fnlwgt')

# Boxplot on the second subplot
axes[1].boxplot(box_plot(df['fnlwgt']), vert=False)
axes[1].set_title('Boxplot of fnlwgt after removing outliers')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

# Boxplot on the capital_loss
axes[0].boxplot(df['capital_loss'], vert=False)
axes[0].set_title('Boxplot of capital_loss')

# Boxplot on the second subplot
axes[1].boxplot(box_plot(df['capital_loss']), vert=False)
axes[1].set_title('Boxplot of capital_loss after removing outliers')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

# Boxplot on the capital_gain
axes[0].boxplot(df['capital_gain'], vert=False)
axes[0].set_title('Boxplot of capital_gain')

# Boxplot on the second subplot
axes[1].boxplot(box_plot(df['capital_gain']), vert=False)
axes[1].set_title('Boxplot of capital_gain after removing outliers')
plt.tight_layout()
plt.show()

# ========================================
# Standardising the numerical columns
# ========================================
y = df['income >50K']
df.drop('income >50K', axis=1, inplace=True)
numerical_columns = df.select_dtypes(include=['number']).columns

# Initialize StandardScaler
scaler = StandardScaler()

# Standardize numerical columns
df_stan = scaler.fit_transform(df[numerical_columns])
df_stan = pd.DataFrame(df_stan, columns=numerical_columns)

# Display the standardized DataFrame
print(f'standardised df:\n{df_stan}')

# ========================================
# Encoding the categorical columns
# ========================================
categorical_columns = df.select_dtypes(include=['object']).columns

# Perform one-hot encoding with drop_first=True to avoid dummy variable trap
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True).astype(int)
df_encoded.drop(['Age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week'], axis=1, inplace=True)
# Display the one-hot encoded DataFrame
print(f'encoded df:\n{df_encoded.columns}')


# ========================================
# Splitting dataset to train and test
# ========================================
X = pd.concat([df_stan, df_encoded], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, test_size=0.2, shuffle=True, random_state=5805)
print(f'Training df \n {X_train.columns}')


# Create a table that mentions the feature selection technique and the number of features selected and removed
feature_selection = pd.DataFrame(columns=['Feature Selection Technique', 'Number of Features Selected', 'Number of Features Removed'])

correlation_matrix = df[numerical_columns].corr()
covariance_matrix = df[numerical_columns].cov()
# ========================================
# Heatmaps
# ========================================
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# Set the title
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(covariance_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# Set the title
plt.title('Covariance Heatmap')
plt.show()
# ========================================
# Using PCA for feature reduction
# ========================================

pca = PCA()
X_train_pca = X_train.copy()
X_test_pca = X_test.copy()

# Fit PCA to the training data
X_train_pca = pca.fit_transform(X_train_pca)
X_test_pca = pca.transform(X_test_pca)
# Get explained variance ratio
var_ratio = pca.explained_variance_ratio_

# Initialize variables
sum_var = 0
count = 0

# Find the number of features explaining more than 90% variance
for i, explained_variance in enumerate(var_ratio):
    sum_var += explained_variance
    count += 1
    if sum_var > 0.90:
        break

print("Number of features that explain >90% variance:", count)

# Create lists to store cumulative explained variance
l1 = []
l2 = []
cumulative_sum = 0
# Calculate cumulative explained variance
for i in range(len(var_ratio)):
    cumulative_sum += var_ratio[i]
    l1.append(i + 1)
    l2.append(cumulative_sum)

# Create a DataFrame and plot the cumulative explained variance
tb = pd.DataFrame(l2, index=l1)
ax = tb.plot(kind="line", title='Cumulative explained variance vs Number of features',
             ylabel='Cumulative explained variance', xlabel='Number of features')

# Draw vertical and horizontal lines at 90% threshold
ax.axhline(y=0.90, color='r', linestyle='--', label='90% Threshold')
ax.axvline(x=count, color='g', linestyle='--', label='Features = ' + str(count))
plt.title('Principle Component Analysis')
plt.legend()
plt.show()

# Conditional Number of Original and Reduced data
print('Condition Number of Original data:', np.linalg.cond(X_train))
print('Condition Number of Reduced data:', np.linalg.cond(X_train_pca))

feature_selection = feature_selection._append({'Feature Selection Technique': 'PCA',
                                                'Number of Features Selected': np.where(np.cumsum(pca.explained_variance_ratio_) > 0.9)[0][0] + 1,
                                                'Number of Features Removed': X_train_pca.shape[1] - np.where(np.cumsum(pca.explained_variance_ratio_) > 0.9)[0][0] + 1,},
                                                 ignore_index=True)

# ========================================
# Using Random Forest for feature reduction
# ========================================

X_train_rf = X_train.copy()
X_test_rf = X_test.copy()
model = RandomForestRegressor(random_state=5805, max_depth=10)
model.fit(X_train_rf, y_train)
features=X_train_rf.columns
importances = model.feature_importances_
indices = np.argsort(importances)[:]
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.show()

# print(importances)
# print(indices)
drop_feat = []
for i in range(len(features)):
    if importances[i] < 0.01:
        X_train_rf.drop(features[i], axis=1, inplace=True)
        X_test_rf.drop(features[i], axis=1, inplace=True)
        drop_feat.append(features[i])
print(f'dropped features:\n{drop_feat}')
print(f'selected features:\n{X_train_rf.columns}')

feature_selection = feature_selection._append({'Feature Selection Technique': 'Random Forest',
                                                'Number of Features Selected': len(X_train_rf.columns),
                                                'Number of Features Removed': len(drop_feat)},
                                                 ignore_index=True)

# ========================================
# Using Singular Value Decomposition Analysis
# ========================================
X_train_svd = X_train.copy()
X_test_svd = X_test.copy()
# First SVD with full components
svd_full = TruncatedSVD(n_components=X_train_svd.shape[1] - 1)
X_svd_full = svd_full.fit_transform(X_train_svd)

cumulative_variance_ratio = np.cumsum(svd_full.explained_variance_ratio_)

n_components_90 = np.argmax(cumulative_variance_ratio >= 0.9) + 1

# Now SVD with 90% variance explained
svd_90 = TruncatedSVD(n_components=n_components_90)
X_reduced_90 = svd_90.fit_transform(X_train_svd)

print("Original Shape: ", X_train_svd.shape)
print("Reduced Shape: ", X_reduced_90.shape)

plt.figure(figsize=(10, 10))
plt.plot(np.arange(1, len(cumulative_variance_ratio) + 1, 1), cumulative_variance_ratio, label="Cumulative Explained Variance")
plt.xticks(np.arange(0, len(cumulative_variance_ratio) + 1, 10))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title('Singular Value Decomposition')
plt.axvline(x=n_components_90, color="red", linestyle="--", label='Features = ' + str(n_components_90))
plt.axhline(y=0.9, color='black', linestyle="--", label='90% Threshold')
plt.legend()
plt.tight_layout()
plt.show()

# Conditional Number of Original and Reduced data
print('Condition Number of Original data:', np.linalg.cond(X_train_svd))
print('Condition Number of Reduced data:', np.linalg.cond(X_reduced_90))

feature_selection = feature_selection._append({'Feature Selection Technique': 'Singular Value Decomposition Analysis',
                                                'Number of Features Selected': n_components_90,
                                                'Number of Features Removed': X_train_svd.shape[1] - n_components_90},
                                                 ignore_index=True)

# ========================================
# Using Variance Inflation Factor Analysis
# ========================================

X_train_vif = X_train.copy()
X_test_vif = X_test.copy()
# Calculate VIF and count the number of features removed
vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(X_train_vif, i) for i in range(X_train_vif.shape[1])]
vif["Features"] = X_train_vif.columns
print(vif)

# Drop the columns with highest VIF > 10
vif_filtered = vif[vif['vif']<10]
print(vif_filtered)

vif_iter1 = X.shape[1] - vif_filtered.shape[0]
print("Number of features removed after first iteration:", vif_iter1)

# Calculate VIF again and count the number of features removed
vif_2 = pd.DataFrame()
vif_2["vif"] = [variance_inflation_factor(X_train_vif[vif_filtered['Features']], i) for i in range(X_train_vif[vif_filtered['Features']].shape[1])]
vif_2["Features"] = X_train_vif[vif_filtered['Features']].columns
print(vif_2)

# Drop the columns with highest VIF > 10
vif_filter_2 = vif_2[vif_2['vif']<10]
print(vif_filter_2)

vif_iter2 = vif_filtered.shape[0] - vif_filter_2.shape[0]
print("Number of features removed after second iteration:", vif_iter2)


# Calculate total number of features removed
vif_total = vif_iter1 + vif_iter2
print("Total number of features removed:", vif_total)

feature_selection = feature_selection._append({'Feature Selection Technique': 'Variance Inflation Factor',
                                                'Number of Features Selected': vif_filter_2.shape[0],
                                                'Number of Features Removed': vif_total},
                                                 ignore_index=True)

def create_feature_selecction_table(df, name):
    x = PrettyTable()
    x.title = f"{name} Comparison"
    x.field_names = df.columns

    for index, row in df.iterrows():
        x.add_row(row)

    print(x)

create_feature_selecction_table(feature_selection, 'Feature Selection/Dimensionality Reduction')

# ========================================
# Check for Colinearity
# ========================================
selected_col = X_train_rf.columns
print(selected_col)
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(X_train[selected_col].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
# Set the title
plt.title('Correlation Heatmap')
plt.show()
