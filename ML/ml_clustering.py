import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings("ignore")
from ml_feature_engg import selected_col, X, y

df = X

df_sel = pd.concat([df[selected_col], y], axis=1)
# ========================================
# Clustering
# ========================================
#
# ========================================
# K-Means Clustering
# ========================================
#
def cal_WSS(points, kmax):
    sse = []
    for k in range(1, kmax + 1):
        kmeans = KMeans(n_clusters=k, random_state=5805).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        # points is a NumPy array now, so this indexing is valid
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)

    return sse

# Convert the DataFrame to a NumPy array
points_array = df_sel.to_numpy()

k = 10
sse = cal_WSS(points_array, k)

# Plot
plt.figure()
plt.plot(np.arange(1, k+1, 1), sse)
plt.xticks(np.arange(1, k+1, 1))
plt.grid()
plt.xlabel('k')
plt.ylabel('WSS')
plt.title('k selection in k-mean Elbow Algorithm')
plt.show()


# Silhouette Score
sil_score = []
kmax = 10

for k in range(2, kmax+1):
    kmeans = KMeans(n_clusters=k).fit(points_array)
    labels = kmeans.labels_
    sil_score.append(silhouette_score(points_array, labels, metric='euclidean'))
plt.figure()
plt.plot(np.arange(2, k + 1, 1), sil_score, 'bx-')
plt.xticks(np.arange(2,k+1,1))
plt.grid()
plt.xlabel('k')
plt.ylabel('Silhouette score')
plt.title('Silhouette Method')
plt.show()



# ========================================
# Association Rules
# ========================================


basket = df_sel.groupby('income >50K').agg({'Age': 'sum', 'hours_per_week': 'sum', 'fnlwgt': 'sum'}).reset_index()
def encode(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

numeric_columns = ['Age', 'hours_per_week', 'fnlwgt']
basket_sets = basket[numeric_columns].applymap(encode)

basket_sets['income >50K'] = basket['income >50K']
basket_sets.set_index('income >50K', inplace=True)

# Apply the Apriori algorithm
freq_items = apriori(basket_sets, min_support=0.07, use_colnames=True)
rules = association_rules(freq_items, metric="lift", min_threshold=1)
print(rules.head(5).to_string())