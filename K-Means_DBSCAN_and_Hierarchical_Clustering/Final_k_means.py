import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_excel('New Microsoft Excel Worksheet.xlsx')
class_labels = data['Class']
label_encoder = LabelEncoder()
data['Class'] = label_encoder.fit_transform(class_labels)
data = np.array(data)

y = data[:, -1]
x = data[:, :-1]

selected_features = x[:, [0, 1]]

k = 2
np.random.seed(42)
initial_centers = selected_features[np.reshape(np.random.choice(range(selected_features.shape[0]), k, replace=False),(2)),:]

kmeans = KMeans(n_clusters=k, init=initial_centers)
kmeans.fit(selected_features)

labels = kmeans.labels_
centers = kmeans.cluster_centers_

plt.scatter(selected_features[:, 0], selected_features[:, 1], c=labels, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
plt.scatter(initial_centers[:, 0], initial_centers[:, 1], c='black', marker='*')
plt.title(f'KMeans Clustering with {k} clusters (automatically specified initial centers)')
plt.xlabel('Area')
plt.ylabel('Perimeter')
plt.savefig('k_means.pdf')
plt.show()

print(initial_centers)



