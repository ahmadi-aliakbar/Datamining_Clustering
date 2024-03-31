import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

data = pd.read_excel('New Microsoft Excel Worksheet.xlsx')

class_labels = data['Class']

label_encoder = LabelEncoder()
data['Class'] = label_encoder.fit_transform(class_labels)

data = np.array(data)

y = data[:, -1]
x = data[:, :-1]

data = list(zip(x[:, 0],x[:, 1]))

linkage_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linkage_data)
plt.title('Dendrogram')
plt.savefig('Dendrogram.pdf')
plt.show()

clusters = fcluster(linkage_data, t=2, criterion='maxclust')

data = np.array(data)


print("Cluster Assignments:")
for i, cluster_num in enumerate(clusters, 1):
    print(f"Data point {i} is assigned to Cluster {cluster_num}")


plt.figure(figsize=(8, 6))

for cluster_num in range(1, 3):
    plt.scatter(data[clusters == cluster_num][:, 0], data[clusters == cluster_num][:, 1], label=f'Cluster {cluster_num}')

plt.xlabel('Area')
plt.ylabel('Perimeter')
plt.title('Clustered Data')
plt.legend()
plt.savefig('k-cluster.pdf')
plt.show()
