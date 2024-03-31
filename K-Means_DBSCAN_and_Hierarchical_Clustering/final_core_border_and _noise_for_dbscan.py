import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_excel('New Microsoft Excel Worksheet.xlsx')

X = data[['Area', 'Perimeter']]

eps_val = 10
min_samples_val = 4
dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
clusters = dbscan.fit_predict(X)

core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True
labels = dbscan.labels_

core_points = X[core_samples_mask]
border_points = X[np.logical_and(labels != -1, ~core_samples_mask)]
noise_points = X[labels == -1]

plt.figure(figsize=(8, 6))
plt.scatter(core_points['Area'], core_points['Perimeter'],
            c='blue', marker='o', edgecolor='black', s=200, label='Core Points')
plt.title('DBSCAN Core Points')
plt.xlabel('Area')
plt.ylabel('Perimeter')
plt.legend()
plt.grid(True)
plt.savefig('DBSCAN_Core_Points.pdf')
plt.show()

plt.figure(figsize=(15, 10))
plt.scatter(border_points['Area'], border_points['Perimeter'],
            c='orange', marker='o', edgecolor='black', s=100, label='Border Points')
plt.title('DBSCAN Border Points')
plt.xlabel('Area')
plt.ylabel('Perimeter')
plt.legend()
plt.grid(True)
plt.savefig('DBSCAN_Border_Points.pdf')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(noise_points['Area'], noise_points['Perimeter'],
            c='gray', marker='x', label='Noise Points')
plt.title('DBSCAN Noise Points')
plt.xlabel('Area')
plt.ylabel('Perimeter')
plt.legend()
plt.grid(True)
plt.savefig('DBSCAN_Noise_Points.pdf')
plt.show()

num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
print(f'Number of clusters found: {num_clusters}')
