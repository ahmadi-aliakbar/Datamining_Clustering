import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

data = pd.read_excel('New Microsoft Excel Worksheet.xlsx')

X = data[['Area', 'Perimeter']]

eps_val = 10
min_samples_val = 6
dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
clusters = dbscan.fit_predict(X)

plt.figure(figsize=(8, 6))

plt.scatter(X['Area'], X['Perimeter'], c=clusters, cmap='viridis', marker='o', edgecolor='black', s=80)
plt.title('DBSCAN Clustering(eps = 10, minpts = 6)')
plt.xlabel('Area')
plt.ylabel('Perimeter')
plt.colorbar(label='Cluster ID')
plt.grid(True)
plt.savefig('DBSCAN.pdf')
plt.show()

num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
print(f'Number of clusters found: {num_clusters}')



