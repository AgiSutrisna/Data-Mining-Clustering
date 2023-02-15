import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


dataset = pd.read_csv('dataJabar.csv')
dataset.head()
dataset.info()

dataset.drop('konfirmasi_total_daily_growth', inplace=True, axis=1)
dataset.drop('konfirmasi_sembuh_daily_growth', inplace=True, axis=1)
dataset.drop('konfirmasi_meninggal_daily_growth', inplace=True, axis=1)
dataset.drop('konfirmasi_aktif_daily_growth', inplace=True, axis=1)
dataset.drop('kota_kab_belum_teridentifikasi', inplace=True, axis=1)
dataset.drop('sembuh_unidentified', inplace=True, axis=1)
dataset.drop('meninggal_unidentified', inplace=True, axis=1)
dataset.drop('tanggal', inplace=True, axis=1)
dataset.drop('nama_kab_kota', inplace=True, axis=1)
dataset.drop('id', inplace=True, axis=1)

path = r'C:\Users\agydo\OneDrive\Desktop\mdp\hasil\dataset.csv'
dataFrame = pd.DataFrame(dataset)
dataFrame.to_csv(path)

sns.heatmap(dataset.isnull(), yticklabels=False, cbar=False,cmap='viridis')
dataset.corr()
sns.heatmap(dataset.corr())
plt.show(True)

datacluster = dataset.iloc[:,1:7]
datacluster = datacluster[datacluster["konfirmasi_total"]>0]
datacluster = datacluster[datacluster["konfirmasi_sembuh"]>0]
datacluster = datacluster[datacluster["konfirmasi_meninggal"]>0]
datacluster = datacluster[datacluster["konfirmasi_aktif"]>0]
datacluster.head()

sns.scatterplot(x="konfirmasi_total", y="konfirmasi_meninggal", data=datacluster, s=100, color="red", alpha=0.5)
plt.show()

dataArray = np.array(datacluster)
print(dataArray)

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
data_scale = min_max_scaler.fit_transform(dataArray)

from sklearn.cluster import KMeans

wcss =[]

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter=300, n_init = 10, random_state=0)
    kmeans.fit(data_scale)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11),wcss)
plt.title('The elbow method')
plt.xlabel('Number of Cluster')
plt.ylabel('WCSS')
plt.show()

dataKmeans = KMeans(n_clusters = 4, init = 'k-means++', n_init=1, random_state=0)
dataKmeans.fit(data_scale)
label = dataKmeans.predict(data_scale)



centers = dataKmeans.cluster_centers_

plt.scatter(data_scale[label == 0, 0], data_scale[label == 0, 1], s = 100, c = 'red', label = '0.klaster A')
plt.scatter(data_scale[label == 1, 0], data_scale[label == 1, 1], s = 100, c = 'blue', label = '1.klaster B')
plt.scatter(data_scale[label == 2, 0], data_scale[label == 2, 1], s = 100, c = 'yellow', label = '2.klaster C')
plt.scatter(data_scale[label == 3, 0], data_scale[label == 3, 1], s = 100, c = 'red', label = '3.klaster D')
plt.scatter(centers[:,2], centers[:,0], marker='*', c = 'black', s=200, alpha=0.5, label = 'Centroids')

plt.legend()
plt.title("Clustering COVID di Jabar")
plt.show()




