from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from matplotlib import pyplot
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import pandas as pd

X = pd.read_csv(r"C:/Users/WANGYONG/Desktop/internet+/data/tmp_data.csv")

for i in range(2,16):
    model = KMeans(n_clusters=i,random_state=0)
    clusters_ = model.fit(X)
    score = silhouette_score(X,clusters_.labels_)
    print("cluser:{},score:{}".format(i,score))