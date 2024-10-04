# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start

Step 2: Load and explore customer data.

Step 3: Use the Elbow Method to find the best number of clusters.

Step 4: Perform clustering on customer data.

Step 5: Plot clustered data to visualize customer segments.

Step 6: End

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: MAGATHI D
RegisterNumber: 212223040108
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("C:/Users/admin/Downloads/printed pdfs/Mall_Customers.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No of Cluster")
plt.ylabel("wcss")
plt.title("Elbow Method")

km = KMeans(n_clusters = 5)
km.fit(data.iloc[:,3:])

KMeans(n_clusters=5)

y_pred = km.predict(data.iloc[:,3:])
y_pred


data["cluster"]=y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segments")


```
## Output:
Elbow Method:

![image](https://github.com/user-attachments/assets/6c307f08-fd47-497a-8f1d-ae1c4d19bae6)

Y-Predict:

![image](https://github.com/user-attachments/assets/9aa27834-87a1-4292-aca2-8aa34d1852de)

Customer Segments:

![image](https://github.com/user-attachments/assets/f55815a5-6bd7-4780-a82d-23a72a16d136)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
