# Clustering Western Classics Using Apache Mahout

Date: 19.12.2021 <br>
by: Mohamed Gutale

## 1. Introduction

This is a big data clustering analysis project. The project will be using 27 large corpus documents of British fiction novels of Jane Austin, Emily Bronte and Charles Dickens and likes. As this is a large dataset, i will be using the distributed K-means algorithm on apache mahout and hadoop file systems - HDFS.  

## 2. Problem Definition and Framework

### 2.1 Task Definition

The aim of the project is perform the k-means algorithm on this large corpus in order to perform cluster analysis using distributed systems. I will be trying out 7k values between 2 to 14 and two different distance measure **Cosine** v **Euclidean** which is the default for Apache Mahout K-means. <br> I will then be plotting the result of the k-means algorithm on both distance measure and comment on the process. 

### 2.2 K-Means Algorithm

K-Means is the best known algorithm for performing cluster analysis using distance measures. The **K** in the K-means refers to the number of clusters which is assumed to be known in advance and **means** refers to the fact that it uses the mean as the centroid of that k-cluster. <br> The algorithm alternatives between two steps and uses iterrative refinement techniques. Given an initial set of k means, the algorithm proceeds by alternating between two steps:
1. Assignment - in this step, the algorithm assigns each observation to its nearest mean using distance measures
2. Update step - Recalculate mean centroids for observations assigned to each cluster.

The algorithm has converged when the assignments no longer change. The algorithm is often presented as assigning objects to the nearest cluster by distance. There are various distance measures used depending on the type of task. by default most of the alogithms use **Euclidean** distance however when performing text analysis its usually prefered **Cosine** distance. 

**Initialising K**
When selecting the K value, we usually want to select points that have good chance of lying in different clusters and there are various approaches:
A. Pick points that are far away from one another as possible
B. Cluster a sample of data, perform hierarchically, so there are k clusters. Pick a point from each cluster, perform that point closest to the centroid of the cluster. 

**Evaluating the Right K**
We may not know the correct value of **K** to use in the clustering. However, if we can measure the quantity of vaious values of K, we can usually guess the right value of K. we can perform this by trying a number of k values and plotting this against the average cluster distance.  This will be demonstrated by the Inter Cluster Density and Intra Cluster Density in the case of K means using Apache Mahout. Inter-cluster distance is a measure of how well the data is separated and Intra-cluster distance is a measure of how close the points lie to each other.

This k-means iterations can be seen with the images shown below from wikipedia.
![K-Means Iteration](https://upload.wikimedia.org/wikipedia/commons/e/ea/K-means_convergence.gif)

## 3. Experimental Evaluation

### 3.1 Methodology

Due to the size of files, i will be using the university of London's HDFS and Apache MapReduce framework.  These will be the steps to the project:

1. Securely copy the files from the local systems to HDFS
2. Create a sequence of files from the raw text
3. create a sparse representation of vectors from the sequence of files in a form of TFIF-IDF
4. Run the canopy algorithm to initialise the centroids and pre-cluster as this will add to the improvement of k-means
5. Run the k-means algorithm and vary the k values 
6. Run a cluster dump to review the result of kmeans. 
7. Record the Inter Cluster Density and Intra Cluster Density for each k values
8. Plot the results of K values and comment.  


This result file will then be loaded to this notebook and then visualised using Pandas and Matplotlib. Finally this project will be using the commandline to complete the tasks.  

Once the data is present on HDFS; these are the steps to executing the k-means algorithm. 

**Mahout Cluster Analysis Steps**

```
#Step1 
mahout seqdirectory \
	-i docs \
	-o docs-seqfiles \
	-c UTF-8 -chunk 5 

#Step2
mahout seq2sparse \
	-nv \
	-i docs-seqfiles \
	-o docs-vectors 

#Step3
mahout canopy \
	-i docs-vectors/tfidf-vectors \
	-ow -o docs-vectors/docs-canopy-centroids \
	-dm org.apache.mahout.common.distance.CosineDistanceMeasure \
	-t1 0.5 \
	-t2 0.1


#Step4
mahout kmeans \
	-i docs-vectors/tfidf-vectors \
	-c docs-canopy-centroids \
	-ow /
	-o hdfs://lena/user/userid/docs-kmeans-clusters \
	-dm org.apache.mahout.common.distance.CosineDistanceMeasure \
	-cl \
	-cd 0.1 \
	-x 20 \
	-k 10

#Step5
mahout clusterdump \
	-dt sequencefile \
	-d docs-vectors/dictionary.file-* \
	-i docs-kmeans-clusters/clusters-2-final \
	-o clusters.txt \
	-b 100 \
	-p docs-kmeans-clusters/clusteredPoints \
	-n 20 \ 
	--evaluate

```

its important to note that step3 above describes the Cosine distance and by removing this parameter will default to Euclidean. 

### 3.2 Datasets

As mentioned in the outline, the data is a collection of 27 books about the British Classic fiction novels. The size of the dataset is 37MB.

### 3.3 Results

After implementing the mahout k-means clustering following the steps above, i have produced the following results.


```python
#import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
```


```python
current_path = os.getcwd()
values = []

#loop through the output file from the MapReduce and convert to pandas df
for i in os.listdir(current_path):
    if 'various_k_results' in i:
        with open(os.path.join(current_path, i), 'r') as file:
            for index, value in enumerate(file.readlines()):
                if index == 0:
                    col_header = value.replace('\n', '').replace(' ','').split(',')
                else:
                    row = value.replace('\n','').split(',')
                    values.append(row)
#output file
output = pd.DataFrame(values, columns = col_header)
```


```python
#print output info
output.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 14 entries, 0 to 13
    Data columns (total 4 columns):
     #   Column                 Non-Null Count  Dtype 
    ---  ------                 --------------  ----- 
     0   Type                   14 non-null     object
     1   K_value                14 non-null     object
     2   Inter-Cluster-Density  14 non-null     object
     3   Intra-Cluster-Density  14 non-null     object
    dtypes: object(4)
    memory usage: 576.0+ bytes



```python
#clean up the data types to the correct format type
output['K_value'] = output['K_value'].astype('int')
output['Inter-Cluster-Density'] =output['Inter-Cluster-Density'].astype('float').round(2)
output['Intra-Cluster-Density'] =output['Intra-Cluster-Density'].astype('float').round(2)
```


```python
cosine = output[output['Type'] == 'Cosine']
fig, ax = plt.subplots(1,2, figsize = (16, 8))
ax[0].plot(cosine['K_value'], cosine['Inter-Cluster-Density'], c = 'y')
ax[0].set_title('Inter-Cluster Density v K values for Cosine Distance')
ax[1].plot(cosine['K_value'], cosine['Intra-Cluster-Density'], c = 'g')
ax[1].set_title('Intra-Cluster Density v K values for Cosine Distance')
fig.suptitle('Cluster Density v K values for Cosine Distance')
plt.show()
```


    
![png](output_20_0.png)
    



```python
euclidean = output[output['Type'] == 'Euclidean']
fig, ax = plt.subplots(1,2, figsize = (16, 8))
ax[0].plot(euclidean['K_value'], euclidean['Inter-Cluster-Density'], c = 'y')
ax[0].set_title('Inter-Cluster Density v K values for Euclidean Distance')
ax[1].plot(euclidean['K_value'], euclidean['Intra-Cluster-Density'], c = 'g')
ax[1].set_title('Intra-Cluster Density v K values for Euclidean Distance')
fig.suptitle('Cluster Density v K values for Euclidean Distance')
plt.show()
```


    
![png](output_21_0.png)
    


## 4. Conclusions

As you can see from the above plots from both distance measures that as the k values increase the inter-cluster density between clusters seem to increase and in the same context the intra-cluster density between points seem to decrease when the k value goes up. although this may not be smooth, the number of clusters seem to be where the two converge between 4 - 8. The lack of smoothing of these different k values may be due to the fact that the algorithm randomises the start points.  
