'''
Clustering for datase exploration
-Unsupervised learning finds patterns in data
-E.g., clustering customers by their purchases 
-Copressing the data using purchase patterns (dimension reduction)

Supervised vs. Unsupervised learning 
-Supervised learning finds patterns for a prediced task 
-E.g., classify tumors as benign or cancerous (labels)
-Unsupervised learning find patterns in data 
-... but without a specific prediction task in mind 

Arrays, features & samples 
* 2D Numpy array 
* Columns are measuremets (the features)
* Rows represent iris plants (the samples)

Iris data is 4-dimensional 
* Iris samples are pooints in 4 dimensional space 
* Dimension = number of features 
* Dimension too high to visualize!
 .. but unsupervised learning gives insight 
 
 k-means clustering 
 -finds clusters of samples 
 -Number of clusters must be specified 
 -Implemented in sklearn ("scikit-learn")
 
 from sklearn.cluster impor KMeans
 model = KMeans(n_clusters=3)
 model.fit(samples) ##Fits the model to the data by locating and remembering the regions were the different classes occur 
 labels = model.predict(samples)##this returns a cluster lable for each sample indicating to which cluster a sample belongs 
 
 Cluster labls for new sampes 
 * New samples can be assigned to existing clusters 
 * k-means remembers the mean of each cluster (the "centroids")
 * Finds the nearest centroid to each new sample 
 
 New samples can be passed to the predict method 
 
 new_labels = model.predict(new_samples)
 
 Generating a scatter plot
 import matplotlib.pyplot as plt
 xs = samples[:,0]
 ys = samples[:,2]
 plt.scatter(xs,ys, c=labels)
 plt.show
 '''
 
 ###Clustering 2D points 
 # Import KMeans
from sklearn.cluster import KMeans

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)

# Fit model to points
model.fit(points)

# Determine the cluster labels of new_points: labels
labels = model.predict(new_points)

# Print cluster labels of new_points
print(labels)

###Inspect your clustering 
# Import pyplot
import matplotlib.pyplot as plt

# Assign the columns of new_points: xs and ys
xs = new_points[:,0]
ys = new_points[:,1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs,ys, c=labels, alpha=0.5)

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x,centroids_y, marker='D', s=50)
plt.show()


'''
Evaluating a cluster 
* Can check correspondence with e.g. iris species 
* ... but what if there are no species to check against?
* Measure quality of a clustering 
* Informs choice of how many clusters to look for 

Iriscluster vs species 
* k-means found 3 clusters amongst the iris samples 
* D the clusters correspond to the species 


Cross tabulation with pandas 
species     setosa      versicolor      virginica
labels
0           0           2               36
1           50          0               0
2           0           48              14


species of each samples is given as strings 

import pandas as pd 
df = pd.DataFrame({'labels': labels, 'species': species})
print(df)
ct = pd.crosstab(df['labels'], df['species'])
print(ct)


In most datasets the samples are not labeled by species 
How can the quality of the clustering be evaluated in these cases 

Measuring clustering quality 
* Using only samples and their cluster labels 
* A good clustering has tight clusters
* Samples in each cluster bunched together 

Inertia measures clustering quality 
* Measuring how spread out the clusters are (lower is better)
* Distance from each sample to centroid of its cluster 
* After fit(), available as attribute inertia_

from sklean.cluster import KMeans

model = KMeans(n_clusters=3)
model.fit(samples)
print(model.inertia_)


How many clusters to choose?
* A good clustering has tight clusters (so low inertia)
* ... but not to many clusters!
* Choose an "elbow" in the ineertia plot
* Where inertia begins to decrease more slowly
* E.g., for iris dataset, 3 is a good choice
'''

###How many clusters of grain?

ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(samples)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


###Evaluating the grain clustering 
# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters=3)

# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(samples)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)


'''
Piedmont wines dataset 
* 178 samples from 3 distinct varieties of red wine: Barolo, Grignolino and Barera
* Features measure chemical compoosition e.g. alcohol content 
* Visual propertie like "color intensity"

Clustering the wines 
from sklearn.cluset import KMeans
model = KMeans(n_clusters=3)
labels = model.fit_predict(samples)
df = pd.DataFrame({'labels':labels, 
                    'varieties':varieties})
ct = pd.crosstab(df['labels'], df['varieties'])
print(ct)

Feature variances 
*The wine dataset have very different variances!
*Variance of a feature measures spread of its values  

To give every feature a chance, the data needs to be transformed to the features have equal variance 

StandardScaler
* In kmeans:feature variance = feature influence 
* StandardScalar transforms each featur to have mean 0 and variance 1

sklearn StandardScaler 
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaler.fit(samples)
StandardScaler(copy=True, with_mean=True, with_std=True)
samples_scaled = scaler.transform(samples)

Similar methods 
* StandardScaler and KMeans have similar methods 
* Use fit() / transform() with StandardScaler
* Use fit() / predict()with KMeans

StandardScaler, then KMeans
* need to perform two steps: StandardScaler, then KMeans
* Use sklearn pipeline to combne multiple steps 
* Data flows from one step into the next 

Pipelines combie multiple steps 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
scaler = StandardScaler()
kmeans = KMeans(n_clusters=3)
from sklearn.pipline import make_pipeline
pipeline = make_pipline(scaler, kmeans)
pipeline.fit(samples)
labels = pipline.predict(samples)
'''

###Scaling fish data for clustering 

# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, kmeans)


###Clustering stocks using KMeans
# Import pandas
import pandas as pd

# Fit the pipeline to samples
pipeline.fit(samples)

# Calculate the cluster labels: labels
labels = pipeline.predict(samples)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'labels':labels, 'species':species})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['species'])

# Display ct
print(ct)


###Clustering stock using KMeans
# Import Normalizer
from sklearn.preprocessing import Normalizer

# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters=10)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer, kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(movements)


###Which stocks move together 
# Import pandas
import pandas as pd

# Predict the cluster labels: labels
labels = pipeline.predict(movements)

# Create a DataFrame aligning labels and companies: df
df = pd.DataFrame({'labels': labels, 'companies': companies})

# Display df sorted by cluster label
print(df.sort_values('labels'))


'''
Visualizing hierarchies 
Unsupervised Learning techniques for visualization: t-SNE and hierarchical clustering 
t-SNE: Creates a 2D map of a dataset (later)

A hierarchy of groups 
* Groups of living things cn form a hierarchy 

Hierarchical clusterin with Countries 
* Every country begins in a separate cluster 
* At each step, the two closest clusters are merged 
* Continue intil all countries in a single cluster 
* This is "agglomerative" hierarchical clustering 

The dendrogram of hierarchical clustering 
* Read from the bottom up 
* Vertical lines represent clusters 

Hierarchical clustering with SciPy 
* Given samples (the array of scores), and country_names 

import matplotlib.plyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
mergings = linkage(samples, method='complete')
dendrogram(mergings,
            labels=country_names,
            leaf_rotation=90,
            leaf_front_size=6)
plt.show()
'''

###Hierarchical clustering of the grain data 
# Perform the necessary imports
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Calculate the linkage: mergings
mergings = linkage(samples, method='complete')

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()


###Hierarchies of stocks
# Import normalize
from sklearn.preprocessing import normalize

# Normalize the movements: normalized_movements
normalized_movements = normalize(movements)

# Calculate the linkage: mergings
mergings = linkage(normalized_movements, method='complete')

# Plot the dendrogram
dendrogram(mergings,
           labels=companies,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()


'''
Dendrograms show cluster distances 
* Height on dendrogram = distance between merging clusters 
* Height on dendrogram specifies max. distance between merging clusters 

Distance between clusters 
* Defined by a "linkage method"
* In "complete" linkage: distance between clusters is max. distance between their samples 


* Specified via method parameter, e.g. linkage(samples, method="complete")
* Different linkage method, different hierarchical clustering!

Extracting cluster labels 
* Use the fcluster() function
* Returns a NumPy array of cluster labels 

Extracting cluster labels using fcluster 
from scipy.cluster.hierarchy import linkage
mergings = linkage(samples, method='complete')
from scipy.cluster.hierarchy import fcluster
labels = fcluster(mergings, 15, criterion='distance')
print(labels)

Aligning cluster labels with country names 
Given a list of strings country_names:
import pandas as pd 
pairs = pd.DataFrame({'labels':labels, 'countries': country_names})
print(pairs.sort_values('labels'))
#cluster labels start at 1 not at 0
'''

###Different linkage, different hierarchical clustering!
# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Calculate the linkage: mergings
mergings = linkage(samples, method='single')

# Plot the dendrogram
dendrogram(mergings,
            labels=country_names,
            leaf_rotation=90,
            leaf_font_size=6
            )
plt.show()



###Extracting the cluster labels 
# Perform the necessary imports
import pandas as pd
from scipy.cluster.hierarchy import fcluster

# Use fcluster to extract labels: labels
labels = fcluster(mergings, 6, criterion='distance')

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)


'''
t-SNE for 2-dimensional maps
* t-SNE = "t-distributed stochastic neighbor embedding"
* Maps samples to 2D space (or 3D)
* Map approximately preserves nearness of samples 
* Great for inspecting datasets 

t-SNE in sklearn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
model = TSNE(learning_rate=100)
transformed = model.fit_transform(samples)
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs, ys, c=species)
plt.show()

t-SNE has only fit_transform() method
* Has a fit_transform() method
* Simultaneouosly fits the model and transform() methods
* Can't extend the map to include new data samples 
* Must start over each time!

t-SNE learning rate 
* Choose learning rate for the dataset 
* Wrong choice:points bunch together 
* Try values between 50 and 200 

Different every time 
* t-SNE features are different every time 
* Piedmont wines, 3 runs, 3 different scatter plots 
'''

###t-SNE visualization of grain dataset
# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=200)

# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(samples)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1st feature: ys
ys = tsne_features[:,1]

# Scatter plot, coloring by variety_numbers
plt.scatter(xs, ys, c=variety_numbers)
plt.show()

###A t-SNE map of the stock market 
# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=50)

# Apply fit_transform to normalized_movements: tsne_features
tsne_features = model.fit_transform(normalized_movements)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1th feature: ys
ys = tsne_features[:,1]

# Scatter plot
plt.scatter(xs, ys, alpha=0.5)

# Annotate the points
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()


'''
Principal Component Analysis 
Visualizing the PCA transformation 
Dimension reduction 
* More efficient storage and computation 
* Remove less-informative "noise" features 
* ...which casue problems for prediction tasks, e.g. classification, regression 

Principal Component Analysis 
PCA = "Principal Component Analysis"
Fundamenta dimension reduction techniques
First step "decorrelation" (considered here)
Second step reduces dimension (considered later)

PCA align with axes
Rotates data samples to be aligned with axes 
Shifts data samples to they have a mean of 0
No information is lost 

PCA follows the fit/transform pattern 
PCA is a scikit-learn component like KMeans or StadardScaler 
fit() learns the transformation from given data 
transform() applies the learned transformation 
transform() can also be applied to new data 

PCA features 
* Rows of transformed correspond to samples 
* Columns of transformed are the "PCA features"
* Row gives PCA feature values of corresponding sample

PCA features are not correlated e.g. total_phenols and 0d280
PCA aligns the data with axes
Resulting PCA features are not linearly correlated ("decorrelation")


Using scikit-learn PCA
* samples = array of two features (total_pheols & od280)
from sklearn.decomposition import PCA 
model = PCA()
model.fit(samples)
transformed = model. transform(samples)
print(transformed)

Linear correlation can be measured with the Pearson correlation 
* Measure linear correlation of features 
* Value between -1 and 1 
* Value of 0 means no linear correlation 

"Pricipal component" = directions of variance 

Available as components_ attribute of PCA object 
Each row defines displacement from mean 
print(model.components_)
'''

###Correlated data in nature 
# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Assign the 0th column of grains: width
width = grains[:,0]

# Assign the 1st column of grains: length
length = grains[:,1]

# Scatter plot width vs length
plt.scatter(width, length)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation
correlation, pvalue = pearsonr(width, length)

# Display the correlation
print(correlation)


###Decorrelating the grain measurements with PCA 
# Import PCA
from sklearn.decomposition import PCA

# Create PCA instance: model
model = PCA()

# Apply the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(grains)

# Assign 0th column of pca_features: xs
xs = pca_features[:,0]

# Assign 1st column of pca_features: ys
ys = pca_features[:,1]

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)

# Display the correlation
print(correlation)


'''
Intrinsic dimension of a flight path 
* 2 features: longitude and latitude at points along a flight path 
* Dataset appears to be 2-dimensional
* But can aproximately using one feature: displacement along flight path 
* Is intrinsically 1-dimensional

Intrinsic dimension = number of features needed to approximate the dataset 
Essential idea behind dimension reduction 
What is the most compact representation of the samples?

Versicolor dataset 
* "versicolor", one of the iris species 
* Only 3 features: sepal length, sepal width, and petal width
* Samples are points in 3D space 

Versicolor dataset has intrinsic dimension 2 
* Samples lie close to a flat 2-dimensional sheet 
* So can be approximated using 2 features 

PCA identifies intrinsic dimension
* Scatter plots work only if samples have 2 or 3 features 
* PCA identifies intrinsic dimension when samples have any number of features 

* Intrinsic dimension = number of PCA deatures with significant variance 

Plotting the variances of PCA features 
* samples = array of versicolor samples 

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
pca = PCA()
pca.fit(samples)

features = range(pca.n_components_)


Plotting the variance of PCA features 
plt.bar(features, pca.explainted_variance_)
plt.xticks(features)
plt.ylabel('variance')
plt.xlabel('PCA feature')
plt.show()

Intrinsic dimension can be ambiguous 
* Intrinsic dimension is an idelaization 
* ...there is not always one correct answer!
* Piedmont wines: could argue for 2, or for 3, or more
'''

###The first principal component 
# Make a scatter plot of the untransformed points
plt.scatter(grains[:,0], grains[:,1])

# Create a PCA instance: model
model = PCA()

# Fit model to points
model.fit(grains)

# Get the mean of the grain samples: mean
mean = model.mean_

# Get the first principal component: first_pc
first_pc = model.components_[0,:]

# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)

# Keep axes on same scale
plt.axis('equal')
plt.show()

###Variance of the PCA features 
# Perform the necessary imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, pca)

# Fit the pipeline to 'samples'
pipeline.fit(samples)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

'''
Dimension reduction 
* Represents some data, using less features 
* Important part of machine-learning pipelines 
* Can be performed using PCA 

Dimension reduction with PCA 
* PCA features are in decreasing order of variance 
* Assumes the low variance features are "noise"
* ... and high variance features are informative 

To use PCA for dimension reduction, you need to specify how many PCA features to keep

Dimension reduction with PCA
* Specify how many featres to keep
* E.g. PCA(n-components=2)
* Keep the first 2 PCA features 
* Intrinsic dimension is a good choice

Example: 
samples = array of iris measurements (4 features)
species = list of iris species numbers 

from sklearn.decomposition import PCA 
pca = PCA(n_components=2)
pca.fit(samples)
transformed = pca.transform(sample)
print(transfored.shape)

Iris dataset in 2 dimensions 
import matplotlib.pyplot as plt
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs, ys, c=species)
plt.show()

* PCA has reduced the dimension to 2
* Retained the 2 PCA features with highest variance
* Important information preserved: species remain distinct

Dimension reduction with PCA
* Discards low variance PCA features 
* Assumes the high variance features are informative 
* Assumption typically holds in practive (e.g. for iris)

Word frequecy arrays 
* Rows represent documents, columns represent words 
* Entries measure presence of each word in each document 
* ... measure using "tf-ldf" (more later)

Sparse arrays and csr_matrix
* "Sparse": most entries are zero 
* Can use scipy.sparse.csr_matrix instead of NumPy array 
* csr_matrix remembers only the non-zero entries (saves space)

Scikit-learn's PCA doesn't support csr_matrices, and you'll need to use TruncatedSVD instead 

* scikit-learn PCA doesn't support csr_matrix
* Use scikit-learn TruncatedSVD instead 
* Performs same transformation 

from sklearn.decomposition import TruncatedSVD
model = TruncatedSVD(n_components=3)
model.fit(documents) # documents is csr_matrix
TrucatedSVD(algorithm='randomized', ... )
transformed = model.transform(documents)
'''

###Dimension reduction of the fish measurments 
# Import PCA
from sklearn.decomposition import PCA

# Create a PCA model with 2 components: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)


###A tf-idf word-frequency array 
# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer()

# Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents)

# Print result of toarray() method
print(csr_mat.toarray())

# Get the words: words
words = tfidf.get_feature_names()

# Print words
print(words)


###Clustering Wikipedia part I
# Perform the necessary imports
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=50)

# Create a KMeans instance: kmeans
kmeans = KMeans(n_clusters=6)

# Create a pipeline: pipeline
pipeline = make_pipeline(svd, kmeans)


###Clustering Wikipedia part II
# Import pandas
import pandas as pd

# Fit the pipeline to articles
pipeline.fit(articles)

# Calculate the cluster labels: labels
labels = pipeline.predict(articles)

# Create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'article': titles})

# Display df sorted by cluster label
print(df.sort_values(by='label'))


'''
Non-negative matrix factorization (NMF)
NMF = "non-negative matrix factorization"
Dimension reduction technique 
NMF models are interpretable (unline PCA)
Easy to interpret means easy to explain!
However, all sample features must be non-negative (>=0)

Interpretable parts 
* NMF expresses documents as combination of topics (or "themes")
For example, NMF decomposes documents as combination of common themes 
and images as combinations of common patters

Using scikit-learn NMF 
* Follows fit() / transform() pattern 
* Must specify number of components e.g. 
NMF(n_compoenents=2)
* Works with NumPy arras and with csr_matrix 

Example usage of NMF
* samples is the word-frequency array 
from sklearn.decomposition import NMF
model = NMF(n_components=2)
model.fit(samples)
nmf_features = model.transform(samples)
print(model.components_)

NMF components 
* NMF has components 
* ...just like PCA has principal components 
* Dimension of components = dimension of samples
* Entries are non-negative

NMF features 
* NMF feature values are non-negative
* Can be used to reconstruct the samples 
* ... combine feature values with components 
print(nmf_features)
print(samples[i,:])
print(nmf_features[i,:])

Sample reconstruction 
* Multiply components by feature values, and add up 
* Can also be expressed as a product of matrices 
* This is the "Matrix Factorization" in "NMF"

NMF fits to non-negative data only 
* Word frequencies in each document 
* Imgaes encoded as arrays 
* Audio spectrograms
* Purchase histories on e-commerce sites 
* ... and many more!
'''

###NMF applied to Wikipedia articles
# Import NMF
from sklearn.decomposition import NMF

# Create an NMF instance: model
model = NMF(n_components=6)

# Fit the model to articles
model.fit(articles)

# Transform the articles: nmf_features
nmf_features = model.transform(articles)

# Print the NMF features
print(nmf_features.round(2))


###NMF features of the Wikipedia
# Import pandas
import pandas as pd

# Create a pandas DataFrame: df
df = pd.DataFrame(nmf_features, index=titles)

# Print the row for 'Anne Hathaway'
print(df.loc['Anne Hathaway'])

# Print the row for 'Denzel Washington'
print(df.loc['Denzel Washington'])

'''
If NMF is applied to documents then 
-NMF components represent topics
-NMF features combine topics into documents

If NMF is applied to images, NMF components are parts of images 
In this example, NMF decomposes images from an LCD display into the individual cell

Grayscale images
* "Grayscale" image = no colors, only shades of gray
* Measure pixel brightness
* Represent with value between 0 and 1 (0 is black)

Grayscale images as flat arrays
* Enumerate the entries
* Row by row
* From left to right, top to bottom

Encoding a collection of images
* Collection of images of the same size
* Encode as 2D array
* Each row corresponds to an image
* Each column corresponds to a pixel
* ...can apply NMF
'''

###NMF learns topics of documents
# Import pandas
import pandas as pd

# Create a DataFrame: components_df
components_df = pd.DataFrame(model.components_, columns=words)

# Print the shape of the DataFrame
print(components_df.shape)

# Select row 3: component
component = components_df.iloc[3,:]

# Print result of nlargest
print(component.nlargest())


###Explore the LED digits datasets
# Import pyplot
from matplotlib import pyplot as plt

# Select the 0th row: digit
digit = samples[0,:]

# Print digit
print(digit)

# Reshape digit to a 13x8 array: bitmap
bitmap = digit.reshape(13, 8)

# Print bitmap
print(bitmap)

# Use plt.imshow to display bitmap
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()


###NMF learns the parts of images 
# Import NMF
from sklearn.decomposition import NMF

# Create an NMF model: model
model = NMF(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)

# Assign the 0th row of features: digit_features
digit_features = features[0,:]

# Print digit_features
print(digit_features)


###PCA doesn't learn parts
# Import PCA
from sklearn.decomposition import PCA

# Create a PCA instance: model
model = PCA(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)


'''
Finding similar articles
* Engineer at a large online newspaper
* Task: recommend articles similar being read by customer
* Similar articles should have similar topics

Strategy 
* Apply NMF to the world-freuency array
* NMF feature values describe the topics
* ... so similar documents have similar NMF feature values
* Compare NMF feature values?

Apply NMF to the word-frequency array
* articles is a word frequency array

from sklearn.decomposition import NMF
nmf = NMF(n_components=6)
nmf_features = nmf.fit_transform(articles)

Similar documents have similar topics, but it isn't always the case that
the NMF feature values are exactly the same
For instance one version of a document might use very direct language
whereas another might interleave the same content with meaningless chatter

Versions of articles
* Different versions of the same document have same topic proportions
* ... exact feature values may be different
* E.g. because one version uses many meaningless words
* But all versions lie on the same line through the origin

The two articles are representted linearly on a graph and cosine similarity
can be used to compare the documents represented. Cosine similarity uses the angle
between the two lines

Cosine similarity
* Uses the angle between the lines
* Higher values means more similar
* Maximum value is 1, when angle is 0

from sklearn.preprocessing import normalize
norm_features = normalize(nmf_features)
# if has index 23
current_article = norm_features[23,:]
similarities = norm_features.dot(current_article)

DataFrames and labels
* Label similarities with the article titles, using a DataFrame
* Titles given as a list: titles

import pandas as pd 
norm_features = normalize(nmf_features)
df = pd.DataFrame(norm_features, index=titles)
current_article = df.loc['Dog bites man']
similarities = df.dot(current_article)
print(similarities.nlargest())
'''

###Which articles are similar to 'Cristiano Ronaldo'?
# Perform the necessary imports
import pandas as pd
from sklearn.preprocessing import normalize

# Normalize the NMF features: norm_features
norm_features = normalize(nmf_features)

# Create a DataFrame: df
df = pd.DataFrame(norm_features, index=titles)

# Select the row corresponding to 'Cristiano Ronaldo': article
article = df.loc['Cristiano Ronaldo']

# Compute the dot products: similarities
similarities = df.dot(article)

# Display those with the largest cosine similarity
print(similarities.nlargest())

###Recommend musical artists part II# Perform the necessary imports
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline

# Create a MaxAbsScaler: scaler
scaler = MaxAbsScaler()

# Create an NMF model: nmf
nmf = NMF(n_components=20)

# Create a Normalizer: normalizer
normalizer = Normalizer()

# Create a pipeline: pipeline
pipeline = make_pipeline(scaler, nmf, normalizer)

# Apply fit_transform to artists: norm_features
norm_features = pipeline.fit_transform(artists)


###Recommend musical artists part II
# Import pandas
import pandas as pd

# Create a DataFrame: df
df = pd.DataFrame(norm_features, artist_names)

# Select row of 'Bruce Springsteen': artist
artist = df.loc['Bruce Springsteen']

# Compute cosine similarities: similarities
similarities = df.dot(artist)

# Display those with highest cosine similarity
print(similarities.nlargest())
 