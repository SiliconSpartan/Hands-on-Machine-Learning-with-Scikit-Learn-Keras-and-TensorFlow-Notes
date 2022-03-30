# What is Machine Learning?
- ML is programming computers so that they learn from data
- Examples the system uses to learn are the training set
	- Each example of a training set is called training instance

# Why use Machine Learning?
- ML is helpful to write programs that need to learn over time in contrast to the normal programming model of consistently adding more code to accommodate new cases
- ML is also good for problems where there is no known solution or only have traditional complex approaches
- ML is also good for identifying patterns and helping humans learn more about a problem
	- Applying ML techniques to large amount of data to discover patterns is known as Data Mining

# Types of Machine Learning Systems
## Supervised/Unsupervised Learning
### Supervised Learning
- Training data that is fed to algorithm includes the desired solutions known as Labels
- Classification is an example of supervised task since it assigns a label to data points
- Predicting a numeric value is another common task and is known as Regression
- Features are Columns of data
- Logistic Regression is an example of regression and classification
	- ex: return a percentage for how likely a data point belongs to a certain class
- Examples of Supervised Learning Algorithms
	- k-Nearest Neighbors
	- Linear Regression
	- Logistic Regression
	- Support Vector Machines
	- Decisions Trees and Random Forests
	- Neural Networks

### Unsupervised Learning
- Data is Unlabeled and system tries to learn on its own
- Examples on Unsupervised Algorithms
	- Clustering
		- K-Means
		- DBSCAN
		- Hierarchical Cluster Analysis (HCA)
	- Anomaly detection and novelty detection
		- One-Class SVM
		- Isolation Forest
	- Visualization and Dimensionality Reduction
		- Principal Component Analysis
		- Kernel PCA
		- Locally-Linear Embedding
		- t-distributed Stochastic Neighbor Embedding
	- Association rule learning
		- Apriori
		- Eclat
- Clustering algorithms can be good for creating groups from data
- Visualization algorithms can take unlabeled data and output 2D/3D representations that preserve cluster structure
- Dimensionality Reduction is meant to simplify the data structure without losing too much information
	- Feature extraction is to merge closely correlated features together in order to reduce dimensionality
- Anomaly detection are algorithms that are shown "normal" data and used to identify and usually remove outliers
- Novelty detection algorithms are similar except that they expect to only see normal data during training unlike anomaly detection which can handle some outliers in the training data
- Association Rule Learning is digging into large amounts of data to find relations between different features

### Semisupervised Learning
- Semisupervised algorithms can deal with partially labeled data and lots of unlabeled data
- Semisupervised are usually a mix of supervised and unsupervised algorithms
- Ex. Deep Belief Networks are actually	restricted Boltzmann machines trained sequentially in an unsupervised manner and are tweaked by supervised learning methods

### Reinforcement Learning
- In RL the learning systems are called agents and they observe the environment, select and perform actions, then receive rewards or penalties. After all that it will learn by itself what the best strategy, also known as policy, is to obtain the most reward

## Batch and Online Learning
### Batch Learning
- System is incapable of learning incrementally and needs to be trained on all the data as once, also known as offline learning
- If there is new data then you will have to retrain the model on all the new data + the old data, which can take lots of compute resources and time

### Online Learning
- System is capable of learning incrementally in a fast and cheap way by feeding in data instances sequentially or in mini-batches
- Good if you are limited on compute resources since you can throw away the data after you train on it
- Useful if you can't load all the data in memory since you can load the data in small batches and then run some process on it, also known as "out-of-core learning"
- Contrary to the name "Online Learning" it is usually done offline
- Learning Rate is how fast the system should adapt to new data -- higher learning rate means the system will adapt more to new data and lower rates means the system won't change as much as to the new data
- Major risk with Online Learning is that if bad data is fed to the system it can degrade the performance of the model over time and you may have to revert to an earlier working state if there is a drop in performance

## Instance-Based Versus Model-Based Learning
- ML Systems can also be classified by how they "genrealize", which is how the system makes prediction on data it hasn't seen before
### Instance Based Learning
- Instance Based Learning using a "measure of simalirty" to generalize new examples, as in it looks at the new example and compares how similar it is to other example data
- Ex. For spam email you can compare the word count of new emails that come in to emails that are classified as spam

### Model Based Learning
- Model Based Learning is a generalization method to build a model based off the known examples and use that model to make predictions
- When evaluating your model you use a "utility/fitness function" to determine how good the model is or a "cost function" to determine how bad a model is
- Training is the process of feeding data to a model and it will work on optimizing the parameters for that model

## Main Challenges of Machine Learning

