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
### Insufficient Quantity of Training Data
- Data is very important since most algorithms usually need thousands of example to train on before having meaningful results
- Study showed that very different ML algos performed reasonably the same when given enough data

### Nonrepresentative Training Data
- The data you want to train on should be representative of the new examples you want the model to predict
- Too small samples can cause "sampling noise" where data in nonrepresentative by chance
- Larger sample can have flawed sampling methods and still be nonrepresentative which is known as "sampling bias"

### Poor-Quality Data
- Bad data can happen as a result of outliers or missing attributes, so you must decided whether to drop these outlier and attributes

### Irrevelant Features
- Make sure your training data has relevant features
- Feature Engineering is determining which features are good for training on and involves:
	- "Feature Selection" is deciding the most useful features to train on in the data
	- "Feature Extraction" is combining existing features to produce a more useful feature
	- Creating new features by obtaining new data

### Overfitting the Training Data
- "Overfitting" is when the model performs well on the training data but fails to generalize to non-training data
- Overfitting can happen when a model is too complex relative to the amount and quality of the data, you can reduce overfitting by:
	- Gathering more data
	- Simplify the model with less parameters and a simpler model
	- Reduce noise in data and improve quality by removing outliers, errors, etc
- "Regularization" is the process of making a model simpler through contraints to reduce chance of overfitting
- "Hyper-parameter" is a parameter of a learning algorithm that control how much regularization to apply

### Underfitting the Training Data
- "Underfitting" is when the model is too simple to learn the underlying structure of the data
- Can help prevent underfitting by:
	- More powerful model
	- Add features to your data
	- Reducing hyper-parameter for less regularization

## Testing and Validating
- An option for testing is to split the data into "training set" and "test set", so after model is done training on training set you can test out new cases with the test set
- "Generalization Error" is the error rate of the model for new cases
	- If the training error rate is low on the training set but the generalization error is high then it means the model is overfitted to the data

## Hyperparameter Tuning and Model Selection
- Comparing generalization errors can be a good way to determine which model to use
- A possible scenario is that you create a model that performs well on the training set but performs worse on the test set. You could modify the hyperparameter to get a lower error rate for the test set but this can lead to overfitting on the test set
	- Common solution is to use a validation set:
		- Set aside part of the training set as the validation set
		- Train model and fine tune hyperparameters on the reduce training set(original training set - validation set)
		- Select model that performs best on validation set
		- Finally retrain the model on the entire training set(reduced training set + validation set)
- "Cross-Validation" is having multiple smaller validation sets and evaluating the model against each one and averaging out the results

## Data Mismatch
- The Data the model trains on should be representative of the data you expect to see in production


