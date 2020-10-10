# Machine-learning-projects
# Credit card Fraud Detection

Credit Card Fraud Detection using Machine Learning


				INTRODUCTION

'Fraud' in credit card transactions is unauthorized and unwanted usage of an account by someone other than the owner of that account. Necessary prevention measures can be taken to stop this abuse and the behaviour of such fraudulent practices can be studied to minimize it and protect against similar occurrences in the future.In other words, Credit Card Fraud can be defined as a case where a person uses someone else’s credit card for personal reasons while the owner and the card issuing authorities are unaware of the fact that the card is being used.
Fraud detection involves monitoring the activities of populations of users in order to estimate, perceive or avoid objectionable behaviour, which consist of fraud, intrusion, and defaulting.

Some of the currently used approaches to detection of such fraud are:
•	Artificial Neural Network
•	Fuzzy Logic
•	Genetic Algorithm
•	Logistic Regression
•	Decision tree
•	Support Vector Machines
•	Bayesian Networks
•	Hidden Markov Model
•	K-Nearest Neighbour





Methodology in my project

First of all, we obtained our dataset from Kaggle, a data analysis website which provides datasets.
Inside this dataset, there are 31 columns out of which 28 are named as v1-v28 to protect sensitive data.
The other columns represent Time, Amount and Class. Time shows the time gap between the first transaction and the following one. Amount is the amount of money transacted. Class 0 represents a valid transaction and 1 represents a fraudulent one.
We plot different graphs to check for inconsistencies in the dataset and to visually comprehend it.
After checking this dataset, we plot a histogram for every column. This is done to get a graphical representation of the dataset which can be used to verify that there are no missing
After this analysis, we plot a heatmap to get a coloured representation of the data and to study the correlation between out predicting variables and the class variable. This heatmap is shown below:

 
The dataset is now formatted and processed. The time and amount column are standardized and the Class column is removed to ensure fairness of evaluation. The data is processed by a set of algorithms from modules.
The following module diagram explains how these algorithms work together: This data is fit into a model and the following outlier detection modules are applied on it:
1)	Local Outlier Factor
2)	Isolation Forest Algorithm
These algorithms are a part of sklearn. The ensemble module in the sklearn package includes ensemble-based methods and functions for the classification, regression and outlier detection.
This free and open-source Python library is built using NumPy, SciPy and matplotlib modules which provides a lot of simple and efficient tools which can be used for data analysis

A)	Local Outlier Factor

It is an Unsupervised Outlier Detection algorithm. 'Local Outlier Factor' refers to the anomaly score of each sample. It measures the local deviation of the sample data with respect to its neighbours.
More precisely, locality is given by k-nearest neighbours, whose distance is used to estimate the local data.
By comparing the local values of a sample to that of its neighbours, one can identify samples that are substantially lower than their neighbours. These values are quite amanous and they are considered as outliers.
As the dataset is very large, we used only a fraction of it in out tests to reduce processing times.
The final result with the complete dataset processed is also determined and is given in the results section of this paper.

B)	Isolation Forest Algorithm

The Isolation Forest ‘isolates’ observations by arbitrarily selecting a feature and then randomly selecting a split value between the maximum and minimum values of the designated feature.
Recursive partitioning can be represented by a tree, the number of splits required to isolate a sample is equivalent to the path length root node to terminating node.
The average of this path length gives a measure of normality and the decision function which we use.

RESULTS

The code prints out the number of false positives it detected and compares it with the actual values. This is used to calculate the accuracy score and precision of the algorithms.
The fraction of data we used for faster testing is 10% of the entire dataset. The complete dataset is also used at the end and both the results are printed.
These results along with the classification report for each algorithm is given in the output as follows, where class 0 means the transaction was determined to be valid and 1 means it was determined as a fraud transaction.
Results when 10% of the dataset is used:

 

CONCLUSION

While the algorithm does reach over 99.6% accuracy, its precision remains only at 28% when a tenth of the data set is taken into consideration. However, when the entire dataset is fed into the algorithm, the precision rises to 33%. This high percentage of accuracy is to be expected due to the huge imbalance between the number of valid and number of genuine transactions.
