
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#reading the data
data = pd.read_csv('creditcard.csv')

data=data.sample(frac=0.1, random_state=1)  # we cut the data to 10% to make it more manageable

# plot histogram for each parameter
data.hist(figsize=(20,20))


fraud=data[data['Class']==1]
valid=data[data['Class']==0]
outlier_fraction= len(fraud)/float(len(valid))



#corelation matrix
cormat=data.corr()
fig=plt.figure(figsize=(12,9))
sns.heatmap(cormat,vmax=.8,square=True)
plt.show()

# get all the columns fromm the data frame
columns=data.columns.tolist()

#filter the columns to remove the data we do not want
columns=[c for c in columns if c not in ["Class"]]

# store the variable we will be predicting on
target="Class"

x=data[columns]
y=data[target]


#define a random state

state=1
 #define the outlier detectoin method

classifiers={
    "Isolation Forest": IsolationForest(max_samples=len(x),
                                        contamination=outlier_fraction,
                                        random_state=state),
    "Local Outlier Factor":LocalOutlierFactor(
        n_neighbors=20,
        contamination=outlier_fraction
    )
}

#fit the model

n_outliers=len(fraud)

for i, (clf_name,clf) in enumerate(classifiers.items()):
    #fit the data and tag the outliers
    if clf_name=="Local Outlier Factor":
        y_pred=clf.fit_predict(x)
        scores_pred=clf.negative_outlier_factor_
    else:
        clf.fit(x)
        scores_pred=clf.decision_function(x)
        y_pred=clf.predict(x)
# Reshape the prediction value to 0 for valid and 1 for fraud

y_pred[y_pred==1]=0
y_pred[y_pred==-1]=1

n_errors=(y_pred!=y).sum()

print(f'{clf_name},{n_errors}')
print(accuracy_score(y,y_pred))
print(classification_report(y,y_pred))