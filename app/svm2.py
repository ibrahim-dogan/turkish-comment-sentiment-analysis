# Importing the libraries
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Importing the dataset
dataset = pd.read_csv('train_loan.csv', sep=';')
y = dataset.iloc[:, 12].values  # loan status

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

dataset = pd.get_dummies(dataset, columns=['Gender', 'Education', 'Property_Area', 'Married', 'Self_Employed'])
dataset = dataset.drop(columns=['Loan_ID'])
dataset = dataset.drop(columns=['Loan_Status'])

X = dataset.iloc[:, 0:17].values
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
classifier = SVC(random_state=0)
classifier.fit(X_train, y_train)

# Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Fitting Decision Tree Classification to the Training set
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


def preprocessing(row):
    dataset_x = pd.read_csv('train_loan.csv', sep=';')
    dataset_x = dataset_x.append(row, ignore_index=True)

    dataset_x = pd.get_dummies(dataset_x, columns=['Gender', 'Education', 'Property_Area', 'Married', 'Self_Employed'])
    dataset_x = dataset_x.drop(columns=['Loan_ID'])
    dataset_x = dataset_x.drop(columns=['Loan_Status'])

    dataset_x = dataset_x.tail(1)
    print(dataset_x.columns)
    scaler_new_data = sc.transform(dataset_x)
    result = classifier.predict(scaler_new_data)
    return result


row = {'Gender': 'Female',
       'Married': 'Yes',
       'Dependents': 3,
       'Education': 'Not Graduate',
       'Self_Employed': 'No',
       'ApplicantIncome': 1000,
       'CoapplicantIncome': 0,
       'LoanAmount': 50,
       'Loan_Amount_Term': 240,
       'Credit_History': 1,
       'Property_Area': 'Rural'}
print(preprocessing(row))
