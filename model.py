from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import pandas as pd
from joblib import dump

breast_cancer_detection = pd.read_csv('datasets/dataR2.csv')
X = breast_cancer_detection[['Age', 'BMI', 'Glucose', 'Insulin',
        'HOMA','Leptin', 'Adiponectin', 'Resistin', 'MCP.1']]
y = breast_cancer_detection['Classification']
scaler = MinMaxScaler()
scaler.fit(X)
scaler.transform(X)
model = SVC()
model.fit(X, y)
dump(model, 'models/cancer_classification.joblib')


cancer_stage = pd.read_csv('datasets/breast-cancer-wisconsin.csv')
cancer_stage_X = cancer_stage[['ClumpThickness', ' UniformityofCellSize', 'UniformityCellShape',
       'MarginalAdhesion', 'SingleEpithelialCellSize', 'BareNuclei',
       'BlandChromatin', 'NormalNucleoli', 'Mitoses']]
cancer_stage_y = cancer_stage[' Class']
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(cancer_stage_X, cancer_stage_y)
dump(dtc, 'models/cancer_stage.joblib')


from sklearn.ensemble import RandomForestRegressor
insurance_data = pd.read_csv('datasets/policy.csv')
insurance_data_X = insurance_data[['age', 'incomepm', 'cancerstages', 'Smoking']]
insurance_data_y = insurance_data['insurance']
rfr = RandomForestRegressor()
rfr.fit(insurance_data_X, insurance_data_y)
dump(rfr, 'models/insurance_claiming.joblib')