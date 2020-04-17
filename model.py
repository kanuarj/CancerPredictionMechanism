import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
from flask import *
import numpy as np 
df = pd.read_csv('dataR2.csv')


X = df[['Age', 'BMI', 'Glucose', 'Insulin',
        'HOMA', 'Adiponectin', 'Resistin', 'MCP.1']]
y = df['Classification']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=100, test_size=0.1)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

bc = pd.read_csv('breast-cancer-wisconsin.csv')
Xbc = bc[['ClumpThickness', ' UniformityofCellSize', 
                  'MarginalAdhesion', 'SingleEpithelialCellSize', 'BareNuclei',
                  'BlandChromatin', 'NormalNucleoli']]
ybc = bc[' Class']
lr = DecisionTreeClassifier()
Xbc_train, Xbc_test, ybc_train, ybc_test = train_test_split(
Xbc, ybc, test_size=0.2)
lr.fit(Xbc_train, ybc_train)

linearReg = LinearRegression()

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
ins = pd.read_csv('policy.csv')
Xin = ins[['age', 'incomepm', 'cancerstages', 'Smoking']]
yin = ins['insurance']
rfr.fit(Xin, yin)

rfcc = pd.read_csv('risk_factors_cervical_cancer.csv')
Xrfcc = rfcc[['Age', 'pregnancies', 'Smokes', 'HormonalContraceptives']]
yrfcc = rfcc['Cancer']
Xrfcc_train, Xrfcc_test, yrfcc_train, yrfcc_test = train_test_split(Xrfcc, yrfcc, test_size=0.1)
rj = DecisionTreeClassifier()
rj.fit(Xrfcc_train, yrfcc_train)



app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def cancer():
    if request.method == 'POST':
        age = request.form['age']
        bmi = request.form['bmi']
        glucose = request.form['glucose']
        insulin = request.form['insulin']
        homa = request.form['homa']
        # leptin = request.form['leptin']
        adiponectin = request.form['adi']
        resistin = request.form['resistin']
        mcp = request.form['mcp']
        pred_vals = [[age, bmi, glucose, insulin, homa, adiponectin, resistin, mcp]]
        print(pred_vals)
        y_pred = model.predict(pred_vals)
        free = "You're free from becoming a pray of cancer. God bless!"
        if y_pred == 2:
            return redirect(url_for('breastcancermt'))
        else:
            
            return render_template('index.html', f=free)
    return render_template('index.html')

@app.route('/breastcancermt', methods=['GET', 'POST'])
def breastcancermt():
    if request.method == 'POST':
        ClumpThickness = request.form['ClumpThickness']
        UniformityofCellSize = request.form['UniformityofCellSize']
        MarginalAdhesion = request.form['MarginalAdhesion']
        SingleEpithelialCellSize = request.form['SingleEpithelialCellSize']
        BareNuclei = request.form['BareNuclei']
        BlandChromatin = request.form['BlandChromatin']
        NormalNucleoli = request.form['NormalNucleoli']
        
        
        pred_val_bc = [[ClumpThickness, UniformityofCellSize,
                   MarginalAdhesion, SingleEpithelialCellSize, BareNuclei,
                   BlandChromatin, NormalNucleoli]]
        print(pred_val_bc)
        y_preds = lr.predict(pred_val_bc)
        if y_preds == 2:
            return render_template('breastcancermt.html', b='Benign')
        else:
            return render_template('breastcancermt.html', m='Malignant')



    return render_template('breastcancermt.html')

@app.route('/insurance', methods=['GET', 'POST'])
def insurance():
    if request.method == 'POST':
        age = request.form['age']
        income = request.form['ipm']
        cancerstages = request.form['cs']
        smoking = request.form['smoking']

        rfr_test = [[age, income, cancerstages, smoking]]
        rfr_pred = rfr.predict(rfr_test)
        rfr_preds = str(rfr_pred)
        return render_template('insurance.html', r=rfr_preds)



    return render_template('insurance.html')

@app.route('/cervical', methods=['GET', 'POST'])
def cervical():
    if request.method == 'POST':
        age = request.form['age']
        pregnancies = request.form['pregnancies']
        Smokes = request.form['smokes']
        HormonalContraceptives = request.form['HormonalContraceptives']
        rfcc_preds = [[age, pregnancies, Smokes, HormonalContraceptives]]
        rj_preds = rj.predict(rfcc_preds)
        cervicalcancer = "You're free from cancer. God Bless!"
        if rj_preds == 0:
            return render_template('index.html', cc=cervicalcancer)       
        else:
            return redirect(url_for('barebones'))
    return render_template('index.html')


@app.route('/barebones', methods=['GET', 'POST'])
def barebones():
    if request.method == 'POST':
        if request.form['bareb'] == 'takeit':
            return redirect(url_for('insurance'))
        else:
            return render_template('index.html')
    return render_template('barebones.html')


if __name__ == '__main__':
    app.run(debug=True)


