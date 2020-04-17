from flask import Flask, render_template, request, redirect, url_for
from sklearn.preprocessing import MinMaxScaler
# from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
breast_cancer_detection = pd.read_csv('dataR2.csv')
X = breast_cancer_detection[['Age', 'BMI', 'Glucose', 'Insulin',
        'HOMA','Leptin', 'Adiponectin', 'Resistin', 'MCP.1']]
y = breast_cancer_detection['Classification']
scaler = MinMaxScaler()
scaler.fit(X)
scaler.transform(X)
model = SVC()
model.fit(X, y)
# print(model.score(X, y))
cancer_stage = pd.read_csv('breast-cancer-wisconsin.csv')
cancer_stage_X = cancer_stage[['ClumpThickness', ' UniformityofCellSize', 'UniformityCellShape',
       'MarginalAdhesion', 'SingleEpithelialCellSize', 'BareNuclei',
       'BlandChromatin', 'NormalNucleoli', 'Mitoses']]
cancer_stage_y = cancer_stage[' Class']
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(cancer_stage_X, cancer_stage_y)
from sklearn.ensemble import RandomForestRegressor
insurance_data = pd.read_csv('policy.csv')
insurance_data_X = insurance_data[['age', 'incomepm', 'cancerstages', 'Smoking']]
insurance_data_y = insurance_data['insurance']
rfr = RandomForestRegressor()
rfr.fit(insurance_data_X, insurance_data_y)
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def barebones():
    if request.method == 'POST':
        age = request.form['age']
        bmi = request.form['bmi']
        glucose = request.form['glucose']
        insulin = request.form['insulin']
        homa = request.form['homa']
        leptin = request.form['leptin']
        adiponectin = request.form['adiponectin']
        resistin = request.form['resistin']
        mcp = request.form['mcp']
        y_pred = [[age, bmi, glucose, insulin,
                   homa, leptin, adiponectin, resistin, mcp]]
        y_pred = scaler.transform(y_pred)
        preds = model.predict(y_pred)
        free = "You're free from becoming a pray of cancer. God bless!"
        pray = "Sorry. You're a victim of cancer."
        if preds == 2:
            return render_template('index.html', p=pray)
        elif preds == 1:
            return render_template('index.html', f=free)


    return render_template('index.html')


@app.route('/stage', methods=['GET', 'POST'])
def stage():
    if request.method == 'POST':
        return render_template('stage.html')
    return render_template('index.html')

@app.route('/stages', methods=['GET', 'POST'])
def stages():
    if request.method == 'POST':
        ClumpThickness = request.form['ClumpThickness']
        UniformityofCellSize = request.form['UniformityofCellSize']
        UniformityCellShape = request.form['UniformityCellShape']
        MarginalAdhesion = request.form['MarginalAdhesion']
        SingleEpithelialCellSize = request.form['SingleEpithelialCellSize']
        BareNuclei = request.form['BareNuclei']
        BlandChromatin = request.form['BlandChromatin']
        NormalNucleoli = request.form['NormalNucleoli']
        Mitoses = request.form['Mitoses']
        y_pred = [[ClumpThickness, UniformityofCellSize, UniformityCellShape,
                    MarginalAdhesion, SingleEpithelialCellSize, BareNuclei, BlandChromatin,
                    NormalNucleoli, Mitoses]]
        preds = dtc.predict(y_pred)
        benign = "You won't need surgery. Insurance is not much necessary."
        malignant = "Sorry to say, but you'll need surgery. Please claim insurance if needed."
        if preds == 2:
            return render_template('stage.html', b=benign)
        elif preds == 4:
            return render_template('stage.html', m=malignant)
    return render_template('stage.html')

@app.route('/insurance', methods=['GET', 'POST'])
def insurance():
    if request.method == 'POST':
        submit = request.form['submit']
        if submit == 'claim':
            return render_template('insurance.html')
        elif submit == 'no':
            return redirect(url_for('barebones'))
    return render_template('stage.html')

@app.route('/insurances', methods=['GET', 'POST'])
def insurances():
    if request.method == 'POST':
        age = request.form['age']
        income = request.form['incomepm']
        cancerstage = request.form['cancerstages']
        if cancerstage == 'one':
            stage = 1
        elif cancerstage == 'two':
            stage = 2
        smoking = request.form['smoking']
        if smoking == 'yes':
            consumption = 1
        elif smoking == 'no':
            consumption = 0
        y_pred = [[age, income, stage, consumption]]
        preds = rfr.predict(y_pred)
        # preds = str(preds)
        return render_template('insurance.html', i=preds)
        
    return render_template('insurance.html')


@app.route('/reverts', methods=['GET', 'POST'])
def reverts():
    if request.method == 'POST':
        submit = request.form['submit']
        if submit == 'try':
            return redirect(url_for('insurances'))
        elif submit == 'home':
            return redirect(url_for('barebones'))
    return render_template('insurance.html')

if __name__ == '__main__':
    app.run(debug=True)
