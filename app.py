from flask import Flask, render_template, request, redirect, url_for
from joblib import load
import numpy as np
model = load('models/cancer_classification.joblib')
dtc = load('models/cancer_stage.joblib')
rfr = load('models/insurance_claiming.joblib')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def barebones():
    if request.method == 'POST':
        X_values = [float(x) for x in request.form.values()]
        y_pred = [np.array(X_values)]
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
        X_values = [float(x) for x in request.form.values()]
        y_pred = [np.array(X_values)]
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
