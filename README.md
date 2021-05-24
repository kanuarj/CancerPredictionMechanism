# Insurance Claiming Prognostication based on current stage of Breast Cancer

This repository contains improvements in the prognostication mechanism as well as claiming insurance automatically based on the stage of the cancer.
The implementation is divided prominently into 3 major modules. Breast Cancer Detection, Breast Cancer Stage Detection, Insurance Claiming.
The models are implmented using <a href="https://github.com/scikit-learn/scikit-learn">Sklearn</a> and Deployed in Web Based Application using <a href="https://github.com/pallets/flask">Flask</a>.

## Datasets Used
<a href="https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra">Breast Cancer Detection</a>.</br>
<a href="https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)">Breast Cancer Stage Detection</a>.</br>
Insurance Claiming ~ Custom made dummy dataset.

## Using this Repo

1. Clone this repository
```
git clone https://github.com/kanuarj/CancerPredictionMechanism.git
cd CancerPredictionMechanism
```
2. Install required dependencies
```
pip install pandas scikit-learn flask seaborn matplotlib
```
3. Run the Application
```
py app.py
```
> Note that `py app.py` specifically works for Windows. If you're using any other systems, please use `python app.py`

## Understanding Files
`app.py` is Flask Deployment of Models.<br>
`model.py` is Models implemented in joblib.<br>
`datasets` folder has all the required datasets for this projects.<br>
`models` folder all joblib models.<br>
`notebooks` folder has all the Jupyter Notebooks of ML Implementations.<br>
`templates` folder has all the Front end of the Flask Application.<br>
`visuals.py` file has some experimental visualizations.<br>

<hr>
Catch me on <a href="https://www.youtube.com/c/RaunakJoshi">YouTube</a> for some exciting videos on ML and Flask. Peace.
