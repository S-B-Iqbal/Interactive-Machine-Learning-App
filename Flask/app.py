from flask import Flask, render_template, url_for, request
from flask_bootstrap import Bootstrap
from flask_uploads import UploadSet, configure_uploads, ALL, DATA
from werkzeug import secure_filename
app = Flask(__name__)

Bootstrap(app)

# Configuration
files = UploadSet('files', ALL)
app.config['UPLOADED_FILES_DEST'] = 'static/uploadstorage'
configure_uploads(app, files)

import os
import datetime
import time

# EDA Packages
import pandas as pd
import numpy as np

# ML Packages

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/datauploads', methods = ['GET', 'POST'] )
def datauploads():
    if request.method == 'POST' and 'csv_data' in request.files:
        file = request.files['csv_data']
        ## To upload a file
        filename = secure_filename(file.filename)
        ## To save a file
        file.save(os.path.join('static/uploadstorage', filename))
        fullfile = os.path.join('static/uploadstorage', filename)

        # DATE
        date = str(datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"))

        ## EDA Function
        df = pd.read_csv(os.path.join('static/uploadstorage', filename))
        df_size = df.size
        df_shape = df.shape
        df_columns = list(df.columns)
        df_targetname = df[df.columns[-1]].name
        df_featurenames = df_columns[0:-1]

        #Selecting all columns till last column
        df_XFeatures = df.iloc[:, 0:-1]
        #Selecting the last column as target
        df_YLabels = df[df.columns[-1]]


        ## Table
        df_table = df

        X = df_XFeatures
        Y = df_YLabels

        ## Model Building
        models = []

        models.append(('LR', LogisticRegression()))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC()))

        # Evaluate each model in turn
        results = []
        names = []
        allmodels = []
        scoring = 'accuracy'

        for name, model in models:
            kfold = model_selection.KFold(n_splits=10, random_state=666)
            cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            allmodels.append(msg)
            model_results = results
            model_names = names

    return render_template('details.html',
                           filename = filename,
                           dfplot= df,
                           date = date,
                           df_size = df_size,
                           df_shape = df_shape,
                           df_columns = df_columns,
                           df_targetname = df_targetname,
                           model_results = allmodels,
                           model_names = names,
                           fullfile = fullfile
                           )

if __name__ == '__main__':
	app.run(debug=True, port=12345)
