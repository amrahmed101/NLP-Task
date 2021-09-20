#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    str_features = request.form.values()
    prediction = model.predict(str_features)

    output = prediction

    return render_template('index.html', prediction_text='The Corresponding Industry Should Be {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)

