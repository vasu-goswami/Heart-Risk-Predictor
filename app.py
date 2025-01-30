from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

app = Flask(__name__)

# Load the pre-trained model (ensure that the model is saved as 'heart_model.pkl')
model = pickle.load(open('heart_model.pkl', 'rb'))
print(type(model))  # Add this line to check the type of the model


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/calc', methods=['GET', 'POST'])
def calc():
    if request.method == 'POST':
        # Extract form data
        features = [
            float(request.form['age']),
            int(request.form['sex']),
            int(request.form['cp']),
            int(request.form['trestbps']),
            int(request.form['chol']),
            int(request.form['fbs']),
            int(request.form['restecg']),
            int(request.form['thalach']),
            int(request.form['exang']),
            float(request.form['oldpeak']),
            int(request.form['slope']),
            int(request.form['ca']),
            int(request.form['thal'])
        ]
        
        # Convert input data into a DataFrame for prediction
        input_data = pd.DataFrame([features], columns=[
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ])

        input_data=input_data.values.reshape(1,-1)
        
        # Make a prediction
        prediction = model.predict(input_data)
        
        # Show the result on the page
        result = "Heart Disease Risk" if prediction == 1 else "No Heart Disease"
        
        return render_template('index.html', result=result)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
