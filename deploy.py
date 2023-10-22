import os
from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)

# Define the absolute paths to the model and scaler files
model_path = "rf_model.pkl"
scaler_path = "scaled_data.pkl"

# Load the trained model
model = pickle.load(open(model_path, 'rb')) 

# Load the trained StandardScaler
scaler = pickle.load(open(scaler_path, 'rb')) 

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect user input
    user_input = [float(request.form['value_eur']), 
                  float(request.form['release_clause_eur']), 
                  float(request.form['potential']), 
                  float(request.form['movement_reactions']),
                  float(request.form['age'])]
    
    feature_names = ['value_eur', 'release_clause_eur', 'potential', 'movement_reactions', 'age']
    scaler.feature_names_in_ = feature_names

    # Transform user input using the loaded scaler
    scaled_input = scaler.transform([user_input])

    # Make predictions using the scaled input
    prediction = model.predict(scaled_input)
    
    #reading the csv file and taking ypred and ytest
    dataframe = pd.read_csv('Ytest_and_y_pred.csv')
    y_test = dataframe['Ytest'].values
    y_pred = dataframe['y_pred'].values

    #Calculating the mean_absolute_error
    mae = mean_absolute_error(y_test,y_pred)
    formatted_mae = "{:.2f}".format(mae) # Format the mae to two decimal places
    
    # Render the prediction and confidence score
    return render_template('result.html',mae=formatted_mae, prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)