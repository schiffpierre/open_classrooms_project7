# Projet 7 Scoring Model
Project created as part of the OpenClassrooms Data Science track. This was project number 7 out of 8.

To run the dashboard, go to: https://credit-risk-dashboard.herokuapp.com/
The dashboard will run and provide you with different options (global or client-specific dashboard).
The global dashboard provides information related to the general dataset.
The client-specific dashboard allows you to input a client ID, retrieve a prediction, get an interpretation of this prediction 
and dynamically update feature values to see the impact on the default risk prediction.

The API from which the prediction is pulled is located here: https://home-credit-risk.herokuapp.com/
When navigating to this address, you should receive a message saying the app is up and running. 

List of files:
-api_request: File to test the connection of the API
-app: Script for the API deployed on Heroku
-dashboard: Script for running the dashboard
-note_methodologique: Document about the methodology for the project
-preprocess_model_train: Script for preprocessing the data and for training the models