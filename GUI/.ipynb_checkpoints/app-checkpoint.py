from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from form
    # Process the input data
    # Make predictions using the model
    # Return the result to the user
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
