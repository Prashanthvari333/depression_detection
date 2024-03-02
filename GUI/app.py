from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('..//depression_model.pkl')

# Define thresholds and corresponding labels for depression stages
thresholds = {
    (0.0, 0.2): 'No depression',
    (0.2, 0.5): 'Mild depression',
    (0.5, 0.8): 'Moderate depression',
    (0.8, 1.0): 'Severe depression'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from form
    age=request.form['Age']
    feeling_sad= request.form['Feeling sad or Tearful']
    irritable= request.form['Irritable towards baby & partner']
    trouble_sleeping= request.form['Trouble sleeping at night']
    decision_problems = request.form['Problems concentrating or making decision']
    eating_problems = request.form['Overeating or loss of appetite']
    feeling_guilt = request.form['Feeling of guilt']
    bonding_problems = request.form['Problems of bonding with baby']
    suicide_attempt = request.form['Suicide attempt']

    # Preprocess the input data
    # Convert categorical features to numerical
    age_mapping = {'25-30':0,'30-35': 1, '35-40': 2, '40-45': 3, '45-50': 4}
    feeling_mapping = {'Yes': 2, 'No': 0, 'Sometimes': 1, 'Two or more days a week': 1, 'Often': 1,
                       'Not interested to say': 1, 'Maybe': 1}
    
    age = age_mapping.get(age)
    feeling_sad = feeling_mapping.get(feeling_sad)
    irritable = feeling_mapping.get(irritable)
    trouble_sleeping = feeling_mapping.get(trouble_sleeping)
    decision_problems = feeling_mapping.get(decision_problems)
    eating_problems = feeling_mapping.get(eating_problems)
    feeling_guilt = feeling_mapping.get(feeling_guilt)
    bonding_problems = feeling_mapping.get(bonding_problems)
    suicide_attempt = feeling_mapping.get(suicide_attempt)

    # Make predictions using the model
    #prediction = model.predict([[age, feeling_sad, irritable, trouble_sleeping, decision_problems, eating_problems, feeling_guilt, bonding_problems,suicide_attempt]])[0]
    input_data = [[age, feeling_sad, irritable, trouble_sleeping, decision_problems, eating_problems, feeling_guilt, bonding_problems,suicide_attempt]]
    # Make predictions using the model
    prediction = model.predict_proba(input_data)[0][1]  # Assuming binary classification with probabilities

    # Determine depression stage based on prediction score
    depression_stage = 'No depression'
    for (lower, upper), label in thresholds.items():
        if lower <= prediction < upper:
            depression_stage = label
            break
    # Static suggestions based on depression stage (temporary)
    suggestions = {
        'Severe depression': [
            {'title': 'Crisis Hotline', 'url': 'https://www.example.com/hotline'},
            {'title': 'Online Therapy', 'url': 'https://www.example.com/therapy'},
            {'title': 'Self-help Resources', 'url': 'https://www.example.com/self-help'}
        ],
        'Moderate depression': [
            {'title': 'Mental Health Forums', 'url': 'https://www.example.com/forums'},
            {'title': 'Relaxation Techniques', 'url': 'https://www.example.com/relaxation'}
        ],
        'Mild depression': [
            {'title': 'Daily Exercise', 'url': 'https://www.example.com/exercise'},
            {'title': 'Mindfulness Meditation', 'url': 'https://www.example.com/meditation'}
        ],
        'No depression': [
            {'title': 'Healthy Lifestyle Tips', 'url': 'https://www.example.com/health-tips'},
            {'title': 'Social Activities', 'url': 'https://www.example.com/social'}
        ]
    }

    # Static joke (temporary)
    joke = "Why did the scarecrow win an award? Because he was outstanding in his field!"

    # Return the result to the user along with suggestions and joke
    return render_template('results.html', depression_stage=depression_stage, suggestions=suggestions.get(depression_stage, []), joke=joke)

if __name__ == '__main__':
    app.run(debug=True)
    
a = 5/0