from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load models
log_model = joblib.load('logistic_regression_model.pkl')
tree_model = joblib.load('decision_tree_model.pkl')
forest_model = joblib.load('random_forest_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form['feature1']), float(request.form['feature2'])]
    # Add more lines to retrieve other feature values

    # Make predictions
    log_pred = int(log_model.predict([features])[0])
    tree_pred = int(tree_model.predict([features])[0])
    forest_pred = int(forest_model.predict([features])[0])

    result = {
        'logistic_regression': log_pred,
        'decision_tree': tree_pred,
        'random_forest': forest_pred
    }

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
