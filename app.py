from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler from the pickle files
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


def format_duration_in_minutes(duration):
    parts = duration.split()
    total_minutes = 0
    for part in parts:
        if 'h' in part:
            total_minutes += int(part.replace('h', '')) * 60
        elif 'min' in part:
            total_minutes += int(part.replace('min', ''))
    return total_minutes


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        budget = float(request.form['budget'])
        total_time = format_duration_in_minutes(request.form['total_time'])
        genres = request.form['genres']

        # Convert genres to binary representation based on the available genres in the dataset
        genres = [1 if genre in genres else 0 for genre in available_genres]

        # Create the input array for prediction
        input_data = np.array([[budget, total_time] + genres], dtype='float32')

        # Scale the input data using the same scaler used during training
        input_data[:, :2] = scaler.transform(input_data[:, :2])

        # Make the prediction
        prediction = model.predict(input_data)[0][0]

        # Convert the predicted revenue back to real money value
        prediction = scaler.inverse_transform([[0, prediction]])[0][1]

        return render_template('result.html', prediction=prediction)
    except Exception as e:
        return render_template('error.html', error=str(e))


if __name__ == '__main__':
    # Get the available genres from the dataset
    available_genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy',
                        'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie',
                        'Thriller', 'War', 'Western']

    app.run(debug=True)
