import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

# Read the CSV file
data = pd.read_csv("movie_data_with_preprocessing.csv")
pd.set_option("display.max_columns", None)
# print(data)
print(data.columns)

def time_to_minutes(time_str):
    time_parts = time_str.split()
    total_minutes = 0
    for part in time_parts:
        if 'h' in part:
            total_minutes += int(part[:-1]) * 60
        elif 'min' in part:
            total_minutes += int(part[:-3])
    return total_minutes

data['Total Time'] = data['Total Time'].astype(str)  # Convert 'Total Time' to string type
data['Total Time'] = data['Total Time'].apply(time_to_minutes)

# Extract features and target
X = data[['Budget', 'Total Time', 'Action', 'Adventure',
       'Animation', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy', 'History',
       'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie',
       'Thriller', 'War', 'Western']].values.astype('float32')
y = data['Revenue'].values.astype('float32')

# Normalize the numerical features (budget and time) for better training performance
scaler = StandardScaler()
X[:, :2] = scaler.fit_transform(X[:, :2])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the learning rate range
start_lr = 1e-8
end_lr = 1e-2
epochs = 200
batch_size = 64

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)), #input layer with 64 neurons
    tf.keras.layers.Dense(32, activation='relu'), #hidden layer with 32 neurons
    tf.keras.layers.Dense(1)  # Output layer with a single neuron for regression
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Save the initial weights of the model
initial_weights = model.get_weights()

# Perform the learning rate range test
def learning_rate_range_test(model, X, y, start_lr, end_lr, epochs, batch_size):
    num_batches = len(X) // batch_size
    lr_multiplier = (end_lr / start_lr) ** (1 / num_batches)
    model.set_weights(initial_weights)

    lr = start_lr
    best_lr = start_lr
    best_loss = float('inf')
    losses = []
    for epoch in range(epochs):
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            loss = model.train_on_batch(X_batch, y_batch)[0]  # Select the first element (loss) from the output
            losses.append(loss)

            if loss < best_loss:
                best_loss = loss
                best_lr = lr

            lr *= lr_multiplier

    return best_lr, losses

# After performing the learning rate range test:
lr_rates, losses = learning_rate_range_test(model, X_train, y_train, start_lr, end_lr, epochs, batch_size)

# Create an array of learning rates that corresponds to each loss value
lr_rates_arr = np.geomspace(start_lr, end_lr, len(losses))

# Plot the learning rate vs. loss curve
plt.figure(figsize=(10, 6))
plt.plot(lr_rates_arr, losses)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Learning Rate Range Test')
plt.grid(True)
plt.show()

# Find the index of the minimum loss
min_loss_index = np.argmin(losses)

# Get the corresponding learning rate value at the minimum loss index
best_lr = lr_rates_arr[min_loss_index]

# Print the best learning rate value to the console
print("Best Learning Rate:", best_lr)

# Use the best learning rate to compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_lr), loss='mse', metrics=['mae'])

# Hyperparameter Tuning
def build_model(hidden_layers=1, units_per_layer=32, dropout_rate=0.0, l1_regularizer=0.0, l2_regularizer=0.0):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units_per_layer, activation='relu', input_shape=(X_train.shape[1],)))

    # Add hidden layers
    for _ in range(hidden_layers):
        model.add(tf.keras.layers.Dense(units_per_layer, activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_regularizer,
                                                                                       l2=l2_regularizer)))
        # Apply dropout
        model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(1))  # Output layer with a single neuron for regression
    return model

# Learning Rate Scheduling: Instead of using a fixed learning rate, try using a learning rate schedule that adjusts the learning rate during training.
def lr_schedule(epoch, lr):
    if epoch < 100:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Use callbacks for learning rate scheduling and early stopping
callbacks = [
    tf.keras.callbacks.LearningRateScheduler(lr_schedule),
    tf.keras.callbacks.EarlyStopping(patience=80, restore_best_weights=True)
]
best_hidden_layers = None
best_units_per_layer = None
best_dropout_rate = None
best_l1_regularizer = None
best_l2_regularizer = None
best_mae = float('inf')
best_model = None


# Hyperparameter Tuning
hidden_layers_list = [1, 2, 3]  # Vary the number of hidden layers
units_per_layer_list = [32, 64, 128]  # Vary the number of units per layer
dropout_rate_list = [0.0, 0.2, 0.4]  # Vary the dropout rate
l1_regularizer_list = [0.0, 0.001, 0.01]  # Vary the L1 regularization strength
l2_regularizer_list = [0.0, 0.001, 0.01]  # Vary the L2 regularization strength

for hidden_layers in hidden_layers_list:
    for units_per_layer in units_per_layer_list:
        for dropout_rate in dropout_rate_list:
            for l1_regularizer in l1_regularizer_list:
                for l2_regularizer in l2_regularizer_list:
                    # Build the model with current hyperparameters
                    model = build_model(hidden_layers=hidden_layers, units_per_layer=units_per_layer,
                                        dropout_rate=dropout_rate, l1_regularizer=l1_regularizer,
                                        l2_regularizer=l2_regularizer)

                    # Compile the model with the best learning rate found earlier
                    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_lr),
                                  loss='mse', metrics=['mae'])

                    # Train the model with callbacks for learning rate scheduling and early stopping
                    history = model.fit(X_train, y_train, epochs=200, batch_size=64, validation_split=0.1,
                                        callbacks=callbacks, verbose=0)

                    # Evaluate the model on the test set
                    _, mae = model.evaluate(X_test, y_test)

                    # Keep track of the best model and its performance
                    if mae < best_mae:
                        best_mae = mae
                        best_model = model
                        best_hidden_layers = hidden_layers
                        best_units_per_layer = units_per_layer
                        best_dropout_rate = dropout_rate
                        best_l1_regularizer = l1_regularizer
                        best_l2_regularizer = l2_regularizer

print("Best Hidden Layers:", best_hidden_layers)
print("Best Units Per Layer:", best_units_per_layer)
print("Best Dropout Rate:", best_dropout_rate)
print("Best L1 Regularizer:", best_l1_regularizer)
print("Best L2 Regularizer:", best_l2_regularizer)
print("Best MAE:", best_mae)

# Use the best model for predictions on the test set
predictions = best_model.predict(X_test)

# Evaluate the model on the test set
loss, mae = best_model.evaluate(X_test, y_test)
print("Mean Absolute Error on Test Set:", mae)

# Display the predictions and corresponding actual values
for i in range(len(predictions)):
    print(f"Prediction {i+1}: {predictions[i][0]:.2f} \t Actual Value: {y_test[i]:.2f}")

def save_model_to_pickle(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Replace 'best_model.pkl' with the desired filename for your saved model
save_model_to_pickle(best_model, 'best_model.pkl')

# Save the scaler to a pickle file
def save_scaler_to_pickle(scaler, filename):
    with open(filename, 'wb') as file:
        pickle.dump(scaler, file)

# Replace 'scaler.pkl' with the desired filename for saving the scaler
save_scaler_to_pickle(scaler, 'scaler.pkl')
