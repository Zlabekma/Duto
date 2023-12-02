# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd

# Get the number of features in the training data
num_features = X_train.shape[1]

# Initialize the models
rf_model = RandomForestClassifier(n_estimators=100)
model = Sequential()
model.add(Dense(10, input_dim=num_features, activation='relu'))  # Adjust input_dim to match the number of features
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the training data
train_data = []
for i in range(39, 43):
    with open(f'team_score_{i}.txt', 'r') as file:
        for line in file:
            team1, score, team2 = line.split('\t')
            score1, score2 = map(int, score.split(' : '))
            outcome = 1 if score1 > score2 else 0
            train_data.append([team1, team2, score1, score2, outcome])

# Create a DataFrame for the training data
df_train = pd.DataFrame(train_data, columns=['Team1', 'Team2', 'Score1', 'Score2', 'Outcome'])

# Preprocess the training data
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
one_hot_encoder.fit(df_train[['Team1', 'Team2']])
df_train_encoded = one_hot_encoder.transform(df_train[['Team1', 'Team2']]).toarray()

# Prepare the training data for the models
X_train = np.concatenate([df_train_encoded, df_train[['Score1', 'Score2']].values], axis=1)
y_train = df_train['Outcome']

# Train the models
rf_model.fit(X_train, y_train)
model.fit(X_train, y_train, epochs=10, batch_size=32)


# %% 
# Load the test data
test_data = []
with open('team_score_43.txt', 'r') as file:
    for line in file:
        team1, score, team2 = line.split('\t')
        score1, score2 = map(int, score.split(' : '))
        outcome = 1 if score1 > score2 else 0
        test_data.append([team1, team2, score1, score2, outcome])

# Create a DataFrame for the test data
df_test = pd.DataFrame(test_data, columns=['Team1', 'Team2', 'Score1', 'Score2', 'Outcome'])

# Preprocess the test data
df_test_encoded = one_hot_encoder.transform(df_test[['Team1', 'Team2']]).toarray()

# Prepare the test data for prediction
X_test = np.concatenate([df_test_encoded, df_test[['Score1', 'Score2']].values], axis=1)

# Predict the outcomes of the games in the test data
rf_predictions = rf_model.predict(X_test)
nn_predictions = (model.predict(X_test) > 0.5).astype("int32")

# Compare the predictions with the actual outcomes
rf_correct_predictions = np.sum(rf_predictions == df_test['Outcome'])
rf_incorrect_predictions = len(rf_predictions) - rf_correct_predictions

nn_correct_predictions = np.sum(nn_predictions.flatten() == df_test['Outcome'])
nn_incorrect_predictions = len(nn_predictions) - nn_correct_predictions

print('Random Forest: Correct predictions:', rf_correct_predictions, 'Incorrect predictions:', rf_incorrect_predictions)
print('Neural Network: Correct predictions:', nn_correct_predictions, 'Incorrect predictions:', nn_incorrect_predictions)


# %%
# Load the test data
test_data = []
for i in range(43, 45):  # This will loop over both 43 and 44
    with open(f'team_score_{i}.txt', 'r') as file:
        for line in file:
            team1, score, team2 = line.split('\t')
            score1, score2 = map(int, score.split(' : '))
            outcome = 1 if score1 > score2 else 0
            test_data.append([team1, team2, score1, score2, outcome])

# Create a DataFrame for the test data
df_test = pd.DataFrame(test_data, columns=['Team1', 'Team2', 'Score1', 'Score2', 'Outcome'])

# Preprocess the test data
df_test_encoded = one_hot_encoder.transform(df_test[['Team1', 'Team2']]).toarray()

# Prepare the test data for prediction
X_test = np.concatenate([df_test_encoded, df_test[['Score1', 'Score2']].values], axis=1)

# Predict the outcomes of the games in the test data
rf_predictions = rf_model.predict(X_test)
nn_predictions = (model.predict(X_test) > 0.5).astype("int32")

# Compare the predictions with the actual outcomes
rf_correct_predictions = np.sum(rf_predictions == df_test['Outcome'])
rf_incorrect_predictions = len(rf_predictions) - rf_correct_predictions

nn_correct_predictions = np.sum(nn_predictions.flatten() == df_test['Outcome'])
nn_incorrect_predictions = len(nn_predictions) - nn_correct_predictions

print('Random Forest: Correct predictions:', rf_correct_predictions, 'Incorrect predictions:', rf_incorrect_predictions)
print('Neural Network: Correct predictions:', nn_correct_predictions, 'Incorrect predictions:', nn_incorrect_predictions)


'''
This seems incredibally high, but it is because the data is not split into training and testing data.
I will create some random data to see if the model is working correctly.
'''


# %%
# Load the test data
test_data = []
with open('random_scores.txt', 'r') as file:
    for line in file:
        team1, score, team2 = line.split('\t')
        score1, score2 = map(int, score.split(' : '))
        outcome = 1 if score1 > score2 else 0
        test_data.append([team1, team2, score1, score2, outcome])

# Create a DataFrame for the test data
df_test = pd.DataFrame(test_data, columns=['Team1', 'Team2', 'Score1', 'Score2', 'Outcome'])

# Preprocess the test data
df_test_encoded = one_hot_encoder.transform(df_test[['Team1', 'Team2']]).toarray()

# Prepare the test data for prediction
X_test = np.concatenate([df_test_encoded, df_test[['Score1', 'Score2']].values], axis=1)

# Predict the outcomes of the games in the test data
rf_predictions = rf_model.predict(X_test)
nn_predictions = (model.predict(X_test) > 0.5).astype("int32")

# Compare the predictions with the actual outcomes
rf_correct_predictions = np.sum(rf_predictions == df_test['Outcome'])
rf_incorrect_predictions = len(rf_predictions) - rf_correct_predictions

nn_correct_predictions = np.sum(nn_predictions.flatten() == df_test['Outcome'])
nn_incorrect_predictions = len(nn_predictions) - nn_correct_predictions

print('Random Forest: Correct predictions:', rf_correct_predictions, 'Incorrect predictions:', rf_incorrect_predictions)
print('Neural Network: Correct predictions:', nn_correct_predictions, 'Incorrect predictions:', nn_incorrect_predictions)