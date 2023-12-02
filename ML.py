# %%
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the training data
data = []
for i in range(39, 43):
    with open(f'team_score_{i}.txt', 'r') as file:
        for line in file:
            team1, score, team2 = line.split('\t')
            score1, score2 = map(int, score.split(' : '))
            data.append([team1, team2, score1, score2])

# Create a DataFrame for the training data
df = pd.DataFrame(data, columns=['Team1', 'Team2', 'Score1', 'Score2'])

# Add 'Unknown' to the training data
df = df.append({'Team1': 'Unknown', 'Team2': 'Unknown', 'Score1': 0, 'Score2': 0}, ignore_index=True)

# Fit the label encoder with the training data
label_encoder = LabelEncoder()
label_encoder.fit(pd.concat([df['Team1'], df['Team2']]))

# Preprocess the training data
df['Team1'] = label_encoder.transform(df['Team1'])
df['Team2'] = label_encoder.transform(df['Team2'])

# Prepare the training data for modeling
X_train = df[['Team1', 'Team2', 'Score1', 'Score2']]
y_train = df.apply(lambda x: 1 if x['Score1'] > x['Score2'] else 0, axis=1)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Train a Neural Network
model = Sequential()
model.add(Dense(32, input_dim=4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

# %%
# Load the test data
test_data = []
with open('team_score_43.txt', 'r') as file:
    for line in file:
        team1, score, team2 = line.split('\t')
        score1, score2 = map(int, score.split(' : '))
        test_data.append([team1, team2, score1, score2])

# Create a DataFrame for the test data
df_test = pd.DataFrame(test_data, columns=['Team1', 'Team2', 'Score1', 'Score2'])

# Preprocess the test data
df_test['Team1'] = df_test['Team1'].apply(lambda x: 'Unknown' if x not in label_encoder.classes_ else x)
df_test['Team2'] = df_test['Team2'].apply(lambda x: 'Unknown' if x not in label_encoder.classes_ else x)
df_test['Team1'] = label_encoder.transform(df_test['Team1'])
df_test['Team2'] = label_encoder.transform(df_test['Team2'])

# Prepare the test data for prediction
X_test = df_test[['Team1', 'Team2', 'Score1', 'Score2']]

# Predict the outcomes of the games in the test data
rf_predictions = rf_model.predict(X_test)
nn_predictions = (model.predict(X_test) > 0.5).astype("int32")

print('Random Forest Predictions:', rf_predictions)
print('Neural Network Predictions:', nn_predictions)


# %%

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

# Assuming df is your DataFrame with the training data
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
one_hot_encoder.fit(df[['Team1', 'Team2']])

df_encoded = one_hot_encoder.transform(df[['Team1', 'Team2']]).toarray()
X_train = np.concatenate([df_encoded, df[['Score1', 'Score2']].values], axis=1)
y_train = df.apply(lambda x: 1 if x['Score1'] > x['Score2'] else 0, axis=1)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Train a Neural Network
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Load the test data
test_data = []
with open('team_score_43.txt', 'r') as file:
    for line in file:
        team1, score, team2 = line.split('\t')
        score1, score2 = map(int, score.split(' : '))
        test_data.append([team1, team2, score1, score2])

# Create a DataFrame for the test data
df_test = pd.DataFrame(test_data, columns=['Team1', 'Team2', 'Score1', 'Score2'])

# Preprocess the test data
df_test_encoded = one_hot_encoder.transform(df_test[['Team1', 'Team2']]).toarray()

# Prepare the test data for prediction
X_test = np.concatenate([df_test_encoded, df_test[['Score1', 'Score2']].values], axis=1)

# Predict the outcomes of the games in the test data
rf_predictions = rf_model.predict(X_test)
nn_predictions = (model.predict(X_test) > 0.5).astype("int32")

print('Random Forest Predictions:', rf_predictions)
print('Neural Network Predictions:', nn_predictions)