# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Load the data
with open('teams_urls.txt', 'r') as file:
    team_names = [line.split('\t')[0] for line in file]

data = []
for i in range(39, 43):
    with open(f'team_score_{i}.txt', 'r') as file:
        for line in file:
            team1, score, team2 = line.split('\t')
            score1, score2 = map(int, score.split(' : '))
            data.append([team1, team2, score1, score2])

# Create a DataFrame
df = pd.DataFrame(data, columns=['Team1', 'Team2', 'Score1', 'Score2'])

# Data Preprocessing
label_encoder = LabelEncoder()
label_encoder.fit(df['Team1'].append(df['Team2']))  # Fit the encoder on all team names
df['Team1'] = label_encoder.transform(df['Team1'])
df['Team2'] = label_encoder.transform(df['Team2'])

# Model Training
X = df[['Team1', 'Team2', 'Score1', 'Score2']]
y = (df['Score1'] > df['Score2']).astype(int)  # Outcome of the match (1 if Team1 wins, 0 otherwise)
model = LinearRegression()
model.fit(X, y)

# Load the data from team_score_43.txt and team_score_44.txt
data_test = []
actual_outcomes = []
for i in range(43, 45):
    with open(f'team_score_{i}.txt', 'r') as file:
        for line in file:
            team1, score, team2 = line.split('\t')
            score1, score2 = map(int, score.split(' : '))
            data_test.append([team1, team2, score1, score2])
            actual_outcomes.append(score1 > score2)

# Create a DataFrame for testing data
df_test = pd.DataFrame(data_test, columns=['Team1', 'Team2', 'Score1', 'Score2'])

# Preprocess the testing data using the same label encoder and handle unseen labels
df_test['Team1'] = df_test['Team1'].apply(lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1)
df_test['Team2'] = df_test['Team2'].apply(lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1)

# Filter out rows with unseen labels
df_test = df_test[(df_test['Team1'] != -1) & (df_test['Team2'] != -1)]

# Use the trained model to predict the outcomes of the matches
X_test_new = df_test[['Team1', 'Team2', 'Score1', 'Score2']]
y_pred = model.predict(X_test_new) > 0.5

# Compare the predicted outcomes with the actual outcomes and print the corresponding data
print("Comparison of predicted outcomes with actual outcomes:")
correct_predictions = 0
incorrect_predictions = 0
for i in range(len(y_pred)):
    outcome = "Team1 wins" if y_pred[i] else "Team2 wins or draw"
    actual = "Team1 wins" if actual_outcomes[i] else "Team2 wins or draw"
    print(f"Match {i+1}: Predicted: {outcome}, Actual: {actual}")
    print(f"   Team1: {data_test[i][0]}, Team2: {data_test[i][1]}, Score: {data_test[i][2]} - {data_test[i][3]}")
    if y_pred[i] == actual_outcomes[i]:
        correct_predictions += 1
    else:
        incorrect_predictions += 1

print("\nStatistics:")
print(f"Correct Predictions: {correct_predictions}")
print(f"Incorrect Predictions: {incorrect_predictions}")

