# %%
import requests
from bs4 import BeautifulSoup
import re

base_url = 'https://strahovskaliga.cz'

# Send a GET request to the URL
response = requests.get(base_url + '/Statistiky/?LID=4&YID=44&Filter=tymy')

# Check if the request was successful
if response.status_code == 200:
    # Parse the page content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all elements that contain team names and links to their profiles
    team_elements = soup.find_all('a', href=re.compile(r'TymProfil/\?TID=\d+'))

    # Create a list to store team names and URLs
    team_info = []

    # Extract and process the data for each team
    for team_element in team_elements:
        team_name = team_element.text  # Extract the team name
        team_profile_url = base_url + '/' + team_element['href']  # Construct the team profile URL
        team_info.append((team_name, team_profile_url))

    # Save the team names and URLs to a .txt file
    with open('teams_urls.txt', 'w') as file:
        for name, url in team_info:
            file.write(f"{name}\t{url}\n")

else:
    print('Failed to retrieve the webpage')


# %%
import requests
from bs4 import BeautifulSoup

# Read team names and URLs from names_urls.txt
with open('teams_urls.txt', 'r') as file:
    lines = file.readlines()

# Process each line
for line in lines:
    team_name, url = line.strip().split('\t')
    response = requests.get(url + '&YID=44')  # Access the URL with added &YID=44
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract opponent and scores from the HTML
        opponent = soup.find('a', href=lambda x: x and 'TID=' in x).text.strip()
        score_elements = soup.find_all('h4')
        scores = [element.text.strip() for element in score_elements if ':' in element.text]
        # Save to txt file
        with open('output.txt', 'a') as output_file:
            for score in scores:
                output_file.write(f'{team_name} {score} {opponent}\n')

# %%
# Process each line
for line in lines:
    team_name, url = line.strip().split('\t')
    response = requests.get(url + '&YID=44')  # Access the URL with added &YID=44
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract opponent and scores from the HTML
        opponent = soup.find('a', href=lambda x: x and 'TID=' in x).text.strip()
        score_elements = soup.find_all('h4')
        scores = [element.text.strip() for element in score_elements if ':' in element.text]
        # Extract only the scores without the date
        clean_scores = [score.split('<br/>')[-1].strip().split()[-1] for score in scores]
        # Save to txt file
        with open('output.txt', 'a') as output_file:
            for score in clean_scores:
                output_file.write(f'{team_name} {score} {opponent}\n')
