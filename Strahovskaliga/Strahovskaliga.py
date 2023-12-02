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

with open('teams_urls.txt', 'r') as file:
    lines = file.readlines()

team_info_set = set()

for line in lines:
    team_name, url = line.strip().split('\t')
    response = requests.get(url + '&YID=44')
    soup = BeautifulSoup(response.content, 'html.parser')
    target_div = soup.find('div', {'class': 'pole', 'id': 'zapasySmall'})

    if target_div:
        for row in target_div.select('div.pole table tbody tr'):
            team = row.select('td:nth-of-type(1) strong')[0].get_text()
            score = row.select('td:nth-of-type(2) h4')[0].get_text().split('\n')[2].strip()
            opponent = row.select('td:nth-of-type(3) strong')[0].get_text()
            team_info_set.add(tuple(sorted((team, score, opponent))))
    else:
        print(f"Target div not found in the HTML content for {team_name}")

# Write the extracted unique information to a text file
with open('unique_team_scores.txt', 'w') as file:
    for info in sorted(team_info_set):
        file.write(f"{info[1]} {info[0]} {info[2]}\n")
