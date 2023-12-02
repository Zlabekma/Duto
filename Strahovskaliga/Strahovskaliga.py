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

# Iterate through YID values and handle potential non-existent pages

# YID_max = 44
for yid in range(40, 38, -1):
    team_info_set = set()  # Create a new set for each YID
    for line in lines:
        team_name, url = line.strip().split('\t')
        try:
            response = requests.get(url + f'&YID={yid}')
            response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
            soup = BeautifulSoup(response.content, 'html.parser')
            target_div = soup.find('div', {'class': 'pole', 'id': 'zapasySmall'})

            if target_div:
                for row in target_div.select('div.pole table tbody tr'):
                    team = row.select('td:nth-of-type(1) strong')[0].get_text()
                    score = row.select('td:nth-of-type(2) h4')[0].get_text().split('\n')[2].strip()
                    opponent = row.select('td:nth-of-type(3) strong')[0].get_text()
                    team_info_set.add(tuple(sorted((team, score, opponent))))
            else:
                print(f"Target div not found in the HTML content for {team_name} with YID={yid}")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {team_name} with YID={yid}: {e}")

    # Write the extracted unique information to a text file
    filename = f'team_score_{yid}.txt'
    with open(filename, 'w') as file:
        for info in sorted(team_info_set):
            file.write(f"{info[1]} \t {info[0]} \t {info[2]}\n")

# %%
import os

# Define the file names
file_names = ['team_score_39.txt', 'team_score_40.txt']

# Function to switch the first and second element of each line
def switch_elements(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
    
    new_lines = []
    for line in lines:
        elements = line.strip().split('\t')
        if len(elements) >= 2 and elements[0].isdigit():
            new_line = elements[1] + '\t' + elements[0] + '\t' + '\t'.join(elements[2:])
            new_lines.append(new_line + '\n')
        else:
            new_lines.append(line)
    
    with open(file_name, 'w') as file:
        file.writelines(new_lines)

# Process each file
for file_name in file_names:
    if os.path.exists(file_name):
        switch_elements(file_name)
    else:
        print(f"The file {file_name} does not exist.")

