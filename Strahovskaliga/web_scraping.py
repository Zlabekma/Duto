# %%
import requests
from bs4 import BeautifulSoup
import re

# %%
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
            file.write(f"{name} \t {url}\n")

else:
    print('Failed to retrieve the webpage')

# %%

with open('teams_urls.txt', 'r') as file:
    lines = file.readlines()

# Iterate through YID values and handle potential non-existent pages

# YID_max = 44
for yid in range(44, 38, -1):
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
# there is a problem with orderding, we also remove any team that do not play anymore 
team_names = set()
with open('teams_urls.txt', 'r') as file:
    for line in file:
        team_name = line.split('\t')[0]
        team_names.add(team_name)

for yid in range(44, 38, -1):
    filename = f'team_score_{yid}.txt'
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            data = [tuple(line.strip().split('\t')) for line in lines if line.split('\t')[0] in team_names]

            # Check if the second line is in a different order
            if len(data) > 1 and data[1][1] < data[0][1]:
                data[0], data[1] = data[1], data[0]  

            sorted_data = sorted(data, key=lambda x: (x[1], x[0], x[2]))  

        with open(filename, 'w') as file:
            for item in sorted_data:
                file.write('\t'.join(item) + '\n') 
    except FileNotFoundError:
        print(f"File {filename} does not exist.")
