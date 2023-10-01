# This script reads the user-given tags for a specific game directly from steam's store webpage. To do so,
# the page of the game is requested in GET from the store, and filtered manually via a regex, to find every category
# the game has.
# This operation is made asynchronously to make it faster, with some delays in-between
#
# This procedure is done since Steam api's do not offer any way to get game's user-given tags, which are more
# accurate than the ones that the store gives

import multiprocessing
import re
import time
import urllib.parse
import os.path

import pandas as pd
import requests
from tqdm import tqdm

game_info_url = "https://store.steampowered.com/app/"


def process_game_id(game_id_method):
    game_id_method = str(game_id_method)
    response = requests.get(game_info_url + game_id_method)

    matches = re.findall(r'href="(https://store\.steampowered\.com/tags/en/([^/]+)/)', response.text)

    if not matches:
        print(f"No match found for game ID {game_id_method}. Check the response:")
        print(response.text)
        return game_id_method, []

    extracted_text_res = [urllib.parse.unquote(match[1]) for match in matches]
    return game_id_method, extracted_text_res


def start():
    # Get steam api key (Maybe not needed for now)
    # Domain requested: localhost

    # Define the file path
    file_path = 'api_key.txt'

    try:
        # Open the file in read mode
        with open(file_path, 'r') as file:
            # Read the API key from the file
            api_key = file.read().strip()
            print(f"Steam API Key found")
    except FileNotFoundError:
        print(f"Api file '{file_path}' not found.")

    # Get the list of games found in the data, and generate a list of categories
    revs = pd.read_csv('kaggle/dataset.csv')
    game_ids = revs['app_id'].unique()
    print(game_ids.size)

    id_tags = {}

    counter = 0

    print("Starting to send to steam the requests")

    # Define the number of processes (adjust as needed)
    num_processes = multiprocessing.cpu_count()  # Use all available CPU cores

    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=num_processes)

    # Create a progress bar with tqdm
    with tqdm(total=len(game_ids), desc="Processing") as pbar:
        results = []
        for game_id in game_ids:
            result = pool.apply_async(process_game_id, (game_id,))
            results.append(result)
            pbar.update(0.2)

        # Wait for all processes to finish and collect the results
        for result in results:
            result_data = result.get()
            if result_data:
                game_id, extracted_texts = result_data
                id_tags[game_id] = extracted_texts
            pbar.update(0.8)

    # Close the pool of worker processes
    pool.close()
    pool.join()

    print(id_tags)
    games_tag_df = pd.DataFrame.from_dict(id_tags, orient='index').transpose()
    # Save such list
    games_tag_df.to_csv('games_tags.csv', index=False)


# Guard to make the call asynchronous
if __name__ == "__main__":
    if os.path.exists('games_tags.csv'):
        res = input("Analysis already done. Are you sure you'd like to restart? (Y)es/(N)o")
        if res.lower() == 'n' or res.lower() == 'no':
            exit(0)
        print(
            "Putting a 10 seconds delay just to make sure you know what you're doing "
            "and not overwriting the result by error.")
        time.sleep(10)

    print("Starting the tags scraping")
    start()
