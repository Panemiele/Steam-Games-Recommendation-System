import json

import nltk
import requests
from kafka import KafkaProducer

nltk.download('punkt')


#####################
# FUNCTIONS SECTION #
#####################
# --------------------------------------------------------------------------------------------------#
def get_api_key():
    try:
        # Open the file in read mode
        with open(file_path, 'r') as file:
            # Read the API key from the file
            api_key = file.read().strip()
            print(f"Steam API Key found")
            return api_key
    except FileNotFoundError:
        print(f"Api file '{file_path}' not found.")


def connect_kafka_producer(host_port):
    _producer = None
    try:
        _producer = KafkaProducer(bootstrap_servers=[host_port],
                                  value_serializer=lambda v: json.dumps(v).encode('utf-8'))
    except Exception as ex:
        print('Exception while connecting Kafka')
        print(str(ex))
    finally:
        return _producer


def get_users_games(users_ids):
    users_games_dict = []
    for uid in users_ids:
        url = "http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key={}&steamid={}&format=json&include_played_free_games".format(
            api_key, uid)
        response = requests.get(url)
        if response.status_code == 200:
            print("Success")
            u_g_dict = {}
            u_g_dict["user_id"] = uid
            u_g_dict["games"] = response.json()["response"]["games"]
            users_games_dict.append(u_g_dict)
        else:
            print("Something went wrong: ", response.status_code, response.text)
    return users_games_dict


def send_messages_to_kafka(dataframe):
    for user_games in dataframe:
        producer.send(topic_name, value=user_games)


#####################
# VARIABLES SECTION #
#####################
# --------------------------------------------------------------------------------------------------#
file_path = './api_key.txt'
kafka_host_port = "localhost:9092"
topic_name = "users-ingestion-topic"
steam_user_ids = [76561198055831348]

################
# MAIN SECTION #
################
# --------------------------------------------------------------------------------------------------#
api_key = get_api_key()
users_games_df = get_users_games(steam_user_ids)
producer = connect_kafka_producer(kafka_host_port)
send_messages_to_kafka(users_games_df)
