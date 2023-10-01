import json
import math

import nltk
import pandas as pd
from kafka import KafkaProducer

nltk.download('punkt')


#####################
# FUNCTIONS SECTION #
#####################
# --------------------------------------------------------------------------------------------------#
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


def process_csv(path_to_file):
    tag_og = pd.read_csv(path_to_file).T
    games = []
    for game in tag_og.iterrows():
        index, row = game
        game_dict = {}
        game_dict["app_id"] = index
        categorie_gioco = list(filter(lambda x: (isinstance(x, str)) or (not math.isnan(x)), row[0:]))
        game_dict["categories"] = categorie_gioco
        games.append(game_dict)
    return games


def send_messages_to_kafka(dataframe_list):
    for row in dataframe_list:
        producer.send(topic_name, value=row)


#####################
# VARIABLES SECTION #
#####################
# --------------------------------------------------------------------------------------------------#
kafka_host_port = "localhost:9092"
topic_name = "tags-ingestion-topic"
path_to_csv = '../../dataset/games_tags.csv'

################
# MAIN SECTION #
################
# --------------------------------------------------------------------------------------------------#
producer = connect_kafka_producer(kafka_host_port)
tags_df = process_csv(path_to_csv)
send_messages_to_kafka(tags_df)
