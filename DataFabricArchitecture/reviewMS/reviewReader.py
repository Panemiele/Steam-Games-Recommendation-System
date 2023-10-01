import json
import string

import nltk
import pandas as pd
from kafka import KafkaProducer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pandarallel import pandarallel

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


def convert_text(text):
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    x = []
    y = []
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            x.append(i)
    for i in x:
        y.append(ps.stem(i))
    return ' '.join(y)


def process_csv(path_to_file):
    revs_og = pd.read_csv(path_to_file, nrows=50000)
    revs = revs_og[['app_id', 'app_name', 'review_score', 'review_text']]
    revs.dropna(inplace=True)
    new_df = revs.sample(n=25000)
    new_df.reset_index(drop=True, inplace=True)
    new_df.drop_duplicates(inplace=True)
    new_df.rename(columns={'review_score': 'sentiment_target', 'review_text': 'text'}, inplace=True)
    new_df.reset_index(drop=True, inplace=True)
    new_df["id"] = new_df.index
    pandarallel.initialize(nb_workers=4)
    new_df['converted_text'] = new_df['text'].parallel_apply(convert_text)
    return new_df


def send_messages_to_kafka(dataframe):
    for _, row in dataframe.iterrows():
        record = row.to_dict()
        producer.send(topic_name, value=record)


#####################
# VARIABLES SECTION #
#####################
# --------------------------------------------------------------------------------------------------#
kafka_host_port = "localhost:9092"
topic_name = "reviews-ingestion-topic"
path_to_csv = '../../dataset/reviews.csv'

################
# MAIN SECTION #
################
# --------------------------------------------------------------------------------------------------#
producer = connect_kafka_producer(kafka_host_port)
reviews_df = process_csv(path_to_csv)
send_messages_to_kafka(reviews_df)
