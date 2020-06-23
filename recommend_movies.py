from confluent_kafka.avro import AvroConsumer
from confluent_kafka.avro.serializer import SerializerError
from confluent_kafka import avro
from confluent_kafka.avro import AvroProducer
import lib.cbf_movie_recommendations as cbfmr

def recommed_movies(query_title):
    recommendations = cbfmr.movie_recommendations(query_title)
    response = {"query_title": query_title, "recommended_movies": recommendations}
    return response

f = open('data/movie_recommendations.avro')
vs = f.read()
value_schema = avro.loads(vs)
avroProducer = AvroProducer({
                'bootstrap.servers': '0.0.0.0:9092',
                'schema.registry.url': 'http://0.0.0.0:8081'
                },
                default_value_schema=value_schema)

c = AvroConsumer({
    'bootstrap.servers': '0.0.0.0:9092',
    'group.id': 'group01',
    'schema.registry.url': 'http://0.0.0.0:8081'})

c.subscribe(['movies'])

while True:
    try:
        msg = c.poll(10)

    except SerializerError as e:
        print("Message deserialization failed for {}: {}".format(msg, e))
        break

    if msg is None:
        continue

    if msg.error():
        print("AvroConsumer error: {}".format(msg.error()))
        continue

    query_title = msg.value()['query_title']
    print(query_title)
    recommendations = recommed_movies(query_title)
    avroProducer.produce(topic='movie_recommendations', value=recommendations)
    # print(type(recommendations))
    # print(recommendations)

c.close()
