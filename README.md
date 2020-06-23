# movie_recommender
Content Based Filtering - based on Kaggle's movies 
dataset https://www.kaggle.com/rounakbanik/the-movies-dataset?select=keywords.csv

## Description
This content based filtering recommendation models for movies 
has been integrated with Apache Kafka for Stream Processing Real Time.

## Pre-requesites
Install all libraries from `requirements.txt`
with your `pipenv` or `virtualenv` virtualizers.

## Usage 
From command line run 
```
python recommend_movies.py
```
The above commadn will build the model and tes it with one movie title `The Dark Knight`

## Stream for nested JSON
use this command
```
CREATE STREAM movie_recommendations (query_title varchar,
                                     recommended_movies array<STRUCT<
            title VARCHAR, 
            vote_count INTEGER, 
            vote_average INTEGER,
            year VARCHAR,
            wr DOUBLE>> ) 
       WITH (kafka_topic='movie_recommendations', value_format='AVRO');
```

## Author 
Ing. Jimmy Figueroa A.
jimmy@thebearsoft.com 
