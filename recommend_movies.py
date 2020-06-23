import lib.cbf_movie_recommendations as cbfmr

query_title = 'The Dark Knight'
recommendations = cbfmr.movie_recommendations(query_title)
response = {"query_title": query_title, "recommended_movies": recommendations}

print("::: test result")
print(response)
