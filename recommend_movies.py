import lib.cbf_movie_recommendations as cbfmr

query_title = 'The Dark Knight'

print("::: Recommendation for Movie {}:".format(query_title))
print(cbfmr.movie_recommendations(query_title))
