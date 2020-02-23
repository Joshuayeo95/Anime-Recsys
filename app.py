import pickle
import numpy as np
from flask_bootstrap import Bootstrap 
from flask import Flask, request, jsonify, render_template


with open('user_recommendations.pickle', 'rb') as f:
  recommendations_dict = pickle.load(f)

with open('KNNBaseline_recommender.pickle', 'rb') as f:
  algo = pickle.load(f)


app = Flask(__name__, template_folder='templates')
Bootstrap(app)


@app.route('/', methods=['GET'])
def home():
    if request.method == 'GET':
        return render_template('home.html')


@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    # Grab and save query parameters # 
    if request.method == 'GET':
        userID = int(request.args.get("userID"))
        user_recommendations = recommendations_dict[userID]
        return render_template('recommendations.html', user_recommendations=user_recommendations)

    if request.method == 'POST':

        anime = request.json["anime_name"]

        anime_iid = algo.trainset.to_inner_iid(anime)
        anime_neighbors = algo.get_neighbors(anime_iid, k=10)
        anime_neighbors_names = [algo.trainset.to_raw_iid(inner_id) for inner_id in anime_neighbors]

        return jsonify({"anime_list":anime_neighbors_names})


# @app.route('/', methods=['GET', 'POST'])
# def index():

#     if request.method == 'GET':
#         return render_template('index.html')

#     if request.method == 'POST':
#         anime = request.form["anime"]
#         # userID = request.form['userID']

#         anime_iid = algo.trainset.to_inner_iid(anime)
#         anime_neighbors = algo.get_neighbors(anime_iid, k=10)
#         anime_neighbors_names = [algo.trainset.to_raw_iid(inner_id) for inner_id in anime_neighbors]
#         # Create elements javascript #
#         return render_template('index.html', similar_animes = anime_neighbors_names)



if __name__ == '__main__':
    app.run(debug=True)

# @app.route('/similar_anime', methods=['POST'])
# def similar_anime_recommendations():
# 	algo = KNN_algo
	
# 	if request.method == 'POST':
# 		anime = request.form['anime']

#     anime_iid = algo.trainset.to_inner_iid(anime)
#     anime_neighbors = algo.get_neighbors(anime_iid, k=10)
#     anime_neighbors_names = [algo.trainset.to_raw_iid(inner_id) for \
#                              inner_id in anime_neighbors]

#     # print(f'If you liked {anime}, you may also like these animes:')
#     # for similar_anime in anime_neighbors_names:
#     #     print(similar_anime)                        

# 	return render_template('results.hmtl', similar_animes=anime_neighbors_names)                      
    


# # Get user recommendations #
# def user_recommendations(user_id, recommendations_dict):
#     counter=0
#     recommendations = recommendations_dict[user_id]
#     print('Recommended anime for you:\n')
#     for anime_rating_tuple in recommendations:
#         counter += 1 
#         print('{}. {}'.format(counter, anime_rating_tuple[0]))


