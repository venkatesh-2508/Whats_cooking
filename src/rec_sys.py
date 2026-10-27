import os
import ast
import pickle
import unidecode
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
from ingredient_parser import ingredient_parser
import config 

# Top-N recommendations ordered by score
def get_recommendations(N, scores):
    # load in recipe dataset 
    df_recipes = pd.read_csv(config.PARSED_PATH)
    # order the scores and filter to get the highest N scores
    # Extract scalar safely for NumPy 2.x and 1.x compatibility
    top = sorted(range(len(scores)), key=lambda i: float(np.asarray(scores[i]).squeeze()), reverse=True)[:N]
    
    # create dataframe to load in recommendations 
    recommendation = pd.DataFrame(columns=['recipe', 'ingredients', 'score', 'url'])
    count = 0
    for i in top:
        score_val = float(np.asarray(scores[i]).squeeze())
        recommendation.at[count, 'recipe'] = title_parser(df_recipes['recipe_name'][i])
        recommendation.at[count, 'ingredients'] = ingredient_parser_final(df_recipes['ingredients'][i])
        recommendation.at[count, 'url'] = df_recipes['recipe_urls'][i]
        recommendation.at[count, 'score'] = f"{score_val:.3f}"
        count += 1
    return recommendation

# neaten the ingredients being outputted 
def ingredient_parser_final(ingredient):
    if isinstance(ingredient, list):
        ingredients = ingredient
    else:
        ingredients = ast.literal_eval(ingredient)
    
    ingredients = ', '.join(ingredients)
    ingredients = unidecode.unidecode(ingredients)
    return ingredients

def title_parser(title):
    title = unidecode.unidecode(title)
    return title 

def RecSys(ingredients, N=5):
    """
    The recommendation system takes in a list of ingredients and returns a list of top 5 
    recipes based of cosine similarity.
    :param ingredients: a list of ingredients
    :param N: the number of recommendations returned 
    :return: top 5 recommendations for cooking recipes
    """
    # load in tfidf model and encodings 
    with open(config.TFIDF_ENCODING_PATH, 'rb') as f:
        tfidf_encodings = pickle.load(f)

    with open(config.TFIDF_MODEL_PATH, "rb") as f:
        tfidf = pickle.load(f)

    # parse the ingredients using ingredient_parser 
    try: 
        ingredients_parsed = ingredient_parser(ingredients)
    except Exception:
        ingredients_parsed = ingredient_parser([ingredients])
    
    # use our pretrained tfidf model to encode our input ingredients
    ingredients_tfidf = tfidf.transform([ingredients_parsed])

    # calculate cosine similarity and safely convert to float scalars for NumPy 2.x / 1.x compatibility
    scores = [float(np.asarray(cosine_similarity(ingredients_tfidf, x)).squeeze()) for x in tfidf_encodings]

    # Filter top N recommendations 
    recommendations = get_recommendations(N, scores)
    return recommendations

if __name__ == "__main__":
    # test ingredients
    test_ingredients = "pasta, tomato, onion"
    recs = RecSys(test_ingredients)
    print(recs)
