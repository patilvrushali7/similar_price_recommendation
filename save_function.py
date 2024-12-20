import pickle
from recommendation import recommend_similar_products  # Import the function from recommendation.py

# Save the function to a pickle file
with open('recommendation_function.pkl', 'wb') as file:
    pickle.dump(recommend_similar_products, file)

print("Function saved as recommendation_function.pkl")
