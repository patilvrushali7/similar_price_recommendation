import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recommend_similar_products(selected_product_name, selected_product_price, df, threshold=0.8):
    # Vectorizing product names using Bag of Words
    bow_vectorizer = CountVectorizer(stop_words='english')
    bow_matrix = bow_vectorizer.fit_transform(df['Name'])
    
    # Find the index of the selected product
    try:
        selected_index = df[df['Name'] == selected_product_name].index[0]
    except IndexError:
        return "Selected product not found in the dataset!"
    
    # Calculate cosine similarity between the selected product and all others
    cosine_similarities = cosine_similarity(bow_matrix[selected_index], bow_matrix).flatten()
    
    # Filter similar products with similarity greater than the threshold and exclude the selected product itself
    similar_indices = [i for i, score in enumerate(cosine_similarities) if score > threshold and i != selected_index]
    
    similar_products = df.iloc[similar_indices]
    
    # Exclude products with the same price as the selected product
    similar_products = similar_products[similar_products['Price'] != selected_product_price]
    
    if similar_products.empty:
        return "No similar products found!"
    
    # Return recommended products
    return similar_products
