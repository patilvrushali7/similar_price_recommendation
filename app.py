from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)

# Load dataset (ensure it's accessible or use a relative path)
# If you have hosted the CSV externally, use pd.read_csv(<external_url>)
try:
    df = pd.read_csv('ecommerce_recommendation.csv')  # Use the file path if it's in your repo
except Exception as e:
    print(f"Error loading CSV: {e}")
    df = pd.DataFrame()  # Fallback to empty DataFrame if loading fails

# Load the serialized recommendation function
try:
    with open('recommendation_function.pkl', 'rb') as file:
        recommend_similar_products = pickle.load(file)
except Exception as e:
    print(f"Error loading model: {e}")
    recommend_similar_products = None  # Fallback in case model loading fails

@app.route('/recommend', methods=['GET'])
def recommend():
    # Get input data from the URL query parameters
    product_name = request.args.get('name')
    product_price = request.args.get('price', type=float)  # Convert price to float

    # Validate inputs
    if not product_name or product_price is None:
        return jsonify({"error": "Both 'name' and 'price' query parameters are required."}), 400

    # Call the recommendation function
    if recommend_similar_products:
        recommendations = recommend_similar_products(product_name, product_price, df)
        
        # Convert DataFrame results to JSON
        if isinstance(recommendations, pd.DataFrame):
            recommendations = recommendations.to_dict(orient='records')

        return jsonify({'recommendations': recommendations})
    else:
        return jsonify({"error": "Model not loaded properly."}), 500


if __name__ == '__main__':
    # Use port provided by the environment (e.g., for Render)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)



