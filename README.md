# Recommendation-_system

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: MADDINENI ROHITHA

*INTERN ID*: CT06DL736

*DOMAIN*: MACHINE LEARNING

*DURATION*: 6 WEEEKS

*MENTOR*: NEELA SANTOSH

## Project Objective:

This project implements a Recommendation System using User-Based Collaborative Filtering. The goal is to provide personalized item suggestions for users based on their similarity to other users in the dataset. This type of recommendation system is widely used in e-commerce, streaming platforms, and social media applications to enhance user experience through intelligent personalization.

## Dataset:

The dataset used in this project is large_ratings.csv, which contains user-item interaction data in the form of explicit ratings. Each row represents a rating given by a user to a specific item.
- user_id: Unique identifier for each us
- item_id: Unique identifier for each item (movie, product, etc.)
- rating: Numerical rating provided by the user

## Key Features:

- Implements User-User Collaborative Filtering using cosine similarity
- Generates personalized item recommendations
- Makes use of efficient vectorized operations with NumPy
- Easy to extend to top-N recommendations or evaluation metrics
- Fully implemented in Python using pandas, numpy, and scikit-learn

## Code Breakdown:

### 1. Import Libraries:

- pandas and numpy are used for data manipulation.
- cosine_similarity is used to compute user-to-user similarity.
- mean_squared_error is imported for potential evaluation but not used here.

```python
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

```

### 2. Load the Dataset:

- Reads a CSV file named large_ratings.csv, expected to contain user_id, item_id, and rating.
- Displays the first few rows to understand the structure.

```python
df = pd.read_csv('large_ratings.csv')  
print("Original Ratings Data:")
print(df.head())
```

### 3. Create the User-Item Matrix:

- Converts raw data into a user-item matrix, where rows are users, columns are items, and values are ratings.
- Missing values (i.e., items not rated by a user) are filled with 0.

```python
user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
print("\n User-Item Matrix:")
print(user_item_matrix.head())
```

### 4. Compute User Similarity Matrix:

- Computes pairwise cosine similarity between all users.
- Returns a square matrix where each cell [i, j] indicates how similar User i is to User j.

```python
user_similarity = pd.DataFrame(
    cosine_similarity(user_item_matrix),
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)
print("\n User Similarity Matrix:")
print(user_similarity.round(6).head())
```

### 5. Recommendation Function:

- Recommends the best item for a user based on ratings from similar users.
Key Steps:
- sim_scores: Similarity between the target user and others.
- user_ratings: Ratings given by the target user.
- unrated_items: Items the user hasn't rated.
- For each unrated item:
   - Collect ratings from other users.
   - Filter to only users who rated the item.
   - Compute a similarity-weighted average rating.
- Recommend the item with the highest predicted score.

```python
def recommend_best_item(user_id, user_item_matrix, user_similarity):
    sim_scores = user_similarity[user_id]
    user_ratings = user_item_matrix.loc[user_id]
    unrated_items = user_ratings[user_ratings == 0].index
    scores = {}
    for item in unrated_items:
        item_ratings = user_item_matrix[item]
        mask = item_ratings > 0
        relevant_sims = sim_scores[mask]
        relevant_ratings = item_ratings[mask]
        if relevant_sims.sum() > 0:
            scores[item] = np.dot(relevant_sims, relevant_ratings) / relevant_sims.sum()
    if scores:
        top_item = max(scores, key=scores.get)
        print(f"\n Top Recommendation for User {user_id}:")
        print(f" item_id\n{top_item}    {scores[top_item]:.5f}")
    else:
        print(f"\n No recommendations available for User {user_id}.")
```

### 6. Execute the Recommendation:

- Recommends the best item for User 2, based on ratings from similar users.

```python
recommend_best_item(user_id=2, user_item_matrix=user_item_matrix, user_similarity=user_similarity)
```

## Conclusion:

This project demonstrates a simple yet powerful User-Based Collaborative Filtering system. It helps identify personalized recommendations using only user behavior without needing any item metadata or content. The approach is widely applicable in recommendation domains such as movie suggestions, product recommendations, and online content personalization.

The code is clean, interpretable, and forms a strong baseline for more complex models or production-level recommendation systems. It’s ideal for learning and demonstrating how collaborative filtering works.
