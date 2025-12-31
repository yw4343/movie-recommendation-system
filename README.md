# 🎬 Movie Recommendation System with Neural Collaborative Filtering

This project builds a **personalized movie recommendation system** using **Neural Collaborative Filtering (NCF)** implemented in **TensorFlow**. By leveraging historical user–movie ratings and movie metadata, the model predicts individual user preferences and recommends top-N movies for each user.

> Course Project — *Advanced Marketing Analytics, Columbia Business School*

---

## Project Overview

**Research Question:**  
How can we leverage structured and unstructured data to predict individual movie preferences and generate personalized recommendations?

**Objective:**  
Develop a recommendation model that learns user preferences based on similarities between users and between movie items, and produces accurate top-N movie recommendations.

---

## Dataset Description

Two primary datasets are used:

### 1. Ratings Data (`ratings.json`)
Contains historical user–movie interactions.
- `user_id`
- `item_id`
- `rating`

### 2. Movie Metadata (`metadata_updated.json`)
Contains descriptive information about movies.
- `item_id`
- `title`
- `directedBy`
- `starring`
- `avgRating`
- `imdbId`

---

## Data Cleaning & Exploration

Key preprocessing steps:
1. Checked for missing values (no nulls found)
2. Removed duplicate user–item ratings via aggregation
3. Performed descriptive statistics
4. Visualized rating distributions to understand user behavior

---

## Modeling Approaches Explored

| Approach | Description | Limitation |
|--------|-------------|------------|
| **SVD** | Matrix factorization for rating prediction | Not scalable for large datasets |
| **ALS** | Efficient factorization using implicit feedback | User ID mapping issues |
| **NCF (Final)** | Deep learning–based collaborative filtering | Requires tuning & more data |

---

## Neural Collaborative Filtering (NCF)

The final model uses **Neural Collaborative Filtering**, which combines matrix factorization with deep neural networks.

### Model Architecture
- User and item IDs are encoded and embedded into dense vectors
- Embeddings are flattened and concatenated
- Fully connected neural network layers learn complex user–item interactions
- Output is a predicted rating score

### Training Details
- Framework: **TensorFlow / Keras**
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Users: **247,383**
- Movies: **67,873**

---

## Model Evaluation

**Metric Used:** Root Mean Squared Error (RMSE)

- **RMSE = 0.818**
- Predictions are, on average, off by ±0.82 rating points (on a 1–5 scale)
- Indicates reasonable predictive performance for a large-scale recommendation task

---

## Recommendation Functionality

The system recommends **Top-N movies** for a given user.

### Recommendation Workflow
1. Validate user ID
2. Encode user and item indices
3. Identify unrated movies
4. Predict ratings using the trained NCF model
5. Rank movies by predicted score
6. Join with metadata and return top-N recommendations

**Function Inputs**
```
user_id, df_meta, N (default = 5)
```

**Edge Case Handling**
- If the user does not exist → returns `"User not found"`

---

## Limitations

- Cold start problem for new users or items
- Potential over-generalization due to sparse interactions
- Limited hyperparameter tuning

---

## Future Improvements

- Increase embedding dimensions and model depth
- Build a hybrid recommender (collaborative + content-based)
- Add MAE and Precision@K as evaluation metrics

---

## Tech Stack

- Python
- TensorFlow / Keras
- Pandas, NumPy
- Scikit-learn
- Matplotlib

---

## Repository Structure

```
├── Movie Recommendation System-NCF-Final.ipynb
├── Project Slides.pdf
├── README.md
```

