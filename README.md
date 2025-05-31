# 📊 Social Media Sentiment Dashboard

This Streamlit app provides an interactive dashboard to analyze user sentiment, trends, and engagement across social media platforms. Upload a merged CSV dataset and explore a range of visual analytics and AI-powered insights.

## 🚀 Features

- 📁 Upload and filter datasets by platform and year
- 📈 Visualize sentiment trends, user activity, and platform performance
- 📌 Sentiment analysis vs follower count
- 🌍 Top locations and active users
- 🕐 Best posting times and days
- 🔠 Common word frequency
- 🔍 Post recommendations using NLP (TF-IDF + Cosine Similarity)
- 📅 Predict sentiment over time (Linear Regression)
- 🧠 Cluster similar posts (KMeans)
- 📥 Export dashboard as PDF

## 📂 Dataset Requirements

Your CSV file should include the following columns (at minimum):

- `SentimentScore`
- `Platform`
- `FollowerCount`
- `Date`
- `Time`
- `PostContent`
- `Username`, `UserID`, `Year`, `Month`, `Location` (optional but recommended)
- `SentimentCategory` (e.g., Positive, Neutral, Negative)
