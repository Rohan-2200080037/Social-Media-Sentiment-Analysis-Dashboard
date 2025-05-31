import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import pdfkit

st.set_page_config(page_title="Social Media Sentiment Dashboard", layout="wide")

st.title("📊 Social Media Sentiment Dashboard")
st.markdown("""
Upload a merged social media dataset to explore user sentiment, platform trends, and activity patterns.
The dataset should include columns such as `SentimentScore`, `Platform`, `FollowerCount`, `Date`, `Time`, `PostContent`, etc.
""")

uploaded_file = st.file_uploader("📁 Upload Merged Dataset (CSV Format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.sidebar.header("🔍 Filters")
    platforms = st.sidebar.multiselect("Choose Social Media Platforms", df["Platform"].unique(), default=df["Platform"].unique())
    years = st.sidebar.multiselect("Choose Years", df["Year"].unique(), default=df["Year"].unique())

    filtered_df = df[(df["Platform"].isin(platforms)) & (df["Year"].isin(years))]

    st.subheader("📄 Quick Look at Your Data")
    st.dataframe(filtered_df.head(), use_container_width=True)

    # Overview Metrics
    st.subheader("📊 Overview Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Number of Posts", filtered_df.shape[0])
    col2.metric("Number of Unique Users", filtered_df['UserID'].nunique())
    col3.metric("Average Sentiment Score", f"{filtered_df['SentimentScore'].astype(float).mean():.2f}")

    # Sentiment per Platform
    st.subheader("📌 Sentiment per Platform")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.countplot(data=filtered_df, x="Platform", hue="SentimentCategory", ax=ax1)
    ax1.set_title("Post Count by Platform and Sentiment")
    st.pyplot(fig1)

    # Sentiment vs Follower Numbers
    st.subheader("📈 Sentiment vs Follower Numbers")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=filtered_df, x="FollowerCount", y="SentimentScore", hue="SentimentCategory", ax=ax2)
    ax2.set_title("Relationship Between Sentiment and Follower Count")
    st.pyplot(fig2)

    # Sentiment Trends Over Time
    if 'Date' in filtered_df.columns:
        st.subheader("📆 Sentiment Trends Over Time")
        filtered_df['Date'] = pd.to_datetime(filtered_df['Date'], errors='coerce')
        trend_df = filtered_df.groupby(filtered_df['Date'].dt.to_period("M")).mean(numeric_only=True).reset_index()
        trend_df["Date"] = trend_df["Date"].dt.to_timestamp()
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=trend_df, x="Date", y="SentimentScore", ax=ax3)
        ax3.set_title("Average Monthly Sentiment")
        st.pyplot(fig3)

    # Sentiment Distribution
    st.subheader("📍 Distribution of Sentiments")
    sentiment_count = filtered_df['SentimentCategory'].value_counts().reset_index()
    sentiment_count.columns = ['Sentiment', 'Count']
    st.bar_chart(sentiment_count.set_index('Sentiment'))

    # Top Locations of Users
    st.subheader("🌍 Top Locations of Users")
    if 'Location' in filtered_df.columns:
        location_count = filtered_df['Location'].value_counts().head(10)
        st.bar_chart(location_count)

    # Posting Activity by Month
    st.subheader("📅 Posting Activity by Month")
    if 'Month' in filtered_df.columns:
        month_group = filtered_df.groupby("Month").size()
        st.line_chart(month_group)

    # Most Active Users
    st.subheader("📢 Most Active Users")
    if 'Username' in filtered_df.columns:
        active_users = filtered_df['Username'].value_counts().head(10).reset_index()
        active_users.columns = ['Username', 'PostCount']
        st.dataframe(active_users)

    # Common Words in Posts
    st.subheader("🔠 Common Words in Posts")
    if 'PostContent' in filtered_df.columns:
        all_words = ' '.join(filtered_df['PostContent'].dropna()).lower()
        words = re.findall(r'\b\w+\b', all_words)
        common_words = Counter(words).most_common(10)
        word_df = pd.DataFrame(common_words, columns=['Word', 'Count'])
        st.bar_chart(word_df.set_index('Word'))

    # Best Time of Day to Post
    st.subheader("🕐 Best Time of Day to Post")
    if 'Time' in filtered_df.columns:
        filtered_df['Hour'] = pd.to_datetime(filtered_df['Time'], errors='coerce').dt.hour
        hour_group = filtered_df.groupby('Hour').size()
        st.line_chart(hour_group.rename("Post Frequency by Hour"))

    # Best Day of the Week to Post
    st.subheader("📆 Best Day of the Week to Post")
    if 'Date' in filtered_df.columns:
        filtered_df['DayOfWeek'] = pd.to_datetime(filtered_df['Date'], errors='coerce').dt.day_name()
        day_group = filtered_df['DayOfWeek'].value_counts().reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        )
        st.bar_chart(day_group.rename("Post Frequency by Day"))

    # Average Sentiment by Day
    st.subheader("📌 Average Sentiment by Day")
    avg_sentiment_day = filtered_df.groupby('DayOfWeek')['SentimentScore'].mean().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    )
    st.line_chart(avg_sentiment_day.rename("Average Sentiment by Day"))

    # Sentiment Trends per Platform
    st.subheader("📈 Sentiment Trends per Platform")
    if 'Date' in filtered_df.columns:
        trend_platform_df = filtered_df.copy()
        trend_platform_df['Date'] = pd.to_datetime(trend_platform_df['Date'], errors='coerce')
        trend_platform_df = trend_platform_df.groupby([trend_platform_df['Date'].dt.to_period("M"), 'Platform'])['SentimentScore'].mean().reset_index()
        trend_platform_df['Date'] = trend_platform_df['Date'].dt.to_timestamp()
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=trend_platform_df, x="Date", y="SentimentScore", hue="Platform", ax=ax4)
        ax4.set_title("Sentiment Over Time by Platform")
        st.pyplot(fig4)

    # New Feature 1: Post Recommendations
    def recommend_posts(post_id, df):
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['PostContent'].dropna())
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        idx = df[df['PostContent'] == df.iloc[post_id]['PostContent']].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]  # Get top 5 similar posts
        post_indices = [i[0] for i in sim_scores]
        return df.iloc[post_indices]

    st.subheader("🔍 Post Recommendations")
    if 'PostContent' in filtered_df.columns:
        post_id = st.number_input("Enter Post ID for Recommendations", min_value=0, max_value=len(filtered_df)-1)
        recommended_posts = recommend_posts(post_id, filtered_df)
        st.write("Recommended Posts:")
        st.dataframe(recommended_posts[['Username', 'Platform', 'SentimentScore', 'PostContent']])

    # New Feature 2: Time-Based Sentiment Prediction
    def predict_sentiment_over_time(df):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['DayOfYear'] = df['Date'].dt.dayofyear
        X = df[['DayOfYear']].dropna()
        y = df.loc[X.index, 'SentimentScore']
        model = LinearRegression()
        model.fit(X, y)
        df['PredictedSentiment'] = model.predict(X)
        return df

    st.subheader("📅 Time-Based Sentiment Prediction")
    if 'Date' in filtered_df.columns:
        predicted_df = predict_sentiment_over_time(filtered_df)
        fig9, ax9 = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=predicted_df, x="Date", y="PredictedSentiment", ax=ax9)
        ax9.set_title("Predicted Sentiment Over Time")
        st.pyplot(fig9)

    # New Feature 3: Clustering Similar Posts
    def cluster_posts(df, n_clusters=5):
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['PostContent'].dropna())
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)
        df['Cluster'] = clusters
        return df

    st.subheader("🔍 Clustering Similar Posts")
    if 'PostContent' in filtered_df.columns:
        clustered_df = cluster_posts(filtered_df)
        st.write("Clustering Results (Top 5 Posts from Each Cluster):")
        for cluster in range(5):
            cluster_posts = clustered_df[clustered_df['Cluster'] == cluster].head(5)
            st.write(f"Cluster {cluster}:")
            st.dataframe(cluster_posts[['Username', 'Platform', 'SentimentScore', 'PostContent']])

    # New Feature 4: Dashboard Export as PDF
    def save_dashboard_as_pdf():
        html = st.markdown("Generated HTML content")
        pdfkit.from_string(html, 'dashboard.pdf')
        return 'dashboard.pdf'

    st.subheader("📥 Export Dashboard as PDF")
    if st.button("Download PDF"):
        file_path = save_dashboard_as_pdf()
        st.download_button("Download PDF", file_path)

else:
    st.warning("⚠️ Please upload the merged dataset to view the dashboard.")
