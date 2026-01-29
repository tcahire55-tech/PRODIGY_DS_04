# sentiment_analysis.py

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
import nltk

nltk.download('punkt')

# -----------------------------
# 1. Load Dataset
# -----------------------------
# Update path if needed
df = pd.read_csv("twitter_validation.csv", header=None)
df.columns = ["id", "topic", "sentiment", "text"]

# If your dataset has no headers:
# df = pd.read_csv("twitter_validation.csv", header=None)
# df.columns = ["id", "topic", "sentiment", "text"]

print(df.head())
print(df.info())

# -----------------------------
# 2. Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)   # remove links
    text = re.sub(r"@\w+", "", text)      # remove mentions
    text = re.sub(r"#\w+", "", text)      # remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text) # remove special chars
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["clean_text"] = df["text"].astype(str).apply(clean_text)

# -----------------------------
# 3. Sentiment Analysis
# -----------------------------
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df["polarity"] = df["clean_text"].apply(get_sentiment)

# Classify sentiment
def sentiment_label(score):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

df["sentiment_pred"] = df["polarity"].apply(sentiment_label)

print(df[["text", "sentiment_pred"]].head())

# -----------------------------
# 4. Sentiment Distribution
# -----------------------------
sentiment_counts = df["sentiment_pred"].value_counts()
print(sentiment_counts)

# Bar Chart
plt.figure(figsize=(6,4))
sns.countplot(x="sentiment_pred", data=df)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Pie Chart
plt.figure(figsize=(6,6))
sentiment_counts.plot.pie(autopct="%1.1f%%")
plt.title("Sentiment Share")
plt.ylabel("")
plt.show()

# -----------------------------
# 5. Sentiment by Topic/Brand
# -----------------------------
topic_sentiment = df.groupby(["topic", "sentiment_pred"]).size().unstack()
print(topic_sentiment)

topic_sentiment.plot(kind="bar", stacked=True, figsize=(10,6))
plt.title("Sentiment by Topic/Brand")
plt.xlabel("Topic")
plt.ylabel("Count")
plt.legend(title="Sentiment")
plt.show()

# -----------------------------
# 6. WordClouds
# -----------------------------
positive_text = " ".join(df[df["sentiment_pred"]=="Positive"]["clean_text"])
negative_text = " ".join(df[df["sentiment_pred"]=="Negative"]["clean_text"])

# Positive WordCloud
wc_pos = WordCloud(width=800, height=400, background_color="white").generate(positive_text)
plt.figure(figsize=(10,5))
plt.imshow(wc_pos, interpolation="bilinear")
plt.axis("off")
plt.title("Positive Sentiment WordCloud")
plt.show()

# Negative WordCloud
wc_neg = WordCloud(width=800, height=400, background_color="black").generate(negative_text)
plt.figure(figsize=(10,5))
plt.imshow(wc_neg, interpolation="bilinear")
plt.axis("off")
plt.title("Negative Sentiment WordCloud")
plt.show()

# -----------------------------
# 7. Save Results
# -----------------------------
df.to_csv("sentiment_results.csv", index=False)
print("Results saved to sentiment_results.csv")
