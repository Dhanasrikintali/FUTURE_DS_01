import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
from collections import Counter
import re

# Load the dataset
df = pd.read_csv("Social_Media_Sentiments.csv")

# Show first 5 records
print("First 5 records:\n", df.head())

# Rename column if needed (in case it's "Text" with capital T)
if 'Text' in df.columns:
    df.rename(columns={'Text': 'text'}, inplace=True)

# Function to clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Apply cleaning
df['clean_text'] = df['text'].apply(clean_text)

# Sentiment analysis function
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "positive"
    elif polarity < 0:
        return "negative"
    else:
        return "neutral"

# Apply sentiment analysis
df['predicted_sentiment'] = df['clean_text'].apply(get_sentiment)

# Show sentiment distribution
print("\nSentiment Distribution:\n", df['predicted_sentiment'].value_counts())

# Visualization 1: Sentiment Count
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='predicted_sentiment', palette='pastel')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Visualization 2: Pie Chart
sentiment_counts = df['predicted_sentiment'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Sentiment Breakdown')
plt.axis('equal')
plt.tight_layout()
plt.show()

# Trending Topics Analysis
all_words = ' '.join(df['clean_text'])
stopwords = set([
    'the', 'to', 'and', 'a', 'is', 'in', 'it', 'for', 'on', 'with', 'of', 'at', 'this', 'that',
    'was', 'are', 'as', 'i', 'my', 'be', 'have', 'has', 'just', 'so', 'not', 'an', 'but', 'from',
    'me', 'you', 'we', 'they', 'will', 'our', 'its', 'your', 'or', 'by', 'if', 'about'
])

words = [word for word in all_words.split() if word.lower() not in stopwords and len(word) > 2]
word_counts = Counter(words)
top_words = word_counts.most_common(10)

# Print trending keywords
print("\nTop 10 Trending Topics:")
for word, count in top_words:
    print(f"{word}: {count}")

# Bar chart of trending keywords
words, counts = zip(*top_words)
plt.figure(figsize=(10, 5))
plt.bar(words, counts, color='skyblue')
plt.title("Top 10 Trending Topics (Keywords)")
plt.xlabel("Keywords")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Word Cloud
wc = WordCloud(width=800, height=400, background_color='white').generate(all_words)
plt.figure(figsize=(12, 6))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Trending Topics")
plt.tight_layout()
plt.show()

# Export results to Excel
output_file = "sentiment_output.xlsx"
df.to_excel(output_file, index=False)

# Save trending keywords to new sheet
keywords_df = pd.DataFrame(top_words, columns=["Keyword", "Frequency"])
with pd.ExcelWriter(output_file, engine='openpyxl', mode='a') as writer:
    keywords_df.to_excel(writer, sheet_name='Trending_Topics', index=False)

print(f"\n✅ Analysis complete. Results saved to: {output_file}")
