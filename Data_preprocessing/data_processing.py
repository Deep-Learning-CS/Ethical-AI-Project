import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from data_read import techclass
import pandas as pd
import spacy

df = techclass

import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load Ukrainian spacy model
nlp_uk = spacy.load("uk_core_news_sm")
stopwords_uk = nlp_uk.Defaults.stop_words

# Get Russian stopwords
stopwords_ru = set(stopwords.words('russian'))

# For Russian we can use the Snowball stemmer
stemmer_ru = SnowballStemmer('russian')

def normalize_text(text, lang):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, punctuation, and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and normalize
    if lang == 'uk':
        # For Ukrainian, use spaCy lemmatization instead of stemming
        doc = nlp_uk(text)
        # Get lemmas and filter out stopwords
        tokens = [token.lemma_ for token in doc if token.text not in stopwords_uk]
    elif lang == 'ru':
        tokens = [stemmer_ru.stem(word) for word in tokens if word not in stopwords_ru]
    
    return ' '.join(tokens)

df['normalized_content'] = df.apply(lambda row: normalize_text(row['content'], row['lang']), axis=1)
print(df[['content', 'normalized_content']].head())