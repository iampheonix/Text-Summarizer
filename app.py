# Text-Summarizer
from flask import Flask, render_template, request
from transformers import pipeline
from bs4 import BeautifulSoup
import requests
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import re

app = Flask(__name__)

# Load spaCy model for TextRank
try:
    nlp = spacy.load('en_core_web_sm')
except:
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Initialize Hugging Face pipelines
t5_summarizer = pipeline("summarization", model="t5-small")
bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Common article text containers
        paragraphs = soup.find_all(['p', 'article', 'div'])
        text = ' '.join([p.get_text() for p in paragraphs])
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None

def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def textrank_summarizer(text, num_sentences=3):
    doc = nlp(text)
    
    # Calculate word frequencies
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS) + list(punctuation):
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1
                
    # Normalize frequencies
    max_frequency = max(word_frequencies.values()) if word_frequencies else 1
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word]/max_frequency
        
    # Score sentences
    sentence_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]
    
    # Get top N sentences
    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join([sent.text for sent in summary_sentences])
    return summary

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""
    method = ""
    original_length = 0
    summary_length = 0
    compression_ratio = 0
    
    if request.method == 'POST':
        text = request.form.get('text')
        url = request.form.get('url')
        method = request.form.get('method', 'bart')
        num_sentences = int(request.form.get('num_sentences', 3))
        
        if url:
            text = extract_text_from_url(url)
            if not text:
                return render_template('index.html', 
                                     error="Could not extract text from the URL. Please try another URL or paste the text directly.")
        
        if text:
            original_length = len(text)
            processed_text = preprocess_text(text)
            
            if method == 'bart':
                # BART summarization
                max_length = min(150, len(text.split()) // 2)
                min_length = max(30, max_length // 3)
                summary = bart_summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            elif method == 't5':
                # T5 summarization
                max_length = min(150, len(text.split()) // 2)
                min_length = max(30, max_length // 3)
                summary = t5_summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            elif method == 'textrank':
                # TextRank summarization
                summary = textrank_summarizer(text, num_sentences)
            
            summary_length = len(summary)
            if original_length > 0:
                compression_ratio = round((1 - (summary_length / original_length)) * 100, 2)
    
    return render_template('index.html', 
                         summary=summary,
                         method=method,
                         original_length=original_length,
                         summary_length=summary_length,
                         compression_ratio=compression_ratio)

if __name__ == '__main__':
    app.run(debug=True)
