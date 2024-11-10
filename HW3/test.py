import numpy as np
from Bio import Entrez
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
import pandas as pd
import nltk
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import time
from datetime import datetime

class PubMedWord2Vec:
    def __init__(self, email):
        self.check_nltk_resources()
        Entrez.email = email
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.performance_metrics = {}
        
    @staticmethod
    def check_nltk_resources():
        required_resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        print("Checking NLTK resources...")
        for resource in required_resources:
            try:
                if resource == 'wordnet':
                    nltk.data.find('corpora/wordnet')
                else:
                    nltk.data.find(f'tokenizers/{resource}')
                print(f"✓ {resource} already exists")
            except LookupError:
                print(f"Downloading {resource}...")
                nltk.download(resource)
                print(f"✓ {resource} download complete")
        print("All required NLTK resources check completed!\n")

    def fetch_pubmed_data(self, query, max_results=1000):
        start_time = time.time()
        print(f"Starting to fetch articles related to '{query}' from PubMed...")
        
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        
        id_list = record["IdList"]
        print(f"Found {len(id_list)} articles")
        
        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="text")
        abstracts = handle.read()
        handle.close()
        
        fetch_time = time.time() - start_time
        self.performance_metrics['fetch_time'] = fetch_time
        self.performance_metrics['article_count'] = len(id_list)
        
        print("Article download complete!")
        return abstracts

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words]
        return tokens

    def train_word2vec(self, sentences, model_type='skipgram', window_size=5):
        start_time = time.time()
        print(f"\nStarting {model_type.upper()} model training...")
        print(f"Window size: {window_size}")
        
        sg = 1 if model_type == 'skipgram' else 0
        
        model = Word2Vec(sentences=sentences,
                        vector_size=100,
                        window=window_size,
                        min_count=5,
                        sg=sg,
                        workers=4)
        
        training_time = time.time() - start_time
        self.performance_metrics['training_time'] = training_time
        self.performance_metrics['vocabulary_size'] = len(model.wv.key_to_index)
        self.performance_metrics['model_type'] = model_type
        self.performance_metrics['window_size'] = window_size
        
        print(f"Vocabulary size: {len(model.wv.key_to_index)}")
        print("Model training complete!")
        return model

    def analyze_results(self, model, target_word, topn=10):
        """
        Analyze results: Find most similar words to target word
        """
        try:
            print(f"\nAnalyzing {topn} most similar words to '{target_word}':")
            similar_words = model.wv.most_similar(target_word, topn=topn)
            return pd.DataFrame(similar_words, columns=['word', 'similarity'])
        except KeyError:
            return f"Word '{target_word}' not found in vocabulary"

    def visualize_word_embeddings(self, model, words=None, n_components=2):
        """
        Visualize word embeddings using t-SNE
        """
        if words is None:
            words = list(model.wv.key_to_index.keys())[:100]  # Top 100 words
        
        # Get word vectors
        word_vectors = np.array([model.wv[word] for word in words])
        
        # Perform t-SNE
        tsne = TSNE(n_components=n_components, random_state=42)
        embeddings = tsne.fit_transform(word_vectors)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.5)
        
        # Add word labels
        for i, word in enumerate(words):
            plt.annotate(word, xy=(embeddings[i, 0], embeddings[i, 1]))
            
        plt.title(f'Word Embeddings Visualization (t-SNE)')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        
        # Save plot
        plt.savefig('word_embeddings.png')
        plt.close()

    def generate_report(self, query):
        """
        Generate a comprehensive system report
        """
        report = f"""
Word2Vec Implementation Report
============================
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. Data Collection
-----------------
Query: {query}
Articles retrieved: {self.performance_metrics.get('article_count', 'N/A')}
Fetch time: {self.performance_metrics.get('fetch_time', 'N/A'):.2f} seconds

2. Model Configuration
--------------------
Model type: {self.performance_metrics.get('model_type', 'N/A')}
Window size: {self.performance_metrics.get('window_size', 'N/A')}
Vector dimension: 100
Minimum word count: 5

3. Training Statistics
--------------------
Vocabulary size: {self.performance_metrics.get('vocabulary_size', 'N/A')}
Training time: {self.performance_metrics.get('training_time', 'N/A'):.2f} seconds

4. System Requirements
--------------------
Python version: {sys.version.split()[0]}
Required packages: biopython, nltk, gensim, pandas, numpy, matplotlib, seaborn, scikit-learn
"""
        # Save report
        with open('system_report.txt', 'w') as f:
            f.write(report)
        return report

def main():
    print("=== PubMed Word2Vec Demo ===")
    
    # Initialize
    email = "your.email@example.com"  # Replace with your email
    processor = PubMedWord2Vec(email)
    query = "covid-19"  # or "enterovirus"
    
    # 1. Fetch data
    abstracts = processor.fetch_pubmed_data(query, max_results=1000)
    
    # 2. Preprocess
    print("\nStarting text preprocessing...")
    sentences = [processor.preprocess_text(abstract) for abstract in abstracts.split('\n\n')]
    print(f"Processing complete, total {len(sentences)} sentences")
    
    # 3. Train model (try both CBOW and Skip-gram)
    models = {}
    for model_type in ['skipgram', 'cbow']:
        models[model_type] = processor.train_word2vec(
            sentences, 
            model_type=model_type, 
            window_size=5
        )
    
    # 4. Analyze and visualize results
    for model_type, model in models.items():
        print(f"\nResults for {model_type.upper()} model:")
        results = processor.analyze_results(model, query.split('-')[0])  # Using 'covid' for 'covid-19'
        print(results)
        
        # Visualize word embeddings
        processor.visualize_word_embeddings(model)
        
        # Save model
        model_name = f"{query}_{model_type}_word2vec.model"
        print(f"\nSaving model to {model_name}")
        model.save(model_name)
    
    # 5. Generate system report
    report = processor.generate_report(query)
    print("\nSystem Report Generated:")
    print(report)

if __name__ == "__main__":
    main()