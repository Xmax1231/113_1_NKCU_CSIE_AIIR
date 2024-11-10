# word2vec_processor.py

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import time
from datetime import datetime, timedelta
import os
import json
from wordcloud import WordCloud
from collections import Counter
import traceback
import pickle

def check_nltk_resources():
    """
    Check and download required NLTK resources
    """
    required_resources = [
        'punkt',           # For tokenization
        'stopwords',       # For stopwords
        'wordnet',         # For lemmatization
        'averaged_perceptron_tagger'  # For POS tagging
    ]
    
    print("Checking NLTK resources...")
    for resource in required_resources:
        try:
            if resource == 'wordnet':
                # Special handling for wordnet
                nltk.data.find('corpora/wordnet')
            else:
                nltk.data.find(f'tokenizers/{resource}')
            print(f"✓ {resource} already exists")
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource)
            print(f"✓ {resource} download complete")
    print("All required NLTK resources check completed!\n")

class PubMedWord2Vec:
    def __init__(self, email, static_folder='static', cache_folder='cache'):
        """
        Initialize the Word2Vec processor
        """
        self.email = email
        self.static_folder = static_folder
        self.cache_folder = cache_folder
        
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize metrics and results
        self.performance_metrics = {}
        self.results = {}
        self.word_freq = {}  # 初始化詞頻字典
        
        # 創建快取資料夾
        for folder in [static_folder, cache_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        
        self.cache_index_file = os.path.join(cache_folder, 'cache_index.json')
        self.load_cache_index()
        
        # Set up Entrez email
        Entrez.email = email
    
    def load_cache_index(self):
        """載入快取索引"""
        try:
            if os.path.exists(self.cache_index_file):
                with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                    self.cache_index = json.load(f)
            else:
                self.cache_index = {}
        except Exception as e:
            print(f"Error loading cache index: {e}")
            self.cache_index = {}
    
    def save_cache_index(self):
        """保存快取索引"""
        try:
            with open(self.cache_index_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            print(f"Error saving cache index: {e}")
    
    def get_cache_key(self, query, max_results):
        """生成快取鍵值"""
        return f"{query}_{max_results}"

    def save_to_cache(self, query, max_results, results):
        """保存結果到快取"""
        try:
            cache_key = self.get_cache_key(query, max_results)
            
            # 保存模型檔案
            for model_type in ['skipgram', 'cbow']:
                if model_type in results.get('models', {}):
                    model_file = results['models'][model_type]['model_file']
                    if os.path.exists(os.path.join(self.static_folder, model_file)):
                        cache_model_file = os.path.join(self.cache_folder, model_file)
                        os.replace(
                            os.path.join(self.static_folder, model_file),
                            cache_model_file
                        )
            
            # 保存結果
            cache_file = os.path.join(self.cache_folder, f"{cache_key}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
            
            # 更新索引
            self.cache_index[cache_key] = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'file': cache_file,
                'query': query,
                'max_results': max_results
            }
            self.save_cache_index()
            
            print(f"Results cached for query: {query}")
            
        except Exception as e:
            print(f"Error saving to cache: {e}")
            import traceback
            traceback.print_exc()
    
    def load_from_cache(self, query, max_results):
        """從快取載入結果"""
        try:
            cache_key = self.get_cache_key(query, max_results)
            if cache_key not in self.cache_index:
                return None
            
            cache_info = self.cache_index[cache_key]
            
            # 檢查快取是否過期（例如7天）
            cache_date = datetime.strptime(cache_info['timestamp'], '%Y-%m-%d %H:%M:%S')
            if datetime.now() - cache_date > timedelta(days=7):
                print("Cache expired")
                return None
            
            # 載入快取的結果
            if os.path.exists(cache_info['file']):
                with open(cache_info['file'], 'rb') as f:
                    results = pickle.load(f)
                
                # 重新載入模型檔案到 static 資料夾
                for model_type in ['skipgram', 'cbow']:
                    if model_type in results.get('models', {}):
                        model_file = results['models'][model_type]['model_file']
                        cache_model_file = os.path.join(self.cache_folder, model_file)
                        if os.path.exists(cache_model_file):
                            os.replace(
                                cache_model_file,
                                os.path.join(self.static_folder, model_file)
                            )
                
                print(f"Results loaded from cache for query: {query}")
                return results
            
            return None
            
        except Exception as e:
            print(f"Error loading from cache: {e}")
            import traceback
            traceback.print_exc()
            return None

    def fetch_pubmed_data(self, query, max_results=1000):
        """
        Fetch articles from PubMed
        """
        start_time = time.time()
        print(f"Starting to fetch articles related to '{query}' from PubMed...")
        
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
            record = Entrez.read(handle)
            handle.close()
            
            id_list = record["IdList"]
            self.performance_metrics['article_count'] = len(id_list)
            print(f"Found {len(id_list)} articles")
            
            handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="text")
            abstracts = handle.read()
            handle.close()
            
            self.performance_metrics['fetch_time'] = time.time() - start_time
            print("Article download complete!")
            return abstracts
            
        except Exception as e:
            print(f"Error in fetch_pubmed_data: {str(e)}")
            self.performance_metrics['fetch_time'] = time.time() - start_time
            raise

    def preprocess_text(self, text):
        """
        Preprocess the text data
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words]
        
        return tokens

    def train_word2vec(self, sentences, model_type='skipgram', window_size=5):
        """
        Train Word2Vec model with specified parameters
        """
        print(f"Training {model_type} model with window size {window_size}")
        
        try:
            sg = 1 if model_type == 'skipgram' else 0
            
            model = Word2Vec(
                sentences=sentences,
                vector_size=100,
                window=window_size,
                min_count=5,
                sg=sg,
                workers=4
            )
            
            print(f"Model training complete. Vocabulary size: {len(model.wv.key_to_index)}")
            return model
            
        except Exception as e:
            print(f"Error in train_word2vec: {str(e)}")
            raise

    def analyze_results(self, model, target_word, topn=10):
        """
        Analyze results: Find most similar words to target word
        """
        try:
            print(f"\nAnalyzing {topn} most similar words to '{target_word}':")
            similar_words = model.wv.most_similar(target_word, topn=topn)
            # 轉換為基本 Python 類型
            converted_words = [(str(word), float(score)) for word, score in similar_words]
            return pd.DataFrame(converted_words, columns=['word', 'similarity'])
        except KeyError:
            return pd.DataFrame(columns=['word', 'similarity'])

    def visualize_word_embeddings(self, model, model_type, words=None, n_components=2):
        """
        Visualize word embeddings using t-SNE
        """
        if words is None:
            words = list(model.wv.key_to_index.keys())[:100]
        
        word_vectors = np.array([model.wv[word] for word in words])
        
        tsne = TSNE(n_components=n_components, random_state=42)
        embeddings = tsne.fit_transform(word_vectors)
        
        plt.figure(figsize=(12, 8))
        plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.5)
        
        for i, word in enumerate(words):
            plt.annotate(word, xy=(embeddings[i, 0], embeddings[i, 1]))
            
        plt.title(f'Word Embeddings Visualization ({model_type.upper()})')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        
        filename = f'word_embeddings_{model_type}.png'
        plt.savefig(os.path.join(self.static_folder, filename))
        plt.close()
        return filename

    def process_embeddings(self, model, top_n=150):
        """
        Process word embeddings with frequency information
        """
        try:
            # Get most common words from the vocabulary
            vocab_words = list(model.wv.key_to_index.keys())
            vocab_words.sort(key=lambda x: self.word_freq.get(x, 0), reverse=True)
            vocab_words = vocab_words[:top_n]
            
            # Get vectors for selected words
            vectors = np.array([model.wv[word] for word in vocab_words])
            
            # Perform t-SNE
            tsne = TSNE(
                n_components=2,
                perplexity=30,
                early_exaggeration=12,
                learning_rate=200,
                n_iter=1000,
                random_state=42
            )
            vectors_2d = tsne.fit_transform(vectors)
            
            # Calculate vector norms
            vector_norms = np.linalg.norm(vectors, axis=1)
            
            # Normalize norms to [0,1]
            norm_min, norm_max = vector_norms.min(), vector_norms.max()
            normalized_norms = (vector_norms - norm_min) / (norm_max - norm_min)
            
            # Prepare data for visualization
            embedding_data = []
            for i, word in enumerate(vocab_words):
                embedding_data.append({
                    'word': str(word),
                    'x': float(vectors_2d[i][0]),
                    'y': float(vectors_2d[i][1]),
                    'vector_norm': float(normalized_norms[i]),
                    'frequency': self.word_freq.get(word, 0)
                })
            
            return embedding_data
        
        except Exception as e:
            print(f"Error in process_embeddings: {str(e)}")
            traceback.print_exc()
            return []

    def process_query(self, query, max_results=1000):
        """處理查詢，加入快取機制"""
        try:
            # 嘗試從快取載入
            cached_results = self.load_from_cache(query, max_results)
            if cached_results is not None:
                return cached_results

            print(f"No cache found for query: {query}. Processing new request...")
            
            # 如果沒有快取，執行正常的處理流程
            results = self._process_query_internal(query, max_results)
            
            # 保存結果到快取
            self.save_to_cache(query, max_results, results)
            
            return results
            
        except Exception as e:
            print(f"Error in process_query: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _process_query_internal(self, query, max_results=1000):
        """
        Process the query and return all results
        """
        # Initialize metrics
        self.performance_metrics = {
            'fetch_time': 0,
            'training_time': 0,
            'article_count': 0,
            'vocabulary_size': 0
        }
        
        # Initialize results dictionary with all required keys
        self.results = {
            'query': query,
            'models': {},
            'metrics': self.performance_metrics,
            'embeddings': {
                'skipgram': [],
                'cbow': []
            },
            'word_frequencies': {
                'skipgram': [],
                'cbow': []
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        try:
            # 1. Fetch data
            abstracts = self.fetch_pubmed_data(query, max_results)
            
            # 2. Preprocess and calculate word frequencies
            print("\nStarting text preprocessing...")
            sentences = [self.preprocess_text(abstract) for abstract in abstracts.split('\n\n')]
            print(f"Processing complete, total {len(sentences)} sentences")
            
            # Calculate word frequencies
            self.word_freq = {}
            for sentence in sentences:
                for word in sentence:
                    self.word_freq[word] = self.word_freq.get(word, 0) + 1
            
            # 3. Train models and analyze
            for model_type in ['skipgram', 'cbow']:
                print(f"\nProcessing {model_type.upper()} model...")
                
                try:
                    # Train model
                    start_time = time.time()
                    model = self.train_word2vec(sentences, model_type=model_type, window_size=5)
                    training_time = time.time() - start_time
                    
                    # Update metrics
                    self.performance_metrics['training_time'] = training_time
                    self.performance_metrics['vocabulary_size'] = len(model.wv.key_to_index)
                    
                    # Process embeddings
                    print(f"Processing embeddings for {model_type}...")
                    embedding_data = self.process_embeddings(model)
                    if embedding_data:
                        self.results['embeddings'][model_type] = embedding_data
                        print(f"Generated embeddings for {len(embedding_data)} words")
                    else:
                        print(f"Warning: No embeddings generated for {model_type}")
                    
                    # Process word frequencies
                    vocab_words = list(model.wv.key_to_index.keys())
                    word_frequencies = []
                    for word in vocab_words:
                        word_frequencies.append({
                            "x": word,
                            "value": float(np.linalg.norm(model.wv[word]))
                        })
                    
                    # Sort and limit word frequencies
                    word_frequencies.sort(key=lambda x: x["value"], reverse=True)
                    word_frequencies = word_frequencies[:100]
                    
                    self.results['word_frequencies'][model_type] = word_frequencies
                    
                    # Analyze results
                    search_term = query.split('-')[0]
                    similar_words = self.analyze_results(model, search_term)
                    
                    # Save model
                    model_name = f"{query}_{model_type}_word2vec.model"
                    model_path = os.path.join(self.static_folder, model_name)
                    model.save(model_path)
                    
                    # Store results
                    self.results['models'][model_type] = {
                        'similar_words': similar_words.to_dict('records') if not similar_words.empty else [],
                        'model_file': model_name
                    }
                    
                    print(f"Completed processing {model_type} model")
                
                except Exception as e:
                    print(f"Error processing {model_type} model: {str(e)}")
                    traceback.print_exc()
                    # Continue with the next model even if this one fails
                    continue

            # Print final status
            print("\nProcessing complete!")
            print(f"Available embeddings: {list(self.results['embeddings'].keys())}")
            print(f"Available models: {list(self.results['models'].keys())}")
            
            return self.results
            
        except Exception as e:
            print(f"Error in process_query: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def save_results(self):
        """
        Save results to JSON file
        """
        filename = os.path.join(self.static_folder, 'results.json')
        with open(filename, 'w') as f:
            json.dump(self.results, f)
        return filename
    
    def get_word_frequencies(self, sentences, top_n=100):
        """
        Get word frequencies for word cloud
        """
        # Flatten the list of sentences and count word frequencies
        all_words = [word for sentence in sentences for word in sentence]
        word_freq = Counter(all_words)
        
        # Get top N words
        top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        # Format for frontend
        word_cloud_data = [{"x": word, "value": freq} for word, freq in top_words.items()]
        return word_cloud_data