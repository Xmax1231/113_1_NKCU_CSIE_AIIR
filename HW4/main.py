from flask import Flask, render_template, jsonify, request
from pathlib import Path
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple
import nltk
from collections import Counter

# 下載必要的NLTK數據
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

@dataclass
class Sentence:
    original: str    # 原始句子
    cleaned: str     # 清理後的句子
    index: int       # 原始順序

class MedicalTextVectorizer:
    def __init__(self, min_df=2, max_df=0.95, tfidf_variant='standard'):
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_variant = tfidf_variant

        # 基本配置
        self.vectorizer_config = {
            'min_df': min_df,
            'max_df': max_df,
            'stop_words': 'english',
            'lowercase': True,
            'strip_accents': 'unicode'
        }

        self.configure_tfidf()

    def configure_tfidf(self):
        """配置不同的TF-IDF計算方式"""
        if self.tfidf_variant == 'standard':
            # 標準 TF-IDF
            self.vectorizer = TfidfVectorizer(
                norm='l2',
                use_idf=True,
                smooth_idf=True,
                **self.vectorizer_config
            )
        
        elif self.tfidf_variant == 'binary':
            # 二元 TF-IDF（TF使用二元權重）
            self.vectorizer = TfidfVectorizer(
                binary=True,
                use_idf=True,
                smooth_idf=True,
                **self.vectorizer_config
            )
        
        elif self.tfidf_variant == 'log_norm':
            # 對數正規化 TF-IDF
            self.vectorizer = TfidfVectorizer(
                norm='l2',
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True,  # 使用1 + log(tf)
                **self.vectorizer_config
            )
        
        elif self.tfidf_variant == 'raw_freq':
            # 原始頻率（不使用IDF）
            self.vectorizer = TfidfVectorizer(
                norm='l2',
                use_idf=False,
                **self.vectorizer_config
            )

    def split_into_sentences(self, text: str) -> List[Sentence]:
        """將文本分割為句子，並移除 PubMed 特定的標記"""
        # PubMed 常見的章節標記
        pubmed_tags = [
            "BACKGROUND:", "OBJECTIVE:", "OBJECTIVES:", "METHODS:", 
            "METHODOLOGY:", "RESULTS:", "FINDINGS:", "CONCLUSION:", 
            "CONCLUSIONS:", "DISCUSSION:", "PURPOSE:", "INTRODUCTION:",
            "AIMS:", "AIM:", "DESIGN:", "SUMMARY:", "STUDY DESIGN:",
            "MATERIALS AND METHODS:", "MAIN OUTCOME MEASURES:"
        ]
        
        # 移除所有標記
        text_processed = text
        for tag in pubmed_tags:
            text_processed = text_processed.replace(tag, "")
        
        # 清理可能留下的多餘空格和換行
        text_processed = re.sub(r'\s+', ' ', text_processed).strip()
        
        # 分割句子
        original_sentences = sent_tokenize(text_processed)
        sentences = []
        valid_sentence_count = 0
        
        for idx, orig_sent in enumerate(original_sentences):
            orig_sent = orig_sent.strip()
            if not orig_sent:
                continue
            
            cleaned_sent = self.preprocess_text(orig_sent)
            if cleaned_sent.strip():
                sentences.append(Sentence(
                    original=orig_sent,
                    cleaned=cleaned_sent,
                    index=valid_sentence_count
                ))
                valid_sentence_count += 1
        
        return sentences

    def preprocess_text(self, text: str) -> str:
        """基本的文本清理"""
        text = re.sub(r'[^\w\s.,!?]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def analyze_document(self, text: str, corpus: List[str] = None) -> Dict:
        """分析整個文檔"""
        sentences = self.split_into_sentences(text)
        
        if corpus:
            corpus_cleaned = [self.preprocess_text(doc) for doc in corpus]
            self.vectorizer.fit(corpus_cleaned)
        else:
            self.vectorizer.fit([s.cleaned for s in sentences])
        
        sentence_vectors = self.get_sentence_vectors(sentences)
        ranked_sentences = self.rank_sentences(sentences, sentence_vectors)
        
        document_vector = self.vectorizer.transform([self.preprocess_text(text)]).toarray()[0]
        document_terms = dict(zip(
            self.vectorizer.get_feature_names_out(),
            document_vector
        ))
        top_document_terms = dict(sorted(
            document_terms.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10])
        
        return {
            'ranked_sentences': ranked_sentences,
            'top_document_terms': top_document_terms,
            'vocabulary_size': len(self.vectorizer.vocabulary_),
            'feature_names': self.vectorizer.get_feature_names_out().tolist()
        }

    def get_sentence_vectors(self, sentences: List[Sentence]) -> np.ndarray:
        """將句子轉換為TF-IDF向量"""
        cleaned_sentences = [s.cleaned for s in sentences]
        return self.vectorizer.transform(cleaned_sentences).toarray()

    def rank_sentences(self, sentences: List[Sentence], sentence_vectors: np.ndarray) -> List:
        """對句子進行排名"""
        similarity_matrix = cosine_similarity(sentence_vectors)
        sentence_scores = np.mean(similarity_matrix, axis=1)
        
        # 獲取每個句子的重要詞
        top_terms = self.get_top_terms_per_sentence(sentence_vectors)
        
        ranked = [(sentences[i], float(score), terms) 
                 for i, (score, terms) in enumerate(zip(sentence_scores, top_terms))]
        
        return sorted(ranked, key=lambda x: x[1], reverse=True)

    def get_top_terms_per_sentence(self, sentence_vectors: np.ndarray, n_terms=5) -> List[Dict[str, float]]:
        """獲取每個句子中最重要的詞"""
        top_terms = []
        feature_names = self.vectorizer.get_feature_names_out()
        
        for vector in sentence_vectors:
            term_scores = {feature_names[i]: score 
                         for i, score in enumerate(vector) if score > 0}
            sorted_terms = dict(sorted(term_scores.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)[:n_terms])
            top_terms.append(sorted_terms)
            
        return top_terms

class DatasetLoader:
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.datasets = {
            "cancer": "pubmed_data/cancer_abstracts.pkl",
            "covid": "pubmed_data/covid_abstracts.pkl",
            "enterovirus": "pubmed_data/enterovirus_abstracts.pkl"
        }
    
    def load_dataset(self, dataset_name: str) -> Tuple[str, List[str], str]:
        """載入指定的數據集並返回第10篇作為目標文檔"""
        file_path = self.datasets[dataset_name]
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # 獲取排序後的PMID列表
            pmids = sorted(data.keys())
            if len(pmids) < 10:
                raise ValueError(f"Dataset {dataset_name} contains fewer than 10 documents")
            
            # 選擇第10篇文章
            target_pmid = pmids[9]
            target_doc = data[target_pmid]
            
            # 創建語料庫（排除目標文檔）
            corpus = [data[pid] for pid in pmids if pid != target_pmid]
            
            return target_doc, corpus, target_pmid
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        dataset_name = request.json.get('dataset', 'cancer')
        tfidf_variant = request.json.get('tfidf_variant', 'standard')

        # 載入數據
        loader = DatasetLoader()
        target_doc, corpus, target_pmid = loader.load_dataset(dataset_name)
        
        # 創建並訓練向量化器
        analyzer = MedicalTextVectorizer(min_df=1, max_df=0.95, 
                                       tfidf_variant=tfidf_variant)
        results = analyzer.analyze_document(target_doc, corpus)
        
        # 格式化結果
        formatted_results = {
            "target_pmid": target_pmid,
            "target_document": target_doc,
            "corpus_size": len(corpus),
            "top_document_terms": results["top_document_terms"],
            "ranked_sentences": [
                {
                    "text": sent_obj.original,
                    "score": float(score),
                    "position": sent_obj.index + 1,
                    "terms": terms
                }
                for sent_obj, score, terms in results["ranked_sentences"][:3]
            ],
            "vocabulary_size": results["vocabulary_size"]
        }
        
        return jsonify(formatted_results)
    
    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)