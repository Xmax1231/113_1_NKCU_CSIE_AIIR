import numpy as np
from typing import List, Dict, Tuple
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
import pickle
import random

@dataclass
class Sentence:
    original: str    # 原始句子
    cleaned: str     # 清理後的句子
    index: int       # 原始順序

class MedicalTextVectorizer:
    def __init__(self, min_df=2, max_df=0.95):
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            lowercase=True,
            norm='l2',
            use_idf=True,
            smooth_idf=True,
            strip_accents='unicode'
        )
    
    @staticmethod
    def load_data(pickle_file: str) -> Tuple[str, str, List[str]]:
        """
        載入 pickle 文件並返回目標文檔、語料庫和目標文檔的 PMID
        Returns:
            tuple: (target_document, corpus, target_pmid)
        """
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        
        # 隨機選擇一個 PMID 和對應的摘要作為目標文檔
        target_pmid = random.choice(list(data.keys()))
        target_doc = data[target_pmid]
        
        # 創建語料庫（排除目標文檔）
        corpus = [doc for pid, doc in data.items() if pid != target_pmid]
        
        return target_pmid, target_doc, corpus

    def preprocess_text(self, text: str) -> str:
        """基本的文本清理"""
        # 移除特殊字符，保留句子結構
        text = re.sub(r'[^\w\s.,!?]', '', text)
        # 移除可能的多餘空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def split_into_sentences(self, text: str) -> List[Sentence]:
        """
        將文本分割為句子，並移除 PubMed 特定的標記
        """
        # PubMed 常見的章節標記
        pubmed_tags = [
            "BACKGROUND:", "OBJECTIVE:", "OBJECTIVES:", "METHODS:", 
            "METHODOLOGY:", "RESULTS:", "FINDINGS:", "CONCLUSION:", 
            "CONCLUSIONS:", "DISCUSSION:", "PURPOSE:", "INTRODUCTION:",
            "AIMS:", "AIM:", "DESIGN:", "SUMMARY:", "STUDY DESIGN:",
            "MATERIALS AND METHODS:", "MAIN OUTCOME MEASURES:", "KEYWORDS:", "INTERPRETATION:"
        ]

        # 移除所有標記
        text_processed = text
        for tag in pubmed_tags:
            text_processed = text_processed.replace(tag, "")

        # 清理可能留下的多餘空格和換行
        text_processed = re.sub(r'\s+', ' ', text_processed).strip()

        original_sentences = sent_tokenize(text_processed)
        sentences = []
        valid_sentence_count = 0  # 用於追踪實際的句子索引
        for idx, orig_sent in enumerate(original_sentences):
            orig_sent = orig_sent.strip()
            # 跳過空句子
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
    
    def get_sentence_vectors(self, sentences: List[Sentence]) -> np.ndarray:
        """將句子轉換為TF-IDF向量"""
        cleaned_sentences = [s.cleaned for s in sentences]
        return self.vectorizer.transform(cleaned_sentences).toarray()
    
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
    
    def rank_sentences(self, 
                      sentences: List[Sentence], 
                      sentence_vectors: np.ndarray) -> List[Tuple[Sentence, float, Dict]]:
        """對句子進行排名，返回原始句子、得分和重要詞"""
        similarity_matrix = cosine_similarity(sentence_vectors)
        sentence_scores = np.mean(similarity_matrix, axis=1)
        top_terms = self.get_top_terms_per_sentence(sentence_vectors)
        
        ranked = [(sentences[i], float(score), terms) 
                 for i, (score, terms) in enumerate(zip(sentence_scores, top_terms))]
        
        return sorted(ranked, key=lambda x: x[1], reverse=True)
    
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

def main():
    # 設定隨機種子以確保結果可重現
    random.seed(42)
    
    # 創建分析器實例
    analyzer = MedicalTextVectorizer(min_df=1, max_df=0.95)
    
    # 從 pickle 文件載入數據
    search_term = "covid"
    pickle_file = f"pubmed_data/{search_term}_abstracts.pkl"  # 替換為你的 pickle 文件路徑
    target_pmid, target_document, sample_corpus = analyzer.load_data(pickle_file)
    
    print(f"\nTarget Document (PMID: {target_pmid}):")
    print("-" * 50)
    print(target_document)
    print("\nCorpus Size:", len(sample_corpus))
    
    # 分析文檔
    results = analyzer.analyze_document(target_document, sample_corpus)
    
    # 輸出結果
    print("\nTop Document Terms:")
    print("-" * 50)
    for term, score in results['top_document_terms'].items():
        print(f"{term}: {score:.4f}")
    
    print("\nTop Ranked Sentences:")
    print("-" * 50)
    for sent_obj, score, terms in results['ranked_sentences'][:3]:
        print(f"\nScore {score:.4f}: {sent_obj.original}")
        print("Original position:", sent_obj.index + 1)
        print("Important terms:", terms)
    
    print(f"\nVocabulary size: {results['vocabulary_size']}")

if __name__ == "__main__":
    main()