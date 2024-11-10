from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from word2vec_processor import PubMedWord2Vec, check_nltk_resources
import os
import json
import secrets
import numpy as np  # 添加 numpy 導入
import traceback
import datetime

app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'

# 設置一個安全的 secret key
app.secret_key = secrets.token_hex(16)

DEFAULT_EMAIL = "your.email@example.com"  # 替換為你的 email

# Check All NLTK Resources
check_nltk_resources()

class NumpyEncoder(json.JSONEncoder):
    """處理 NumPy 數據類型的 JSON 編碼器"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

# 確保 static 資料夾存在
if not os.path.exists(app.config['STATIC_FOLDER']):
    os.makedirs(app.config['STATIC_FOLDER'])

@app.route('/')
def home():
    # 清除之前的 session 數據
    session.clear()
    return render_template('index.html', results=None)

@app.route('/process', methods=['GET', 'POST'])
def process():
    if request.method == 'POST':
        try:
            query = request.form.get('query', 'covid-19')
            max_results = int(request.form.get('max_results', '1000'))
            
            processor = PubMedWord2Vec(
                DEFAULT_EMAIL, 
                static_folder=app.config['STATIC_FOLDER'],
                cache_folder='cache'  # 新增快取資料夾設定
            )
            results = processor.process_query(query, max_results=max_results)
            
            # 確保所有必要的鍵存在
            if 'embeddings' not in results:
                results['embeddings'] = {'skipgram': [], 'cbow': []}
            if 'word_frequencies' not in results:
                results['word_frequencies'] = {'skipgram': [], 'cbow': []}
            
            # 將結果轉換為 JSON 安全格式
            processed_results = json.loads(json.dumps(results, cls=NumpyEncoder))

            # 添加快取狀態到結果中
            cache_key = processor.get_cache_key(query, max_results)
            is_cached = cache_key in processor.cache_index
            if is_cached:
                cache_time = processor.cache_index[cache_key]['timestamp']
            else:
                cache_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            return render_template(
                'results.html',
                results=processed_results,
                word_frequencies=json.dumps(processed_results['word_frequencies']),
                is_cached=is_cached,
                cache_time=cache_time
            )
                                
        except Exception as e:
            print(f"Error details: {traceback.format_exc()}")
            return render_template('index.html', 
                                error=f'Error processing request: {str(e)}',
                                results=None)
    
    # 如果是 GET 請求，檢查是否有已存在的結果
    if 'analysis_results' in session:
        return render_template('results.html', 
                            results=session['analysis_results'],
                            word_frequencies=json.dumps(session['analysis_results']['word_frequencies']))
    
    return redirect(url_for('home'))

@app.route('/clear')
def clear_session():
    """清除 session 數據的路由"""
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    # 確保所有必要的目錄都存在
    for directory in ['static', 'templates']:
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    app.run(debug=True)