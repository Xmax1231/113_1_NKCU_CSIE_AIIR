from flask import Flask, render_template, jsonify, request
import pickle
import math

app = Flask(__name__)

def load_data(save_file):
    with open(save_file, 'rb') as f:
        return pickle.load(f)

# DATASET1 = load_data(r'datas\Male_tokenizer_with_porter_data.pkl')
# DATASET2 = load_data(r'datas\Female_tokenizer_with_porter_data.pkl')
DATASET1 = load_data(r'datas\Male_tokenizer_data.pkl')
DATASET2 = load_data(r'datas\Female_tokenizer_data.pkl')


def prepare_zipf_data(word_count):
    # 對單詞按頻率排序
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    
    # 提取排名和頻率
    ranks = list(range(1, len(sorted_words) + 1))
    frequencies = [count for _, count in sorted_words]
    
    # 計算理想的 Zipf 分佈
    c = frequencies[0]  # 最常見單詞的頻率
    ideal_zipf = [c / r for r in ranks]
    
    return {
        'ranks': ranks,
        'frequencies': frequencies,
        'ideal_zipf': ideal_zipf,
        'words': [word for word, _ in sorted_words]
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data1')
def data1():
    # 載入數據
    # _, word_count = load_data(r'datas\Male_tokenizer_data.pkl')
    print(data1)
    _, word_count = DATASET1
    
    # 準備 Zipf 圖數據
    zipf_data = prepare_zipf_data(word_count)
    
    return jsonify(zipf_data)

@app.route('/data2')
def data2():
    # 載入數據
    # _, word_count = load_data(r'datas\Female_tokenizer_data.pkl')
    _, word_count = DATASET2
    
    # 準備 Zipf 圖數據
    zipf_data = prepare_zipf_data(word_count)
    
    return jsonify(zipf_data)

@app.route('/search')
def search():
    term = request.args.get('term', '').lower()
    result = {
        'term': term,
        'found': False,
        'dataset1': {'rank': None, 'frequency': 0, 'positions': []},
        'dataset2': {'rank': None, 'frequency': 0, 'positions': []}
    }

    # 搜索 Dataset 1
    if term in DATASET1[0]:
        result['found'] = True
        result['dataset1']['rank'] = list(DATASET1[0].keys()).index(term) + 1
        result['dataset1']['frequency'] = DATASET1[1][term]
        result['dataset1']['positions'] = [
            {'file': pos[0], 'position': pos[1], 'title': pos[2]}
            for pos in DATASET1[0][term][:10]  # 限制返回前10個位置
        ]

    # 搜索 Dataset 2
    if term in DATASET2[0]:
        result['found'] = True
        result['dataset2']['rank'] = list(DATASET2[0].keys()).index(term) + 1
        result['dataset2']['frequency'] = DATASET2[1][term]
        result['dataset2']['positions'] = [
            {'file': pos[0], 'position': pos[1], 'title': pos[2]}
            for pos in DATASET2[0][term][:10]  # 限制返回前10個位置
        ]

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)