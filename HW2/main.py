from flask import Flask, render_template, jsonify, request
import pickle
import math

app = Flask(__name__)

def load_data(save_file):
    with open(save_file, 'rb') as f:
        return pickle.load(f)

DATASET1_PORTER = load_data(r'datas\Male_tokenizer_with_porter_data.pkl')
DATASET2_PORTER = load_data(r'datas\Female_tokenizer_with_porter_data.pkl')
DATASET1_ORIGINAL = load_data(r'datas\Male_tokenizer_data.pkl')
DATASET2_ORIGINAL = load_data(r'datas\Female_tokenizer_data.pkl')


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
def search_page():
    return render_template('search.html')

@app.route('/results')
def results_page():
    return render_template('results.html')

@app.route('/data1')
def get_data1():
    data_type = request.args.get('type', 'original')
    data = DATASET1_PORTER if data_type == 'porter' else DATASET1_ORIGINAL
    return jsonify(prepare_zipf_data(data[1]))

@app.route('/data2')
def get_data2():
    data_type = request.args.get('type', 'original')
    data = DATASET2_PORTER if data_type == 'porter' else DATASET2_ORIGINAL
    return jsonify(prepare_zipf_data(data[1]))

@app.route('/search')
def search():
    term = request.args.get('term', '').lower()
    keyword = request.args.get('keyword', '').lower()
    use_porter = request.args.get('use_porter', 'false') == 'true'
    
    data1 = DATASET1_PORTER if use_porter else DATASET1_ORIGINAL
    data2 = DATASET2_PORTER if use_porter else DATASET2_ORIGINAL

    result = {
        'term': term,
        'found': False,
        'dataset1': {'rank': None, 'frequency': 0, 'positions': []},
        'dataset2': {'rank': None, 'frequency': 0, 'positions': []}
    }

    if keyword in data1[0]:
        result['found'] = True
        result['dataset1']['rank'] = list(data1[0].keys()).index(keyword) + 1
        result['dataset1']['frequency'] = data1[1][keyword]
        result['dataset1']['positions'] = [
            {'file': pos[0], 'position': pos[1], 'title': pos[2]}
            for pos in data1[0][keyword][:10]  # 限制返回前10個位置
        ]

    if keyword in data2[0]:
        result['found'] = True
        result['dataset2']['rank'] = list(data2[0].keys()).index(keyword) + 1
        result['dataset2']['frequency'] = data2[1][keyword]
        result['dataset2']['positions'] = [
            {'file': pos[0], 'position': pos[1], 'title': pos[2]}
            for pos in data2[0][keyword][:10]  # 限制返回前10個位置
        ]

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)