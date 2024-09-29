import os
import pprint
from typing import Any, Dict
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from searchText import search_document, document_statistics, visualization_web_result, visualization_web_fulltext


@dataclass
class DocumentType:
    Filename: str
    PMID: int
    ArticleTitle: str
    AbstractText: str
    NumChars: int
    NumWords: int
    NumSentences: int


app = Flask(__name__)
UPLOAD_FOLDER = r'files'
ALLOWED_EXTENSIONS = set(['xml'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

DOCUMENTS_DATAS: Dict[str, DocumentType] = {}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def update_file_information() -> None:
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    DOCUMENTS_DATAS.clear()
    for file in files:
        tree = ET.parse(os.path.join(app.config['UPLOAD_FOLDER'], file))
        root = tree.getroot()
        _PMID = root.find("PubmedArticle").find("MedlineCitation").find("PMID").text
        _ArticleTitle = root.find("PubmedArticle").find("MedlineCitation").find("Article").find("ArticleTitle").text
        _AbstractText = root.find("PubmedArticle").find("MedlineCitation").find("Article").find("Abstract").find("AbstractText").text
        num_chars, num_words, num_sentences = document_statistics(_AbstractText)
        
        DOCUMENTS_DATAS[file] = DocumentType(
            Filename=file,
            PMID=_PMID,
            ArticleTitle=_ArticleTitle,
            AbstractText=_AbstractText,
            NumChars=num_chars,
            NumWords=num_words,
            NumSentences=num_sentences
        )
    # pprint.pprint(DOCUMENTS_DATAS)


@app.route('/')
def home():
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    if len(DOCUMENTS_DATAS) <= 0:
        update_file_information()

    return render_template('home.html')


@app.route('/manage', methods=['GET'])
def upload_file():
    # if request.method == 'POST':
    #     uploaded_files = request.files.getlist('files')  # 獲取所有上傳的檔案
    #     for file in uploaded_files:
    #         if file and allowed_file(file.filename):
    #             filename = secure_filename(file.filename)
    #             file.save(os.path.join(app.config['UPLOAD_FOLDER'], 
    #                                 filename))
    #         update_file_information()

    files = []
    for Filename, document in DOCUMENTS_DATAS.items():
        files.append({
                "Filename": document.Filename,
                "PMID": document.PMID,
                "NumChars": document.NumChars,
                "NumWords": document.NumWords,
                "NumSentences": document.NumSentences
                })
    return render_template('manage.html', files=files)


@app.route('/api/upload', methods=['POST'])
def upload_files():
    uploaded_files = request.files.getlist('files')  # 獲取所有上傳的檔案
    filenames = []
    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 
                                filename))
            filenames.append(filename)
        update_file_information()
    return jsonify({'success': True, 'message': f'已成功上傳檔案: {filenames}'})


@app.route('/api/delete', methods=['POST'])
def delete_file():
    filename = request.json.get('filename')  # 從前端接收檔案名
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    if os.path.exists(file_path):
        os.remove(file_path)  # 刪除檔案
        update_file_information()
        return jsonify({'success': True, 'message': f'{filename} 已成功刪除'})
    else:
        return jsonify({'success': False, 'message': '檔案不存在'})


@app.route('/search', methods=['GET', 'POST'])
def search_files():
    keyword = ''
    results = []
    if request.method == 'POST':
        keyword = request.form['query']
        print(f"query={keyword}")

        for Filename, document in DOCUMENTS_DATAS.items():
            AbstractText = document.AbstractText
            search_indexes = search_document(AbstractText, keyword)
            # print(f"{Filename}:\r\n{search_indexes}")

            if len(search_indexes) <= 0:
                    continue
        
            # for index in search_indexes:
            #     vResultString = visualization_web_result(AbstractText, index, keyword)
            #     print(f"Position({index[0]}):\r\n{vResultString}\r\n")
            #     results.append({"Filename": document.Filename, "Title": document.ArticleTitle, "vResultString":vResultString})
            vResultString = visualization_web_result(AbstractText, search_indexes[0])
            vFullTextResult = visualization_web_fulltext(AbstractText, search_indexes)
            results.append({
                "Filename": document.Filename,
                "PMID": document.PMID,
                "Title": document.ArticleTitle,
                "AbstractText": document.AbstractText,
                "vResultString": vResultString,
                "vFullTextResult": vFullTextResult
                })

    return render_template('search.html', query=keyword, results=results)


if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    update_file_information()
    app.run(host="127.0.0.1", port=5000, debug=True)