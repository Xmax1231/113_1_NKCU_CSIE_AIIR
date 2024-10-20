import os
import pickle
import re
from collections import defaultdict
import xml.etree.ElementTree as ET
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def remove_punctuation(text):
    # 使用正則表達式去除所有標點符號
    return re.sub(r'[^\w\s]', '', text)

def ensure_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("下載 NLTK punkt 資源...")
        nltk.download('punkt', quiet=True)

def process_xml_documents(directory, save_file):
    if os.path.exists(save_file):
        print(f"載入現有的數據從 {save_file}")
        with open(save_file, 'rb') as f:
            return pickle.load(f)

    ensure_nltk_resources()
    
    word_index = defaultdict(list)
    word_count = defaultdict(int)
    stemmer = PorterStemmer()
    
    for filename in os.listdir(directory):
        if filename.endswith('.xml'):
            file_path = os.path.join(directory, filename)
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()

                title_element = root.find('.//ArticleTitle')
                title = title_element.text if title_element is not None else "No title"
                
                abstract_texts = root.findall('.//AbstractText')
                
                for abstract_text in abstract_texts:
                    if abstract_text is not None and abstract_text.text:
                        content = abstract_text.text
                        # 去除標點符號
                        content_no_punctuation = remove_punctuation(content)
                        tokens = word_tokenize(content_no_punctuation)
                        
                        for position, token in enumerate(tokens):
                            # 應用 Porter Stemming
                            # stemmed_token = stemmer.stem(token.lower())
                            stemmed_token = token
                            if stemmed_token:
                                word_index[stemmed_token].append((filename, position, title))
                                word_count[stemmed_token] += 1
            
            except ET.ParseError as e:
                print(f"無法解析文件 {filename}: {e}")
    
    print(f"保存數據到 {save_file}")
    with open(save_file, 'wb') as f:
        pickle.dump((word_index, word_count), f)
    
    return word_index, word_count

# 使用示例
target = 'Female'
directory = r'D:\Users\xmax\Documents\GitHub\113_1_NKCU_CSIE_AIIR\SearchDoc\\' + target  # 替換為您的XML文件目錄
save_file = target + '_tokenizer_data.pkl'  # 數據將被保存到這個文件
word_index, word_count = process_xml_documents(directory, save_file)

# 打印結果
print("\n詞彙索引 (前10個):")
for word, positions in list(word_index.items())[:10]:
    print(f"{word}: {positions}")

print("\n詞彙計數 (前10個):")
for word, count in sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{word}: {count}")

# 如何在之後載入數據的示例
def load_data(save_file):
    with open(save_file, 'rb') as f:
        return pickle.load(f)

# 使用示例
# loaded_word_index, loaded_word_count = load_data(save_file)