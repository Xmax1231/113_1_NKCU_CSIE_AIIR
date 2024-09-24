import re
import os
import argparse
import json
import xml.etree.ElementTree as ET


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def extract_text_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    result = []

    def recursive_text_extract(element):
        if element.text:
            result.append(element.text.strip())
        for child in element:
            recursive_text_extract(child)
    
    recursive_text_extract(root)

    return ' '.join(result)


def extract_text_from_json(json_file):
    result = []

    with open(json_file, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            
            result.extend(extract_text_from_json_data(data))
    
    return ' '.join(result)

def extract_text_from_json_data(data):
    result = []
    
    def recursive_text_extract(data):
        if isinstance(data, dict):
            for key, value in data.items():
                recursive_text_extract(value)
        elif isinstance(data, list):
            for item in data:
                recursive_text_extract(item)
        else:
            if isinstance(data, str):
                result.append(data.strip())
    
    recursive_text_extract(data)
    
    return result


def document_statistics(doc: str) -> tuple[int, int, int]:
    num_chars = len(doc)  # len(re.sub(r'\s+', "", doc))
    num_words = len(doc.split())
    sentences = re.split(r'[.!?]', doc)
    num_sentences = len([s for s in sentences if s.strip()])
    return num_chars, num_words, num_sentences


def search_document(doc: str, keyword: str) -> list[tuple[int, int]]:
    # pattern = r"\W(" + keyword + r")\W"
    pattern = r".+?(" + keyword + r").+?"
    return [(m.start(1), m.end(1)) for m in re.finditer(pattern, doc)]


def visualization_results(doc: str, index: tuple[int, int], keyword: str) -> str:

    doc = doc[:index[1]] + bcolors.ENDC + doc[index[1]:]
    doc = doc[:index[0]] + bcolors.FAIL + bcolors.BOLD + doc[index[0]:]

    start_index = index[0] - 20
    end_index = len(doc) - index[1] - 40

    if start_index <= 0:
        return doc[:-end_index]
    if end_index <= 0:
        return doc[start_index:]
    return doc[start_index:-end_index]


def main():
    parser = argparse.ArgumentParser(description='Handle data source paths.')
    parser.add_argument('--path', type=str, help='path to data source')
    args = parser.parse_args()

    if args.path is None:
        data_path = input("Input dataset path? ")  # r"D:/xmax/NKCU CSIE/113-1/ARTIFICIAL INTELLIGENCE INFORMATION RETRIEVAL/HW1/datas/"
    else:
        data_path = args.path

    dataset = {}    
    files = []  # [r"datas/e9fd4cbc-7887-11ef-a0fe-769fca1489e4.xml", r"datas/3dea772e-78a5-11ef-a2e9-769fca1489e4.xml"]
    for root, dirnames, filenames in os.walk(data_path):
        print(filenames)
        for filename in filenames:
            if filename.endswith(('xml', 'json')):
                files.append(os.path.join(root, filename))

    if len(files) <= 0:
        print("ERROR, not found any file.")
        exit(0)

    # Load files
    for file in files:
        print(f"\r\nPath: {file}")

        if file.endswith('xml'):
            # AbstractText = extract_text_from_xml(file)
            
            tree = ET.parse(file)
            root = tree.getroot()
            AbstractText = root.find("PubmedArticle").find("MedlineCitation").find("Article").find("Abstract").find("AbstractText").text
            print(f"AbstractText:\r\n{AbstractText}\r\n")
        elif file.endswith('json'):
            continue
            # AbstractText = extract_text_from_json(file)
            # print(f"AbstractText:\r\n{AbstractText}\r\n")

        num_chars, num_words, num_sentences = document_statistics(AbstractText)
        print(f"num_chars: {num_chars}")
        print(f"num_words: {num_words}")
        print(f"num_sentences: {num_sentences}\r\n")
        dataset[file] = AbstractText

    # Search
    while True:
        try:
            print("-" * 30)
            keyword = input("(CTL-C to break) keyword? ")

            for File, AbstractText in dataset.items():
                search_indexes = search_document(AbstractText, keyword)
                # print(f"search_index (keyword=\"{keyword}\"):\r\n{search_indexes}\r\n")

                if len(search_indexes) <= 0:
                    continue
                
                print("")
                print("*" * 30)
                print(f"{File}")
                for index in search_indexes:
                    vResultString = visualization_results(AbstractText, index, keyword)
                    print(f"Position({index[0]}):\r\n{vResultString}\r\n")                

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    os.system("cls")
    main()