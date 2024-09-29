import re
from typing import List
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


def document_statistics(doc: str) -> tuple[int, int, int]:
    num_chars = len(doc)  # len(re.sub(r'\s+', "", doc))
    num_words = len(doc.split())
    sentences = re.split(r'[.!?]', doc)
    num_sentences = len([s for s in sentences if s.strip()])
    return num_chars, num_words, num_sentences


def search_document(doc: str, keyword: str) -> list[tuple[int, int]]:
    """Search for keyword in document files"""

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


def visualization_web_result(doc: str, index: tuple[int, int]) -> str:
    """Visualization part-context for web"""

    doc = doc[:index[1]] + "</mark>" + doc[index[1]:]
    doc = doc[:index[0]] + "<mark>" + doc[index[0]:]

    start_index = index[0] - 30
    end_index = len(doc) - index[1] - 60

    if start_index <= 0:
        return f"{doc[:-end_index]} ..."
    if end_index <= 0:
        return f"... {doc[start_index:]}"
    return f"... {doc[start_index:-end_index]} ..."


def visualization_web_fulltext(doc: str, indexes: List[List[int]]) -> str:
    """Visualization full-context for web"""
    
    indexes.reverse()
    for index in indexes:
        doc = doc[:index[1]] + "</mark>" + doc[index[1]:]
        doc = doc[:index[0]] + "<mark>" + doc[index[0]:]
    return doc