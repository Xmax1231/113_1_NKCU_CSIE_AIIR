import requests
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from IPython.display import clear_output

def search_pubmed(query, max_results=5):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    # Perform the search
    search_url = f"{base_url}esearch.fcgi?db=pubmed&term={query}&retmax={max_results}&retmode=xml"
    response = requests.get(search_url)
    root = ET.fromstring(response.content)
    
    # Extract PMIDs from the search results
    pmids = [id_elem.text for id_elem in root.findall(".//IdList/Id")]
    
    return pmids

def fetch_pubmed_xml(pmid):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    # Fetch XML for each PMID
    fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
    response = requests.get(fetch_url)
    folder = './pubmed'
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = '{}/{}.xml'.format(folder, pmid)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(response.text)
    
    return response.text

def main():
    query = input("Enter your PubMed search query: ")
    max_results = int(input("Enter the maximum number of results to retrieve: "))
    
    print(f"Searching PubMed for: {query}")
    pmids = search_pubmed(query, max_results)
    
    if not pmids:
        print("No results found.")
        return
    
    print(f"Found {len(pmids)} results. Fetching XML data...")

    for pmid in tqdm(pmids):
        try:
            xml_data = fetch_pubmed_xml(pmid)
            
            print(xml_data[:100] + "..." if len(xml_data) > 1000 else xml_data)  
            clear_output(wait=True)  
        except Exception as e:
            print(f"Error fetching data for PMID {pmid}: {e}")    

if __name__ == "__main__":
    main()