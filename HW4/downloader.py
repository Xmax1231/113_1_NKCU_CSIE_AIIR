from Bio import Entrez
from Bio import Medline
import pickle
import time
import os

def download_abstracts(search_term, email, min_results=100, max_results=200):
    """
    下載 PubMed 文章摘要並以 PMID 為鍵儲存
    
    參數:
    search_term: 搜尋關鍵字
    email: 您的電子郵件
    min_results: 最少下載數量
    max_results: 最大下載數量
    """
    Entrez.email = email
    
    # 搜尋文章
    handle = Entrez.esearch(db="pubmed", 
                          term=search_term,
                          retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    
    # 獲取文章 ID 列表
    id_list = record["IdList"]
    if len(id_list) < min_results:
        print(f"警告：只找到 {len(id_list)} 篇文章，少於要求的最小數量 {min_results}")
    print(f"找到 {len(id_list)} 篇文章")
    
    # 儲存摘要的字典
    abstracts_dict = {}
    
    # 批次下載，每次 50 篇
    batch_size = 50
    for i in range(0, len(id_list), batch_size):
        batch_ids = id_list[i:i + batch_size]
        try:
            # 下載一批文章
            handle = Entrez.efetch(db="pubmed",
                                 id=",".join(batch_ids),
                                 rettype="medline",
                                 retmode="text")
            
            # 解析結果
            records = Medline.parse(handle)
            for record in records:
                pmid = record.get("PMID", "")
                abstract = record.get("AB", "No abstract available")
                if abstract != "No abstract available":  # 只儲存有摘要的文章
                    abstracts_dict[pmid] = abstract
            
            handle.close()
            print(f"已下載 {len(abstracts_dict)} 篇摘要")
            
        except Exception as e:
            print(f"下載時發生錯誤: {str(e)}")
            continue
        
        # 適當延遲以符合 NCBI 規定
        time.sleep(1)
        
        # 如果已經達到最小數量，可以提前結束
        if len(abstracts_dict) >= min_results:
            break
    
    return abstracts_dict

def save_abstracts(abstracts_dict, filename="abstracts.pkl"):
    """
    將摘要儲存為 pickle 檔案
    """
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    # 儲存為 pickle
    with open(filename, 'wb') as f:
        pickle.dump(abstracts_dict, f)
    print(f"已儲存 {len(abstracts_dict)} 篇摘要到 {filename}")

def load_abstracts(filename="abstracts.pkl"):
    """
    載入摘要檔案
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

# 使用示例
if __name__ == "__main__":
    # 設定參數
    search_term = "enterovirus"
    email = "your.email@example.com"  # 請替換為您的電子郵件
    
    # 下載摘要
    abstracts = download_abstracts(search_term, email)
    
    # 儲存檔案
    save_abstracts(abstracts, f"pubmed_data/{search_term}_abstracts.pkl")
    
    # 讀取示例
    loaded_abstracts = load_abstracts(f"pubmed_data/{search_term}_abstracts.pkl")
    print(f"成功載入 {len(loaded_abstracts)} 篇摘要")