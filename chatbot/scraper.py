import requests
from bs4 import BeautifulSoup
import json
import time

BASE_URL = "https://indiankanoon.org"
SEARCH_URL = BASE_URL + "/search/?formInput=murder&type=law&p={}"

def get_document_links(pages):
    links = []
    for page in range(1, pages + 1):
        url = SEARCH_URL.format(page)
        print(f"Scraping: {url}")
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, 'html.parser')
        results = soup.select(".result_title a")
        
        for res in results:
            doc_url = BASE_URL + res['href']
            title = res.get_text(strip=True)
            links.append({"title": title, "url": doc_url})
        
        time.sleep(2)
    return links

def extract_document_text(url):
    resp = requests.get(url, timeout=10)
    soup = BeautifulSoup(resp.text, 'html.parser')
    section = soup.find("section", class_="akn-section")
    content_span = section.find("span", class_="akn-content")
    full_text = content_span.get_text(separator="\n", strip=True)[:1000]
    summary = full_text[:100] + "..." if len(full_text) > 300 else full_text
    return summary, full_text

def scrape_and_save(filename="/Users/mac/Desktop/Projects/Taxmann/data/murder_cases.json", max_docs=100):
    links = get_document_links(1) 
    records = []
    for i, item in enumerate(links):
        if i >= max_docs:
            break
        print(f"[{i+1}] Scraping {item['url']}")
        summary, full_text = extract_document_text(item['url'])
        records.append({
            "title": item['title'],
            "url": item['url'],
            "summary": summary,
            "full_text": full_text
        })
        time.sleep(2)
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Scrapping done..Saved {len(records)} documents to {filename}")

if __name__ == "__main__":
    scrape_and_save()