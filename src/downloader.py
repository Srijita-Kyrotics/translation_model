import requests
from bs4 import BeautifulSoup
import os
import time
from tqdm import tqdm
import urllib3
import ssl
from concurrent.futures import ThreadPoolExecutor, as_completed

# Disable insecure request warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class LegacyRenegotiationAdapter(requests.adapters.HTTPAdapter):
    """An adapter that allows legacy SSL renegotiation."""
    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        kwargs['ssl_context'] = context
        return super(LegacyRenegotiationAdapter, self).init_poolmanager(*args, **kwargs)

def download_file(session, url, folder):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    
    file_name = url.split("/")[-1]
    file_path = os.path.join(folder, file_name)
    
    if os.path.exists(file_path):
        return file_path
    
    try:
        response = session.get(url, stream=True, timeout=30, verify=False)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return file_path
    except Exception as e:
        pass # Silent error for bulk downloads to avoid cluttering tqdm
    return None

def process_pair(session, urls, en_dir, bn_dir):
    """Worker function to download a single pair."""
    download_file(session, urls['english'], en_dir)
    download_file(session, urls['bengali'], bn_dir)

def main():
    base_url = "https://calcuttahighcourt.gov.in"
def get_pdf_links(session, url, base_url="https://calcuttahighcourt.gov.in"):
    """Fetch and return all PDF links from a given URL."""
    print(f"Fetching links from {url}...")
    try:
        response = session.get(url, timeout=30, verify=False)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch page {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', href=True)
    
    pdf_links = []
    for link in links:
        href = link['href']
        if href.endswith('.pdf'):
            full_url = href if href.startswith('http') else base_url + href
            pdf_links.append(full_url)
    
    print(f"Found {len(pdf_links)} PDF links on {url}.")
    return pdf_links

def main():
    base_url = "https://calcuttahighcourt.gov.in"
    source_urls = [
        "https://calcuttahighcourt.gov.in/show_judgements_sc", # Supreme Court
        "https://calcuttahighcourt.gov.in/show_judgements_hc", # High Court
        "https://calcuttahighcourt.gov.in/show_judgements_i"   # Historically Important
    ]
    
    session = requests.Session()
    session.mount("https://", LegacyRenegotiationAdapter())
    
    all_pdf_links = []
    for url in source_urls:
        all_pdf_links.extend(get_pdf_links(session, url, base_url))
        time.sleep(1) # Polite delay between pages
    
    print(f"Aggregated {len(all_pdf_links)} total PDF links across all categories.")

    # Match pairs
    pairs = {}
    for full_url in all_pdf_links:
        file_name = full_url.split("/")[-1]
        
        # Base name is everything before _e.pdf or _b.pdf
        if file_name.endswith('_e.pdf'):
            base = file_name[:-6]
            if base not in pairs: pairs[base] = {}
            pairs[base]['english'] = full_url
        elif file_name.endswith('_b.pdf'):
            base = file_name[:-6]
            if base not in pairs: pairs[base] = {}
            pairs[base]['bengali'] = full_url

    valid_pairs = {k: v for k, v in pairs.items() if 'english' in v and 'bengali' in v}
    print(f"Identified {len(valid_pairs)} valid Bengali-English pairs.")

    # Download
    data_dir = os.path.join("data", "raw", "judgments")
    en_dir = os.path.join(data_dir, "english")
    bn_dir = os.path.join(data_dir, "bengali")

    # Use ThreadPoolExecutor for concurrent downloads
    # 5 workers is a safe balance to avoid server issues
    print("Starting concurrent downloads...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_pair, session, urls, en_dir, bn_dir) for urls in valid_pairs.values()]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading pairs"):
            pass

    print(f"Download complete. Files saved in {data_dir}")

if __name__ == "__main__":
    main()
