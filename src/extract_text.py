import requests
from bs4 import BeautifulSoup
import time
import os

BASE_URL_TEMPLATE = "https://www.nasa.gov/history/afj/ap{}fj/"
START_URL_TEMPLATE = "https://www.nasa.gov/history/afj/ap{}fj/index.html"
HEADERS = {"User-Agent": "Mozilla/5.0"}

def fetch_page(url):
    """Fetches the HTML content of a webpage."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return None

def extract_text(html_content):
    """Extracts text from HTML content."""
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator="\n", strip=True)

def extract_links(html_content, base_url):
    """Extracts all valid sublinks from the main page."""
    soup = BeautifulSoup(html_content, "html.parser")
    links = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.startswith("/"):
            href = base_url + href.lstrip("/")
        elif not href.startswith("http"):
            href = base_url + href
        if href.startswith(base_url):
            links.append(href)
    return list(set(links))

def scrape_apollo_mission(mission_id):
    """Scrapes the main page and all sublinks for a given Apollo mission."""
    base_url = BASE_URL_TEMPLATE.format(mission_id)
    start_url = START_URL_TEMPLATE.format(mission_id)
    print(f"Fetching main page: {start_url}")
    
    main_html = fetch_page(start_url)
    if not main_html:
        return
    
    sublinks = extract_links(main_html, base_url)
    print(f"Found {len(sublinks)} sublinks for Apollo {mission_id}.")
    
    # Create output directory
    mission_dir = f"apollo_flight_journals/apollo_{mission_id}"
    os.makedirs(mission_dir, exist_ok=True)
    
    for idx, link in enumerate(sublinks):
        print(f"Scraping {idx+1}/{len(sublinks)}: {link}")
        sub_html = fetch_page(link)
        if sub_html:
            text_content = extract_text(sub_html)
            file_name = f"{mission_dir}/page_{idx+1}.txt"
            with open(file_name, "w", encoding="utf-8") as file:
                file.write(text_content)
            print(f"Saved: {file_name}")
        time.sleep(1)  # Be respectful and avoid overloading the server

def scrape_all_apollo_missions():
    """Scrapes flight journals for Apollo 7 to Apollo 17."""
    for mission_id in range(7, 18):
        if mission_id < 10:
            mission_id = f"0{mission_id}"
        else:
            mission_id = str(mission_id)
        scrape_apollo_mission(mission_id)

if __name__ == "__main__":
    scrape_all_apollo_missions()