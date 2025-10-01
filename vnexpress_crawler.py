# vnexpress_crawler.py
import time
import json
import requests
from urllib.parse import urljoin
from urllib import robotparser
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from newspaper import Article
import feedparser
from datetime import datetime

# ---------- CONFIG ----------
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
ROBOTS_URL = "https://vnexpress.net/robots.txt"
RSS_FEEDS = [
    "https://vnexpress.net/rss/tin-moi-nhat.rss",
    "https://vnexpress.net/rss/thoi-su.rss",
    "https://vnexpress.net/rss/the-gioi.rss",
    "https://vnexpress.net/rss/kinh-doanh.rss",
]
RATE_LIMIT_SECONDS = 1.0   # adjust to be polite
OUTPUT_FILE = "vnexpress_articles.jsonl"
MAX_ARTICLES = 1000        # cap for testing
# ----------------------------

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

def can_fetch(url, robots_url=ROBOTS_URL):
    rp = robotparser.RobotFileParser()
    rp.set_url(robots_url)
    rp.read()
    return rp.can_fetch(USER_AGENT, url)

def parse_sitemap_recursive(sitemap_url):
    """Return generator of urls found in sitemap (handles sitemap index)."""
    try:
        resp = session.get(sitemap_url, timeout=15)
        resp.raise_for_status()
        # Ensure proper encoding and remove any invalid XML characters
        xml = resp.content.decode('utf-8', errors='ignore')
        # Parse XML with error handling
        try:
            root = ET.fromstring(xml)
            ns = {"sm":"http://www.sitemaps.org/schemas/sitemap/0.9"}
            # sitemap index?
            for sitemap in root.findall("sm:sitemap", ns):
                loc = sitemap.find("sm:loc", ns).text
                if loc:
                    yield from parse_sitemap_recursive(loc)
            # urlset?
            for url in root.findall("sm:url", ns):
                loc = url.find("sm:loc", ns).text
                if loc:
                    yield loc
        except ET.ParseError as e:
            print(f"XML parsing error for {sitemap_url}: {str(e)}")
    except requests.exceptions.RequestException as e:
        print(f"Request error for {sitemap_url}: {str(e)}")
        return

def fetch_rss_urls(rss_url):
    feed = feedparser.parse(rss_url)
    for entry in feed.entries:
        yield entry.link

def extract_meta_description(html):
    soup = BeautifulSoup(html, "lxml")
    m = soup.find("meta", {"name":"description"})
    if m and m.get("content"):
        return m["content"].strip()
    og = soup.find("meta", {"property":"og:description"})
    if og and og.get("content"):
        return og["content"].strip()
    return None

def crawl_article(url):
    # quick robots check
    if not can_fetch(url):
        print("Blocked by robots:", url)
        return None
    try:
        # Use newspaper3k for robust article extraction
        art = Article(url, language="vi")
        art.download()
        art.parse()
        # optional NLP (summarize) - skip for now or use art.nlp()
        # art.nlp()
        meta_desc = extract_meta_description(art.html or "")
        return {
            "url": url,
            "title": art.title,
            "authors": art.authors,
            "publish_date": art.publish_date.isoformat() if art.publish_date else None,
            "content": art.text,
            "summary_meta": meta_desc,
            "top_image": art.top_image,
            "keywords": art.keywords,
            "crawl_date": datetime.utcnow().isoformat(),
            "raw_html_snippet": (art.html[:2000] if art.html else None)
        }
    except Exception as e:
        print("Failed to fetch:", url, e)
        return None

def main_from_rss(max_articles=MAX_ARTICLES):
    count = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for rss_feed in RSS_FEEDS:
            try:
                print(f"Processing RSS feed: {rss_feed}")
                for url in fetch_rss_urls(rss_feed):
                    if count >= max_articles:
                        print(f"Reached maximum article limit ({max_articles})")
                        return
                    
                    print("Trying:", url)
                    obj = crawl_article(url)
                    if obj and obj.get("content"):
                        fout.write(json.dumps(obj, ensure_ascii=False)+"\n")
                        count += 1
                        print("Saved:", count, url)
                    time.sleep(RATE_LIMIT_SECONDS)
                    
            except Exception as e:
                print(f"Error processing RSS feed {rss_feed}: {str(e)}")
                continue
                
    print("Done. saved", count, "articles to", OUTPUT_FILE)

if __name__ == "__main__":
    main_from_rss()
