import requests
from bs4 import BeautifulSoup


def get_wikipedia_page(query):
    """
    Search for the most relevant Wikipedia page content for the query.
    """
    search_url = "https://en.wikipedia.org/w/api.php"
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
    }

    response = requests.get(search_url, params=search_params)
    if response.status_code != 200:
        print("Search request failed!")
        return None

    search_results = response.json()
    if not search_results["query"]["search"]:
        print("Failed to find any relevant Wikipedia page!")
        return None

    page_title = search_results["query"]["search"][0]["title"]
    print(f"Page Title Found: {page_title}")

    page_url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
    page_response = requests.get(page_url)
    if page_response.status_code != 200:
        print("Failed to retrieve Wikipedia page content!")
        return None

    soup = BeautifulSoup(page_response.text, "html.parser")
    content = soup.find("div", {"id": "mw-content-text"})
    if not content:
        print("Failed to parse Wikipedia page content!")
        return None

    paragraphs = content.find_all("p")
    page_content = "\n".join([p.get_text() for p in paragraphs if p.get_text().strip()])

    return page_title, page_url, page_content
