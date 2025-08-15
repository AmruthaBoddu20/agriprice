import requests
from bs4 import BeautifulSoup


# Define a function to fetch Google Scholar content
def fetch_scholar_results(query):
    # Create the URL for the Google Scholar search
    query = query.replace(' ', '+')
    url = f'https://scholar.google.com/scholar?q={query}'

    # Send a request to the URL
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code != 200:
        print(f"Error fetching page: {response.status_code}")
        return None

    # Parse the page content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the results (titles, links, etc.)
    results = soup.find_all('div', class_='gs_ri')

    # Extract information for each result
    scholar_results = []
    for result in results:
        title = result.find('h3', class_='gs_rt').text if result.find('h3', class_='gs_rt') else "No Title"
        link = result.find('a')['href'] if result.find('a') else "No Link"
        snippet = result.find('div', class_='gs_rs').text if result.find('div', class_='gs_rs') else "No Snippet"
        citation_info = result.find('div', class_='gs_fl').text if result.find('div', class_='gs_fl') else "No Citation Info"
        
        scholar_results.append({
            'title': title,
            'link': link,
            'snippet': snippet,
            'citation_info': citation_info
        })

    return scholar_results

# Use the function to fetch results for a query
query = 'machine learning'
results = fetch_scholar_results(query)

# Print the results
for index, result in enumerate(results, start=1):
    print(f"{index}. {result['title']}")
    print(f"   Link: {result['link']}")
    print(f"   Snippet: {result['snippet']}")
    print(f"   Citation Info: {result['citation_info']}")
    print()