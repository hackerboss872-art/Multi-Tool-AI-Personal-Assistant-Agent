"""
Local MCP Server for Research Assistant.

Provides research and utility tools over STDIO:
1. web_search - Live web search using DuckDuckGo (DDGS)
2. arxiv_search - Academic paper search using arXiv API
3. fetch_web_page - Extract readable text content from a web URL
4. calculator - Mathematical expression evaluator
5. get_current_datetime - Current date and time
6. get_weather - Live weather lookup for any city
"""

from datetime import datetime
import html
import re
import urllib.parse
import xml.etree.ElementTree as ET

from ddgs import DDGS
from mcp.server.fastmcp import FastMCP
import requests

# Initialize FastMCP Server
mcp = FastMCP("ResearchAssistantServer")


# ============================================================================
# 1. WEB SEARCH TOOL
# ============================================================================

@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the live web using DuckDuckGo for general information,
    articles, documentation, and news.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 5).

    Returns:
        Formatted text containing search result titles, snippets, and URLs.
    """
    try:
        max_results = max(1, min(int(max_results), 10))
        results = list(DDGS().text(query, max_results=max_results))

        if not results:
            return f"No web search results found for query: '{query}'"

        output = [f"Web search results for: '{query}':\n"]
        for i, item in enumerate(results, start=1):
            title = item.get("title", "No Title").strip()
            body = item.get("body", "No description available.").strip()
            url = item.get("href", "").strip()
            output.append(f"{i}. Title: {title}\n   Snippet: {body}\n   URL: {url}\n")

        return "\n".join(output)

    except Exception as e:
        return f"Web search error: {str(e)}"


# ============================================================================
# 2. ARXIV ACADEMIC RESEARCH TOOL
# ============================================================================

@mcp.tool()
def arxiv_search(query: str, max_results: int = 5) -> str:
    """
    Search arXiv for scholarly articles, scientific preprints, and research papers.

    Args:
        query: The academic topic or keywords to search for.
        max_results: Maximum number of research papers to return (default 5).

    Returns:
        Formatted text with paper titles, authors, published dates, summaries, and links.
    """
    try:
        max_results = max(1, min(int(max_results), 10))
        encoded_query = urllib.parse.quote_plus(query)
        api_url = (
            f"http://export.arxiv.org/api/query?"
            f"search_query=all:{encoded_query}&start=0&max_results={max_results}"
        )

        response = requests.get(api_url, timeout=15)
        response.raise_for_status()

        root = ET.fromstring(response.text)
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", namespace)

        if not entries:
            return f"No academic papers found on arXiv for: '{query}'"

        output = [f"arXiv academic papers for: '{query}':\n"]

        for idx, entry in enumerate(entries, start=1):
            title_elem = entry.find("atom:title", namespace)
            summary_elem = entry.find("atom:summary", namespace)
            published_elem = entry.find("atom:published", namespace)
            id_elem = entry.find("atom:id", namespace)

            title = " ".join(title_elem.text.split()) if title_elem is not None and title_elem.text else "Untitled"
            summary = " ".join(summary_elem.text.split()) if summary_elem is not None and summary_elem.text else "No abstract."
            published = published_elem.text[:10] if published_elem is not None and published_elem.text else "Unknown date"
            url = id_elem.text.strip() if id_elem is not None and id_elem.text else ""

            # Extract authors
            author_elems = entry.findall("atom:author", namespace)
            authors = []
            for author in author_elems:
                name_elem = author.find("atom:name", namespace)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text.strip())
            author_str = ", ".join(authors[:5]) if authors else "Unknown Authors"
            if len(authors) > 5:
                author_str += " et al."

            # Truncate summary if too long for token budget
            if len(summary) > 400:
                summary = summary[:400] + "..."

            output.append(
                f"{idx}. Title: {title}\n"
                f"   Authors: {author_str}\n"
                f"   Published: {published}\n"
                f"   URL: {url}\n"
                f"   Abstract: {summary}\n"
            )

        return "\n".join(output)

    except Exception as e:
        return f"arXiv search error: {str(e)}"


# ============================================================================
# 3. WEB PAGE CONTENT FETCHER
# ============================================================================

@mcp.tool()
def fetch_web_page(url: str) -> str:
    """
    Fetch and extract clean readable text from a web page URL.

    Args:
        url: The HTTP or HTTPS URL to read.

    Returns:
        The extracted text content from the web page (truncated to 3000 chars).
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()

        # Simple HTML tag stripping
        text = resp.text
        # Remove scripts and styles
        text = re.sub(r"<script[^>]*>[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<style[^>]*>[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
        # Remove tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Decode entities and clean whitespace
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return "No readable text content found at URL."

        max_len = 3000
        if len(text) > max_len:
            text = text[:max_len] + "... [truncated]"

        return f"Content extracted from {url}:\n\n{text}"

    except Exception as e:
        return f"Failed to fetch web page at {url}: {str(e)}"


# ============================================================================
# 4. CALCULATOR TOOL
# ============================================================================

@mcp.tool()
def calculator(expression: str) -> str:
    """
    Evaluate basic mathematical calculations safely.

    Args:
        expression: A mathematical expression string, e.g. '25 * 48', '15% of 800', '(100 + 50) / 2'.

    Returns:
        The numeric calculation result or an error message.
    """
    try:
        expr = (
            str(expression)
            .lower()
            .replace("of", "*")
            .replace("%", "/100")
            .replace("^", "**")
        )

        allowed = set("0123456789+-*/(). ")
        sanitized = "".join(c for c in expr if c in allowed)

        if not sanitized.strip():
            return "Error: Invalid mathematical expression."

        result = eval(
            sanitized,
            {"__builtins__": {}},
            {}
        )

        return f"Calculation Result: {result}"

    except Exception as err:
        return f"Calculator Error: {str(err)}"


# ============================================================================
# 5. CURRENT DATE AND TIME TOOL
# ============================================================================

@mcp.tool()
def get_current_datetime() -> str:
    """
    Get the current date, time, and timezone information.

    Returns:
        Current local date and time in human-readable and ISO formats.
    """
    now = datetime.now()
    readable = now.strftime("%A, %B %d, %Y %I:%M:%S %p")
    iso = now.isoformat()
    return f"Current Date and Time: {readable} (ISO: {iso})"


# ============================================================================
# 6. WEATHER TOOL
# ============================================================================

@mcp.tool()
def get_weather(city: str) -> str:
    """
    Get current weather information for a specific city.

    Args:
        city: The name of the city, e.g. 'Chandigarh', 'London', 'Tokyo'.

    Returns:
        Current weather details including temperature, condition, humidity, and wind.
    """
    cleaned_city = city.strip()
    if not cleaned_city:
        return "Error: City name cannot be empty."

    # Try wttr.in first
    try:
        encoded_city = urllib.parse.quote_plus(cleaned_city)
        headers = {"User-Agent": "ResearchAssistant/1.0"}
        resp = requests.get(
            f"https://wttr.in/{encoded_city}?format=j1",
            headers=headers,
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            current = data["current_condition"][0]
            temp_c = current.get("temp_C", "N/A")
            temp_f = current.get("temp_F", "N/A")
            feels_c = current.get("FeelsLikeC", "N/A")
            condition = current.get("weatherDesc", [{}])[0].get("value", "N/A")
            humidity = current.get("humidity", "N/A")
            wind_kmph = current.get("windspeedKmph", "N/A")
            wind_dir = current.get("winddir16Point", "")

            # Get matched location name if available
            area_info = data.get("nearest_area", [{}])[0]
            area_name = area_info.get("areaName", [{}])[0].get("value", cleaned_city)
            country = area_info.get("country", [{}])[0].get("value", "")
            location_str = f"{area_name}, {country}".strip(", ")

            return (
                f"Current Weather in {location_str}:\n"
                f"- Temperature: {temp_c}°C ({temp_f}°F)\n"
                f"- Feels Like: {feels_c}°C\n"
                f"- Condition: {condition}\n"
                f"- Humidity: {humidity}%\n"
                f"- Wind: {wind_kmph} km/h {wind_dir}\n"
                f"- URL: https://wttr.in/{encoded_city}\n"
            )
        elif resp.status_code in (404, 500) and "not found" in resp.text.lower():
            return f"Weather error: Could not find weather data for city '{cleaned_city}'. Please verify the city name."
    except Exception:
        pass

    # Fallback to Open-Meteo
    try:
        geo_url = (
            f"https://geocoding-api.open-meteo.com/v1/search?"
            f"name={urllib.parse.quote_plus(cleaned_city)}&count=1"
        )
        geo_resp = requests.get(geo_url, timeout=10)
        geo_data = geo_resp.json()
        results = geo_data.get("results")
        if not results:
            return f"Weather error: City '{cleaned_city}' not found."

        loc = results[0]
        name = loc.get("name", cleaned_city)
        country = loc.get("country", "")
        lat = loc["latitude"]
        lon = loc["longitude"]

        w_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
        )
        w_resp = requests.get(w_url, timeout=10)
        w_data = w_resp.json().get("current", {})

        temp = w_data.get("temperature_2m", "N/A")
        humidity = w_data.get("relative_humidity_2m", "N/A")
        wind = w_data.get("wind_speed_10m", "N/A")

        return (
            f"Current Weather in {name}, {country}:\n"
            f"- Temperature: {temp}°C\n"
            f"- Humidity: {humidity}%\n"
            f"- Wind Speed: {wind} km/h\n"
            f"- URL: https://open-meteo.com\n"
        )
    except Exception as err:
        return f"Weather service error: {str(err)}"


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")

