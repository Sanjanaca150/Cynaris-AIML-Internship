"""
Tools used by the Automated Research Report Agent.
"""

import requests
from bs4 import BeautifulSoup
from crewai.tools import BaseTool


class WebSearchTool(BaseTool):
    name: str = "Web Search"
    description: str = (
        "Search the web for current information about a research topic "
        "and return readable search results."
    )

    def _run(self, query: str) -> str:
        """Perform a lightweight web search using DuckDuckGo HTML."""

        url = "https://html.duckduckgo.com/html/"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
            )
        }

        response = requests.post(
            url,
            data={"q": query},
            headers=headers,
            timeout=15,
        )

        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        results = []

        for result in soup.select(".result")[:5]:
            title = result.select_one(".result__title")
            snippet = result.select_one(".result__snippet")

            if title and snippet:
                results.append(
                    f"TITLE: {title.get_text(' ', strip=True)}\n"
                    f"SUMMARY: {snippet.get_text(' ', strip=True)}"
                )

        if not results:
            return "No search results were found."

        return "\n\n".join(results)