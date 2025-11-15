from tavily import TavilyClient
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.tools import tool

@tool
def web_search_tool(query: str, max_results: int = 3) -> str:
    """
    Search the web for current information about legal, constitutional, or rights-related topics.
    'query' is the search term to look up on the internet.
    Returns current web search results.
    """
    print(f"\n🌐 [EXECUTE] WEB_SEARCH_TOOL (query={query})")
    try:
        print(f"   ...searching web for: {query}")
        result = search_web(query, max_results)
        print(f"✅ Web search completed")
        return result
    except Exception as e:
        error_msg = f"Error performing web search: {e}"
        print(f"❌ {error_msg}")
        return error_msg

load_dotenv()

class WebSearcher:
    """Web search utility using Tavily API for AI-optimized search results."""
    
    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            print("Warning: TAVILY_API_KEY not found. Web search will use fallback method.")
        
    def search(self, query: str, max_results: int = 5) -> str:
        """
        Search the web for information related to the query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            Formatted search results as a string
        """
        if not self.api_key:
            return self._fallback_search(query)
        
        # Enhance query for better legal search results
        enhanced_query = self._enhance_legal_query(query)
        
        try:            
            client = TavilyClient(api_key=self.api_key)
            
            # Search with context focused on legal/constitutional topics
            response = client.search(
                query=enhanced_query,
                search_depth="advanced",
                max_results=max_results,
                include_domains=["supreme-court.gov.in", "lawmin.gov.in", "indiacode.nic.in", "livelaw.in", "barandbench.com"],
                exclude_domains=["wikipedia.org"]  # Exclude to avoid redundancy with knowledge base
            )
            
            return self._format_results(response.get("results", []), query)
            
        except ImportError:
            print("Tavily package not installed. Install with: pip install tavily-python")
            return self._fallback_search(query)
        except Exception as e:
            print(f"Error during web search: {e}")
            return self._fallback_search(query)
    
    def _enhance_legal_query(self, query: str) -> str:
        """Enhance query with legal context for better results."""
        legal_keywords = ["constitution", "rights", "law", "court", "judgment", "act", "amendment"]
        query_lower = query.lower()
        
        # Add Indian legal context if not already present
        if not any(word in query_lower for word in ["india", "indian", "supreme court"]):
            if any(word in query_lower for word in legal_keywords):
                return f"{query} India"
        
        return query
    
    def _format_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """Format search results into a readable string."""
        if not results:
            return f"No web search results found for: {query}"
        
        formatted_results = [f"Web search results for '{query}':\n"]
        
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            content = result.get("content", "No content available")
            url = result.get("url", "No URL")
            
            # Truncate content for readability
            if len(content) > 300:
                content = content[:300] + "..."
            
            formatted_results.append(
                f"{i}. **{title}**\n"
                f"   {content}\n"
                f"   Source: {url}\n"
            )
        
        return "\n".join(formatted_results)
    
    def _fallback_search(self, query: str) -> str:
        """Fallback method when Tavily API is not available."""
        return (
            f"Web search attempted for: '{query}'\n\n"
            "Note: To enable full web search functionality, please:\n"
            "1. Install tavily-python: pip install tavily-python\n"
            "2. Set TAVILY_API_KEY environment variable\n"
            "3. Get API key from: https://tavily.com/\n\n"
            "For now, please rely on the knowledge base or upload relevant documents."
        )

def search_web(query: str, max_results: int = 5) -> str:
    """Convenience function for web search."""
    searcher = WebSearcher()
    return searcher.search(query, max_results)