from duckduckgo_search import DDGS


def search_web(query, max_results=5):

    with DDGS() as ddgs:

        return list(
            ddgs.text(
                query,
                max_results=max_results
            )
        )