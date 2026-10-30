import json

from planner import generate_research_plan
from search import search_web


question = input("Research Question: ")

plan = json.loads(generate_research_plan(question))

print("\nResearch Objective\n")
print(plan)
print(plan["Research Objective"])

print("\nSearch Queries\n")

for q in plan["search_queries"]:
    print("-", q)

all_sources = []

for query in plan["search_queries"]:

    results = search_web(query)

    for r in results:

        all_sources.append(
            {
                "query": query,
                "title": r["title"],
                "url": r["href"],
                "snippet": r["body"]
            }
        )

print("\nCandidate Sources\n")

for i, source in enumerate(all_sources, 1):

    print(f"{i}. {source['title']}")
    print(source["url"])
    print()