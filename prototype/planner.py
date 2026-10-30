from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()
client = OpenAI()


def generate_research_plan(question: str):

    prompt = f"""
You are an expert research planner.

Given a user's question, produce:

1. Research Objective
2. Five Search Queries

Return JSON.

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"}
    )

    return response.choices[0].message.content