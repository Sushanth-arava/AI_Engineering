"""Summarize the NFL brief into a short, WhatsApp-friendly text message using OpenAI."""

import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def summarize(brief: str) -> str:
    prompt = f"""You are writing a short daily NFL update to send as a WhatsApp text.
Using ONLY the information below, write a tidy, scannable message.

Format:
- Start with a one-line header like: *🏈 NFL Daily — <today's vibe in 3-4 words>*
- Then 4-6 short lines, each a single key result or headline (no fluff).
- WhatsApp bold is *single asterisks*. No markdown headers, no tables.
- Keep the whole thing under 700 characters.
- If it's the offseason with no games, just summarize the top news.

DATA:
{brief}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


if __name__ == "__main__":
    from fetcher import fetch_scores, fetch_headlines, build_brief
    print(summarize(build_brief(fetch_scores(), fetch_headlines())))
