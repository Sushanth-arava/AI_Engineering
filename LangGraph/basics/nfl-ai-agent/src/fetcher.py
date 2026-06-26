"""Fetch NFL scores and headlines from ESPN's free (unofficial) JSON API.

No API key required. These endpoints power espn.com itself, so be polite:
one call each per run is plenty.
"""

import requests

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
HEADERS = {"User-Agent": "Mozilla/5.0 (nfl-agent)"}


def fetch_scores():
    """Return a list of today's games with team names, scores, and status."""
    try:
        r = requests.get(f"{ESPN_BASE}/scoreboard", headers=HEADERS, timeout=15)
        r.raise_for_status()
        events = r.json().get("events", [])
    except Exception as e:
        print(f"Score fetch failed: {e}")
        return []

    games = []
    for ev in events:
        comp = (ev.get("competitions") or [{}])[0]
        status_type = ev.get("status", {}).get("type", {})
        teams = {}
        for c in comp.get("competitors", []):
            teams[c.get("homeAway")] = {
                "name": c.get("team", {}).get("displayName"),
                "abbr": c.get("team", {}).get("abbreviation"),
                "score": c.get("score"),
                "winner": c.get("winner", False),
            }
        games.append({
            "name": ev.get("name"),                      # "Team A at Team B"
            "state": status_type.get("state"),           # pre | in | post
            "completed": status_type.get("completed", False),
            "detail": status_type.get("shortDetail"),    # e.g. "Final", "Sun 1:00 PM"
            "home": teams.get("home"),
            "away": teams.get("away"),
        })
    return games


def fetch_headlines(limit=8):
    """Return the top NFL news headlines (works year-round, incl. offseason)."""
    try:
        r = requests.get(f"{ESPN_BASE}/news", headers=HEADERS, timeout=15)
        r.raise_for_status()
        articles = r.json().get("articles", [])
    except Exception as e:
        print(f"News fetch failed: {e}")
        return []

    headlines = []
    for a in articles[:limit]:
        headlines.append({
            "headline": a.get("headline"),
            "description": a.get("description"),
            "link": (a.get("links", {}).get("web", {}) or {}).get("href"),
        })
    return headlines


def build_brief(games, headlines):
    """Turn raw data into a compact plain-text brief for the LLM to summarize."""
    lines = []

    finals = [g for g in games if g.get("completed")]
    upcoming = [g for g in games if g.get("state") == "pre"]

    if finals:
        lines.append("FINAL SCORES:")
        for g in finals:
            a, h = g["away"], g["home"]
            lines.append(f"- {a['name']} {a['score']} at {h['name']} {h['score']}")

    if upcoming:
        lines.append("UPCOMING:")
        for g in upcoming:
            lines.append(f"- {g['name']} ({g['detail']})")

    if headlines:
        lines.append("HEADLINES:")
        for hl in headlines:
            desc = f" — {hl['description']}" if hl.get("description") else ""
            lines.append(f"- {hl['headline']}{desc}")

    return "\n".join(lines) if lines else "No NFL games or news available right now."


if __name__ == "__main__":
    print(build_brief(fetch_scores(), fetch_headlines()))