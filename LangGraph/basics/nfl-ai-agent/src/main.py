

from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from fetcher import fetch_scores, fetch_headlines, build_brief
from analyzer import summarize
from sms import send_sms



def run_agent():
    print(f"[{datetime.now().isoformat()}] Running NFL agent...")

    games = fetch_scores()
    headlines = fetch_headlines()
    brief = build_brief(games, headlines)
    print("---- BRIEF ----")
    print(brief)

    if not headlines and not any(g.get("completed") for g in games):
        print("Nothing newsworthy today; skipping send.")
        return

    message = summarize(brief)
    print("---- MESSAGE ----")
    print(message)

    try:
        sid = send_sms(message)
        print("WhatsApp message sent! SID:", sid)
    except Exception as e:
        print("Send failed:", e)


if __name__ == "__main__":
    run_agent()