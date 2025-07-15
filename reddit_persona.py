import praw
import os
import sys
import datetime
from dotenv import load_dotenv
from tqdm import tqdm
import cohere

load_dotenv()
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

print("Reddit API Loaded:", bool(REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET))
print("Cohere API Key Loaded:", bool(COHERE_API_KEY))

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
)

co = cohere.Client(COHERE_API_KEY)

def extract_username(url_or_handle):
    if url_or_handle.startswith("u/"):
        return url_or_handle.split("/")[-1]
    return url_or_handle.strip("/").split("/")[-1]

def format_date(timestamp):
    return datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")

def fetch_user_data(username):
    try:
        user = reddit.redditor(username)
        _ = user.id
    except Exception as e:
        print(f" Cannot access user '{username}': {e}")
        return []

    data = []

    for post in tqdm(user.submissions.new(limit=50), desc="Fetching Posts"):
        body = post.selftext[:500].replace("\n", " ").strip()
        data.append(f"""POST:
Title: {post.title}
Subreddit: {post.subreddit}
Date: {format_date(post.created_utc)}
Score: {post.score}
URL: https://reddit.com{post.permalink}
Body: {body}""")

    for comment in tqdm(user.comments.new(limit=50), desc="Fetching Comments"):
        try:
            parent = comment.submission
            body = comment.body[:500].replace("\n", " ").strip()
            data.append(f"""COMMENT:
Comment: {body}
Parent Post Title: {parent.title}
Subreddit: {comment.subreddit}
Date: {format_date(comment.created_utc)}
Score: {comment.score}
URL: https://reddit.com{comment.permalink}""")
        except Exception:
            continue

    with open(f"raw_data_{username}.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(data))

    return data

def generate_persona(text_blocks):
    combined = "\n\n".join(text_blocks[:30])
    prompt = f"""
You are a behavioral analyst. Based on the Reddit activity below, generate a professional user persona document for a product design or marketing team.

Use this format:

### User Persona

**Username:** u/{extract_username(sys.argv[1])}

**1. Interests:** Mention relevant subreddits and topics the user engages with.

**2. Most Common Subreddits:** List 3-5 subreddits the user posts in most.

**3. Likely Occupation or Background:** Make a guess using writing style, technical depth, and subreddit activity.

**4. Tone and Communication Style:** Describe how the user communicates (casual, sarcastic, formal, etc.).

**5. Writing Patterns:** Analyze sentence length, structure, use of humor, etc.

**6. Political or Philosophical Leanings:** Mention if the user leans left, right, or is neutral. Use subreddit or quote for proof.

**7. Demographics:** Estimate Age, Gender, Region based on context and community references.

**8. Key Quotes:** List 2-4 quotes that reflect their personality or mindset.

**9. Overall Summary:** Concise narrative combining all insights into one paragraph.

Reddit User Activity:
{combined}
"""

    print("Generating user persona using Cohere...")
    response = co.generate(
        model="command-r-plus",
        prompt=prompt,
        max_tokens=900,
        temperature=0.7
    )

    generated = response.generations[0].text.strip()
    print("\n Persona Output:\n", generated[:1000], "...\n Output Truncated in Terminal\n")
    return generated

def save_persona(username, persona_text):
    filename = f"persona_{username.lower()}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(persona_text)
    print(f"Persona saved to: {filename}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python reddit_persona.py <Reddit Profile URL or u/username>")
        return

    raw_input = sys.argv[1]
    username = extract_username(raw_input)
    print(f"Scraping Reddit user: u/{username}")

    user_data = fetch_user_data(username)
    if not user_data:
        print("No usable Reddit data found.")
        return

    persona = generate_persona(user_data)
    save_persona(username, persona)

if __name__ == "__main__":
    main()
