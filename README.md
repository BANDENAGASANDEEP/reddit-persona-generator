# Reddit Persona Generator

A Python-based tool that generates detailed user personas from public Reddit profiles using the Reddit API and Cohere LLM. Ideal for UX research, marketing, and product development.

---

## What It Does

- Scrapes a Reddit user’s most recent posts and comments using the Reddit API.
- Uses Cohere’s large language model to analyze the user’s online behavior and generate a structured persona profile.
- Produces:
  - A professional persona report (`persona_<username>.txt`)
  - A raw activity log for reference (`raw_data_<username>.txt`)

---

## Tested Profiles and Output

We tested the script with:

- `python reddit_persona.py https://www.reddit.com/user/kojied/`
- `python reddit_persona.py https://www.reddit.com/user/Hungry-Move-6603/`

Generated files:

- `persona_kojied.txt`
- `raw_data_kojied.txt`
- `persona_hungry-move-6603.txt`
- `raw_data_Hungry-Move-6603.txt`

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/BANDENAGASANDEEP/reddit-persona-generator.git
cd reddit-persona-generator
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure API Keys

Create a `.env` file in the root directory using the `.env.example` file as a template:

```bash
cp .env.example .env
```

Add your keys:

```
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_user_agent
COHERE_API_KEY=your_cohere_api_key
```

---

## Run the Script

You can run the script using either a full Reddit profile URL or a handle:

```bash
python reddit_persona.py https://www.reddit.com/user/kojied/
python reddit_persona.py https://www.reddit.com/user/Hungry-Move-6603/
```

This will generate:

* `raw_data_<username>.txt`
* `persona_<username>.txt`

---
> ⏳ **Note:** The first time you run the script, generating the user persona may take **2–5 minutes**.  
> This delay is due to the time required by Cohere’s language model to analyze and generate a detailed persona.  
> Please be patient — the output will be saved once processing is complete.


## Output Structure

Each persona file includes:

* Interests
* Most Common Subreddits
* Likely Background or Occupation
* Tone and Communication Style
* Writing Patterns
* Political or Philosophical Leanings
* Demographic Estimations
* Key Quotes
* Overall Summary

---

## Dependencies

* `praw` – Reddit API wrapper
* `cohere` – LLM for persona generation
* `dotenv` – Load environment variables
* `tqdm` – Progress bar for data fetching

---

## .gitignore

The following files and folders are excluded from Git:

```
.env
venv/
__pycache__/
*.pyc
```

---

## License

This project is licensed under the MIT License.

```
Let me know if you'd like this uploaded as your `README.md` file on GitHub or want help generating `requirements.txt` or `.env.example`.
```

