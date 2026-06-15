# AI Mail Assistant Setup

## Where API keys go

Put your Groq API key in `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Do not put this key directly inside `main.py`.

## Where Gmail credentials go

Gmail does not use `.env` for the main OAuth file. Download the Google OAuth desktop client file from Google Cloud, rename it to:

```text
credentials.json
```

Then place it in this project folder, beside `main.py`.

When you click **Connect Gmail**, the app creates:

```text
token.json
```

That file remembers your Gmail login locally.

## Private local files

These files are ignored by git:

```text
.env
credentials.json
token.json
events.json
processed_emails.json
```

They should stay only on your computer.
