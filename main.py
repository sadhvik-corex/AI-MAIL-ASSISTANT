from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path
from typing import Any
import base64
import json
import os
import uuid

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configure Groq
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

MODEL_NAME = "llama-3.1-8b-instant"
EVENTS_FILE = Path("events.json")
GMAIL_CREDENTIALS_FILE = Path("credentials.json")
GMAIL_TOKEN_FILE = Path("token.json")
PROCESSED_EMAILS_FILE = Path("processed_emails.json")
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1)


class EmailQuestionRequest(BaseModel):
    email_text: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)


class EmailScheduleRequest(BaseModel):
    email_text: str = Field(..., min_length=1)


class EmailScheduleSaveRequest(BaseModel):
    email_text: str = Field(..., min_length=1)


class EventRequest(BaseModel):
    title: str = Field(..., min_length=1)
    date: str = ""
    date_iso: str = ""
    time: str = ""
    duration: str = ""
    location: str = ""
    notes: str = ""
    event_type: str = "event"


class DraftEmailRequest(BaseModel):
    purpose: str = Field(..., min_length=1)
    recipient: str | None = None
    tone: str = "professional"


class GmailSyncRequest(BaseModel):
    max_results: int = Field(10, ge=1, le=25)
    query: str = "newer_than:30d"


def load_events() -> list[dict[str, Any]]:
    if not EVENTS_FILE.exists():
        return []

    try:
        return json.loads(EVENTS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_events(events: list[dict[str, Any]]) -> None:
    EVENTS_FILE.write_text(
        json.dumps(events, indent=2),
        encoding="utf-8"
    )


def load_processed_email_ids() -> set[str]:
    if not PROCESSED_EMAILS_FILE.exists():
        return set()

    try:
        return set(json.loads(PROCESSED_EMAILS_FILE.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        return set()


def save_processed_email_ids(email_ids: set[str]) -> None:
    PROCESSED_EMAILS_FILE.write_text(
        json.dumps(sorted(email_ids), indent=2),
        encoding="utf-8"
    )


def save_schedule_events(schedule: dict[str, Any]) -> list[dict[str, Any]]:
    saved_events = []
    events = load_events()

    for event in schedule.get("events", []):
        saved_event = {
            "id": str(uuid.uuid4()),
            "title": event.get("title", "Untitled event"),
            "date": event.get("date", ""),
            "date_iso": event.get("date_iso", ""),
            "time": event.get("time", ""),
            "duration": event.get("duration", ""),
            "location": event.get("location", ""),
            "notes": event.get("notes", ""),
            "event_type": event.get("event_type", "event")
        }
        events.append(saved_event)
        saved_events.append(saved_event)

    save_events(events)
    return saved_events


def save_schedule_events_from_email(
    schedule: dict[str, Any],
    email_id: str,
    subject: str
) -> list[dict[str, Any]]:
    saved_events = []
    events = load_events()

    for event in schedule.get("events", []):
        saved_event = {
            "id": str(uuid.uuid4()),
            "title": event.get("title", "Untitled event"),
            "date": event.get("date", ""),
            "date_iso": event.get("date_iso", ""),
            "time": event.get("time", ""),
            "duration": event.get("duration", ""),
            "location": event.get("location", ""),
            "notes": event.get("notes", ""),
            "event_type": event.get("event_type", "event"),
            "source": "gmail",
            "source_email_id": email_id,
            "source_subject": subject
        }
        events.append(saved_event)
        saved_events.append(saved_event)

    save_events(events)
    return saved_events


def event_date(event: dict[str, Any]) -> datetime | None:
    date_iso = event.get("date_iso", "")
    if not date_iso:
        return None

    try:
        return datetime.strptime(date_iso, "%Y-%m-%d")
    except ValueError:
        return None


def upcoming_events(days: int = 7) -> list[dict[str, Any]]:
    today = datetime.now().date()
    upcoming = []

    for event in load_events():
        parsed_date = event_date(event)
        if parsed_date is None:
            continue

        days_until = (parsed_date.date() - today).days
        if 0 <= days_until <= days:
            event_with_alert = {
                **event,
                "days_until": days_until,
                "alert": alert_text(event, days_until)
            }
            upcoming.append(event_with_alert)

    return sorted(upcoming, key=lambda event: event["days_until"])


def alert_text(event: dict[str, Any], days_until: int) -> str:
    title = event.get("title", "Event")
    event_type = event.get("event_type", "event")

    if days_until == 0:
        return f"{title} is today."
    if days_until == 1:
        return f"{title} is tomorrow."
    return f"{title} {event_type} is in {days_until} days."


def ask_groq(messages: list[dict[str, str]]) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Missing GROQ_API_KEY in .env."
        )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return response.choices[0].message.content


def parse_ai_json(content: str) -> dict:
    cleaned = content.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start:end + 1]

    return json.loads(cleaned)


def gmail_credentials() -> Credentials | None:
    if not GMAIL_TOKEN_FILE.exists():
        return None

    creds = Credentials.from_authorized_user_file(
        str(GMAIL_TOKEN_FILE),
        GMAIL_SCOPES
    )

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        GMAIL_TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

    return creds


def gmail_service():
    creds = gmail_credentials()
    if not creds or not creds.valid:
        raise HTTPException(
            status_code=401,
            detail="Gmail is not connected. Add credentials.json and use /gmail/connect."
        )

    return build("gmail", "v1", credentials=creds)


def message_header(message: dict[str, Any], name: str) -> str:
    headers = message.get("payload", {}).get("headers", [])
    for header in headers:
        if header.get("name", "").lower() == name.lower():
            return header.get("value", "")
    return ""


def decode_message_body(payload: dict[str, Any]) -> str:
    body_data = payload.get("body", {}).get("data")
    if body_data:
        return base64.urlsafe_b64decode(body_data).decode("utf-8", errors="ignore")

    for part in payload.get("parts", []):
        mime_type = part.get("mimeType", "")
        if mime_type == "text/plain":
            text = decode_message_body(part)
            if text:
                return text

    for part in payload.get("parts", []):
        text = decode_message_body(part)
        if text:
            return text

    return ""


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Mail Assistant</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #5f6b7a;
      --line: #d9dee7;
      --accent: #0f766e;
      --accent-dark: #0b5f59;
      --danger: #b42318;
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      padding: 20px 24px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 {
      margin: 0;
      font-size: 24px;
      line-height: 1.2;
    }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(280px, 0.8fr);
      gap: 16px;
      width: min(1180px, 100%);
      margin: 0 auto;
      padding: 16px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    h2 {
      margin: 0 0 12px;
      font-size: 17px;
    }
    label {
      display: block;
      margin: 12px 0 6px;
      color: var(--muted);
      font-size: 13px;
    }
    textarea,
    input,
    select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      font: inherit;
      background: #fff;
      color: var(--text);
    }
    textarea {
      min-height: 160px;
      resize: vertical;
    }
    button {
      border: 0;
      border-radius: 6px;
      padding: 10px 12px;
      background: var(--accent);
      color: #fff;
      font: inherit;
      cursor: pointer;
    }
    button:hover {
      background: var(--accent-dark);
    }
    button.secondary {
      background: #e9eef3;
      color: var(--text);
    }
    button.secondary:hover {
      background: #dce3eb;
    }
    button.danger {
      background: #fee4e2;
      color: var(--danger);
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    pre,
    .answer,
    .event {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      padding: 12px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .event {
      margin-top: 10px;
    }
    .alert {
      border-color: #f6c85f;
      background: #fff8e5;
    }
    .event strong {
      display: block;
      margin-bottom: 4px;
    }
    .muted {
      color: var(--muted);
      font-size: 13px;
    }
    .status {
      display: inline-block;
      margin-top: 8px;
      padding: 4px 8px;
      border-radius: 6px;
      background: #eef2f6;
      color: var(--muted);
      font-size: 13px;
    }
    .status.connected {
      background: #dcfce7;
      color: #166534;
    }
    @media (max-width: 820px) {
      main,
      .grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>AI Mail Assistant</h1>
  </header>
  <main>
    <div>
      <section>
        <h2>Gmail Sync</h2>
        <div id="gmailStatus" class="status">Checking Gmail...</div>
        <label for="gmailQuery">Gmail search</label>
        <input id="gmailQuery" value="newer_than:30d">
        <label for="gmailMaxResults">Emails to scan</label>
        <input id="gmailMaxResults" type="number" min="1" max="25" value="10">
        <div class="actions">
          <button onclick="connectGmail()">Connect Gmail</button>
          <button class="secondary" onclick="syncGmail()">Sync Gmail</button>
        </div>
        <pre id="gmailOutput">Gmail is not synced yet.</pre>
      </section>

      <section style="margin-top:16px">
        <h2>Email Workspace</h2>
        <label for="emailText">Email text</label>
        <textarea id="emailText">Hi Sai, let us meet on June 20 at 3 PM for the project demo in Room 204. Please block 1 hour.</textarea>
        <div class="actions">
          <button onclick="extractSchedule()">Extract schedule</button>
          <button onclick="saveSchedule()">Save to schedule</button>
        </div>
        <label>Extracted schedule</label>
        <pre id="scheduleOutput">No schedule extracted yet.</pre>
      </section>

      <section style="margin-top:16px">
        <h2>Ask About This Email</h2>
        <label for="emailQuestion">Question</label>
        <input id="emailQuestion" value="Where is the demo?">
        <div class="actions">
          <button onclick="askAboutEmail()">Ask</button>
        </div>
        <label>Answer</label>
        <div class="answer" id="emailAnswer">No answer yet.</div>
      </section>

      <section style="margin-top:16px">
        <h2>Draft Email</h2>
        <div class="grid">
          <div>
            <label for="draftPurpose">Purpose</label>
            <input id="draftPurpose" value="Ask my teacher to reschedule the project demo">
          </div>
          <div>
            <label for="draftTone">Tone</label>
            <select id="draftTone">
              <option>professional</option>
              <option>friendly</option>
              <option>short</option>
              <option>formal</option>
            </select>
          </div>
        </div>
        <label for="draftRecipient">Recipient</label>
        <input id="draftRecipient" value="teacher">
        <div class="actions">
          <button onclick="draftEmail()">Draft</button>
        </div>
        <label>Draft</label>
        <div class="answer" id="draftOutput">No draft yet.</div>
      </section>
    </div>

    <section>
      <h2>Upcoming Alerts</h2>
      <div class="actions">
        <button class="secondary" onclick="enableNotifications()">Enable notifications</button>
        <button class="secondary" onclick="loadUpcoming()">Check now</button>
      </div>
      <div id="alertsOutput" class="muted">Checking upcoming dates...</div>

      <div style="height:16px"></div>
      <h2>Saved Schedule</h2>
      <div class="actions">
        <button class="secondary" onclick="loadEvents()">Refresh</button>
      </div>
      <div id="eventsOutput" class="muted">Loading events...</div>
    </section>
  </main>

  <script>
    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...options
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Request failed");
      }
      return data;
    }

    function emailText() {
      return document.getElementById("emailText").value;
    }

    async function loadGmailStatus() {
      const status = document.getElementById("gmailStatus");
      try {
        const data = await api("/gmail/status");
        status.textContent = data.connected
          ? "Gmail connected"
          : data.message;
        status.className = data.connected ? "status connected" : "status";
      } catch (error) {
        status.textContent = error.message;
        status.className = "status";
      }
    }

    async function connectGmail() {
      const output = document.getElementById("gmailOutput");
      output.textContent = "Opening Google sign-in...";
      try {
        const data = await api("/gmail/connect", { method: "POST" });
        output.textContent = JSON.stringify(data, null, 2);
        await loadGmailStatus();
      } catch (error) {
        output.textContent = error.message;
      }
    }

    async function syncGmail() {
      const output = document.getElementById("gmailOutput");
      output.textContent = "Scanning Gmail...";
      try {
        const data = await api("/gmail/sync", {
          method: "POST",
          body: JSON.stringify({
            query: document.getElementById("gmailQuery").value,
            max_results: Number(document.getElementById("gmailMaxResults").value)
          })
        });
        output.textContent = JSON.stringify(data, null, 2);
        await loadEvents();
        await loadUpcoming();
      } catch (error) {
        output.textContent = error.message;
      }
    }

    async function extractSchedule() {
      const output = document.getElementById("scheduleOutput");
      output.textContent = "Extracting...";
      try {
        const data = await api("/email/schedule", {
          method: "POST",
          body: JSON.stringify({ email_text: emailText() })
        });
        output.textContent = JSON.stringify(data.schedule, null, 2);
      } catch (error) {
        output.textContent = error.message;
      }
    }

    async function saveSchedule() {
      const output = document.getElementById("scheduleOutput");
      output.textContent = "Saving...";
      try {
        const data = await api("/email/schedule/save", {
          method: "POST",
          body: JSON.stringify({ email_text: emailText() })
        });
        output.textContent = JSON.stringify(data.schedule, null, 2);
        await loadEvents();
        await loadUpcoming();
      } catch (error) {
        output.textContent = error.message;
      }
    }

    async function askAboutEmail() {
      const output = document.getElementById("emailAnswer");
      output.textContent = "Thinking...";
      try {
        const data = await api("/email/answer", {
          method: "POST",
          body: JSON.stringify({
            email_text: emailText(),
            question: document.getElementById("emailQuestion").value
          })
        });
        output.textContent = data.answer;
      } catch (error) {
        output.textContent = error.message;
      }
    }

    async function draftEmail() {
      const output = document.getElementById("draftOutput");
      output.textContent = "Writing...";
      try {
        const data = await api("/email/draft", {
          method: "POST",
          body: JSON.stringify({
            purpose: document.getElementById("draftPurpose").value,
            recipient: document.getElementById("draftRecipient").value,
            tone: document.getElementById("draftTone").value
          })
        });
        output.textContent = data.draft;
      } catch (error) {
        output.textContent = error.message;
      }
    }

    async function loadEvents() {
      const output = document.getElementById("eventsOutput");
      try {
        const data = await api("/events");
        if (!data.events.length) {
          output.className = "muted";
          output.textContent = "No saved events yet.";
          return;
        }

        output.className = "";
        output.innerHTML = data.events.map(event => `
          <div class="event">
            <strong>${escapeHtml(event.title || "Untitled event")}</strong>
            <div>${escapeHtml(event.date || event.date_iso || "No date")} ${escapeHtml(event.time || "")}</div>
            <div class="muted">${escapeHtml(event.event_type || "event")}</div>
            <div class="muted">${escapeHtml(event.duration || "")} ${escapeHtml(event.location || "")}</div>
            <div>${escapeHtml(event.notes || "")}</div>
            <div class="actions">
              <button class="danger" onclick="deleteEvent('${event.id}')">Delete</button>
            </div>
          </div>
        `).join("");
      } catch (error) {
        output.textContent = error.message;
      }
    }

    async function loadUpcoming() {
      const output = document.getElementById("alertsOutput");
      try {
        const data = await api("/events/upcoming?days=7");
        if (!data.events.length) {
          output.className = "muted";
          output.textContent = "No dates coming up in the next 7 days.";
          return;
        }

        output.className = "";
        output.innerHTML = data.events.map(event => `
          <div class="event alert">
            <strong>${escapeHtml(event.alert)}</strong>
            <div>${escapeHtml(event.date || event.date_iso || "No date")} ${escapeHtml(event.time || "")}</div>
            <div class="muted">${escapeHtml(event.notes || "")}</div>
          </div>
        `).join("");
        notifyUpcoming(data.events);
      } catch (error) {
        output.textContent = error.message;
      }
    }

    async function enableNotifications() {
      if (!("Notification" in window)) {
        alert("This browser does not support notifications.");
        return;
      }
      await Notification.requestPermission();
      await loadUpcoming();
    }

    function notifyUpcoming(events) {
      if (!("Notification" in window) || Notification.permission !== "granted") {
        return;
      }

      for (const event of events) {
        const key = `notified-${event.id}-${event.days_until}`;
        if (localStorage.getItem(key)) {
          continue;
        }
        new Notification("AI Mail Assistant", {
          body: event.alert
        });
        localStorage.setItem(key, "yes");
      }
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    async function deleteEvent(eventId) {
      await api(`/events/${eventId}`, { method: "DELETE" });
      await loadEvents();
      await loadUpcoming();
    }

    loadGmailStatus();
    loadEvents();
    loadUpcoming();
    setInterval(loadUpcoming, 60 * 60 * 1000);
  </script>
</body>
</html>
    """


# Simple AI route kept for quick browser testing.
@app.get("/ask_ai")
def ask_ai(question: str):
    answer = ask_groq([
        {"role": "user", "content": question}
    ])

    return {"answer": answer}


@app.post("/ask_ai")
def ask_ai_post(request: QuestionRequest):
    answer = ask_groq([
        {"role": "user", "content": request.question}
    ])

    return {"answer": answer}


@app.post("/email/answer")
def answer_email_question(request: EmailQuestionRequest):
    answer = ask_groq([
        {
            "role": "system",
            "content": (
                "You are an AI mail assistant. Answer questions using only the "
                "email content provided. If the answer is not in the email, say "
                "that the email does not mention it."
            )
        },
        {
            "role": "user",
            "content": (
                f"Email:\n{request.email_text}\n\n"
                f"Question:\n{request.question}"
            )
        }
    ])

    return {"answer": answer}


@app.post("/email/schedule")
def schedule_from_email(request: EmailScheduleRequest):
    schedule = ask_groq([
        {
            "role": "system",
            "content": (
                f"Today's date is {datetime.now().date().isoformat()}. "
                "You extract scheduling information from emails, including "
                "assignment submission dates, deadlines, exams, meetings, "
                "interviews, reminders, and events. Return only valid JSON "
                "with this shape: "
                '{"events":[{"title":"","event_type":"","date":"","date_iso":"",'
                '"time":"","duration":"","location":"","notes":""}],'
                '"missing_details":[]}. Use YYYY-MM-DD for date_iso when the '
                "date can be understood. Use empty strings when a detail is "
                "not provided."
            )
        },
        {
            "role": "user",
            "content": f"Extract schedule dates and events from this email:\n{request.email_text}"
        }
    ])

    try:
        schedule = parse_ai_json(schedule)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="AI returned schedule data that was not valid JSON."
        )

    return {"schedule": schedule}


@app.post("/email/schedule/save")
def save_schedule_from_email(request: EmailScheduleSaveRequest):
    schedule = schedule_from_email(EmailScheduleRequest(email_text=request.email_text))["schedule"]
    saved_events = save_schedule_events(schedule)

    return {
        "schedule": schedule,
        "saved_events": saved_events
    }


@app.get("/gmail/status")
def gmail_status():
    if not GMAIL_CREDENTIALS_FILE.exists():
        return {
            "connected": False,
            "message": "Add credentials.json, then connect Gmail."
        }

    creds = gmail_credentials()
    if creds and creds.valid:
        return {
            "connected": True,
            "message": "Gmail connected."
        }

    return {
        "connected": False,
        "message": "Gmail credentials found. Click Connect Gmail."
    }


@app.post("/gmail/connect")
def connect_gmail():
    if not GMAIL_CREDENTIALS_FILE.exists():
        raise HTTPException(
            status_code=400,
            detail="Missing credentials.json from Google Cloud OAuth client."
        )

    flow = InstalledAppFlow.from_client_secrets_file(
        str(GMAIL_CREDENTIALS_FILE),
        GMAIL_SCOPES
    )
    creds = flow.run_local_server(port=0)
    GMAIL_TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

    return {"connected": True, "message": "Gmail connected successfully."}


@app.get("/gmail/messages")
def gmail_messages(max_results: int = 10, query: str = "newer_than:30d"):
    service = gmail_service()
    result = service.users().messages().list(
        userId="me",
        q=query,
        maxResults=max_results
    ).execute()

    messages = []
    for item in result.get("messages", []):
        message = service.users().messages().get(
            userId="me",
            id=item["id"],
            format="metadata",
            metadataHeaders=["Subject", "From", "Date"]
        ).execute()
        messages.append({
            "id": message["id"],
            "subject": message_header(message, "Subject"),
            "from": message_header(message, "From"),
            "date": message_header(message, "Date"),
            "snippet": message.get("snippet", "")
        })

    return {"messages": messages}


@app.post("/gmail/sync")
def sync_gmail(request: GmailSyncRequest):
    service = gmail_service()
    processed_email_ids = load_processed_email_ids()
    result = service.users().messages().list(
        userId="me",
        q=request.query,
        maxResults=request.max_results
    ).execute()

    synced = []
    skipped = []

    for item in result.get("messages", []):
        email_id = item["id"]
        if email_id in processed_email_ids:
            skipped.append(email_id)
            continue

        message = service.users().messages().get(
            userId="me",
            id=email_id,
            format="full"
        ).execute()
        subject = message_header(message, "Subject")
        sender = message_header(message, "From")
        sent_date = message_header(message, "Date")
        body = decode_message_body(message.get("payload", {}))
        email_text = (
            f"Subject: {subject}\n"
            f"From: {sender}\n"
            f"Date: {sent_date}\n\n"
            f"{body or message.get('snippet', '')}"
        )

        schedule = schedule_from_email(
            EmailScheduleRequest(email_text=email_text)
        )["schedule"]

        saved_events = save_schedule_events_from_email(
            schedule,
            email_id=email_id,
            subject=subject
        )
        processed_email_ids.add(email_id)
        synced.append({
            "email_id": email_id,
            "subject": subject,
            "saved_events": saved_events,
            "missing_details": schedule.get("missing_details", [])
        })

    save_processed_email_ids(processed_email_ids)

    return {
        "scanned": len(result.get("messages", [])),
        "synced": synced,
        "skipped": skipped
    }


@app.get("/events")
def get_events():
    return {"events": load_events()}


@app.get("/events/upcoming")
def get_upcoming_events(days: int = 7):
    return {
        "days": days,
        "events": upcoming_events(days)
    }


@app.post("/events")
def create_event(request: EventRequest):
    events = load_events()
    event = {
        "id": str(uuid.uuid4()),
        "title": request.title,
        "date": request.date,
        "date_iso": request.date_iso,
        "time": request.time,
        "duration": request.duration,
        "location": request.location,
        "notes": request.notes,
        "event_type": request.event_type
    }
    events.append(event)
    save_events(events)

    return {"event": event}


@app.delete("/events/{event_id}")
def delete_event(event_id: str):
    events = load_events()
    remaining_events = [
        event for event in events
        if event.get("id") != event_id
    ]

    if len(remaining_events) == len(events):
        raise HTTPException(status_code=404, detail="Event not found.")

    save_events(remaining_events)
    return {"deleted": event_id}


@app.post("/email/draft")
def draft_email(request: DraftEmailRequest):
    draft = ask_groq([
        {
            "role": "system",
            "content": "You write clear, useful emails. Return only the email body."
        },
        {
            "role": "user",
            "content": (
                f"Write an email with this purpose: {request.purpose}\n"
                f"Recipient: {request.recipient or 'not specified'}\n"
                f"Tone: {request.tone}"
            )
        }
    ])

    return {"draft": draft}
