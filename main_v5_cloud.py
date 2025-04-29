# Install required libraries (first time only)
# pip install --upgrade openai google-api-python-client google-auth-httplib2 google-auth-oauthlib gspread oauth2client pandas beautifulsoup4 rapidfuzz

import os
import pickle
import base64
import re
import time
import json
import openai
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from rapidfuzz import fuzz
from collections import Counter
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ---- SETTINGS ----
GMAIL_CREDENTIALS_FILE = 'credentials_gmail.json'
SHEET_CREDENTIALS_FILE = 'credentials_sheets.json'
GOOGLE_SHEET_ID = os.getenv('GOOGLE_SHEET_ID')
GMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
SHEET_SCOPES = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
TOKEN_DIR = 'tokens/'
LAST_PROCESSED_FILE = 'last_processed.txt'
CHECK_INTERVAL_SECONDS = 900  # 15 minutes
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
USE_AI = os.getenv('USE_AI', 'false').lower() == 'true'

openai.api_key = OPENAI_API_KEY
client = openai.OpenAI()

# ---- FUNCTIONS ----

def authenticate_gmail(account_name):
    token_path = os.path.join(TOKEN_DIR, f'token_{account_name}.pickle')
    creds = None
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(GMAIL_CREDENTIALS_FILE, GMAIL_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)
    service = build('gmail', 'v1', credentials=creds)
    return service


def authenticate_google_sheets():
    creds = ServiceAccountCredentials.from_json_keyfile_name(SHEET_CREDENTIALS_FILE, SHEET_SCOPES)
    client = gspread.authorize(creds)
    return client


def get_last_processed_timestamp():
    if os.path.exists(LAST_PROCESSED_FILE):
        with open(LAST_PROCESSED_FILE, 'r') as f:
            return int(f.read().strip())
    else:
        start_date_env = os.getenv('START_DATE', None)
        if not start_date_env:
            raise Exception("Environment variable START_DATE is required on first run!")
        dt = datetime.strptime(start_date_env, '%Y-%m-%d')
        timestamp = int(dt.replace(tzinfo=timezone.utc).timestamp())
        return timestamp


def save_last_processed_timestamp(timestamp):
    with open(LAST_PROCESSED_FILE, 'w') as f:
        f.write(str(timestamp))


def search_new_emails(service, after_timestamp):
    query = '(subject:application OR subject:interview OR subject:offer OR subject:thank OR subject:hiring)'
    response = service.users().messages().list(userId='me', q=query, maxResults=100).execute()
    messages = response.get('messages', [])

    while 'nextPageToken' in response:
        page_token = response['nextPageToken']
        response = service.users().messages().list(userId='me', q=query, maxResults=100, pageToken=page_token).execute()
        messages.extend(response.get('messages', []))

    filtered_messages = []
    for m in messages:
        msg = service.users().messages().get(userId='me', id=m['id'], format='metadata').execute()
        internal_date = int(msg['internalDate']) / 1000
        if internal_date > after_timestamp:
            filtered_messages.append(m)

    return filtered_messages


def extract_email_data(service, msg_id):
    msg = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
    headers = msg['payload']['headers']
    subject = [h['value'] for h in headers if h['name'] == 'Subject']
    from_email = [h['value'] for h in headers if h['name'] == 'From']
    link = f"https://mail.google.com/mail/u/0/#inbox/{msg_id}"

    parts = msg['payload'].get('parts', [])
    body = ""
    for part in parts:
        if part['mimeType'] == 'text/html':
            data = part['body']['data']
            decoded_data = base64.urlsafe_b64decode(data).decode('utf-8')
            soup = BeautifulSoup(decoded_data, "html.parser")
            body = soup.get_text()
            break

    subject = subject[0] if subject else 'No Subject'
    from_email = from_email[0] if from_email else 'No From'

    return subject, body, from_email, link, int(msg['internalDate']) / 1000


def analyze_email_with_openai(subject, body):
    prompt = f"""
    Analyze the following email and determine:
    1. Is it related to a job application? (yes/no)
    2. If yes, what is the status? (Applied, Interview Scheduled, Offer, Rejected)

    Email Subject: {subject}
    Email Body: {body}

    Respond in JSON format:
    {{"is_job_related": true/false, "status": "Applied/Interview Scheduled/Offer/Rejected"}}
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except Exception as e:
        print(f"Error parsing OpenAI response: {e}")
        return {"is_job_related": False}


def rule_based_classify(subject, body):
    text = f"{subject} {body}".lower()
    if any(kw in text for kw in ["offer", "we are pleased to", "congratulations"]):
        return {"is_job_related": True, "status": "Offer"}
    if any(kw in text for kw in ["interview", "zoom", "meet", "scheduled", "calendar invite"]):
        return {"is_job_related": True, "status": "Interview Scheduled"}
    if any(kw in text for kw in ["not moving forward", "unfortunately", "we regret", "thank you for your interest"]):
        return {"is_job_related": True, "status": "Rejected"}
    if any(kw in text for kw in ["applied", "application", "submission received", "thank you for applying", "you applied"]):
        return {"is_job_related": True, "status": "Applied"}
    return {"is_job_related": False}


def find_matching_row(df, company, job_title):
    for index, row in df.iterrows():
        if row['Company'].lower() == company.lower():
            similarity = fuzz.partial_ratio(row['Job Title'].lower(), job_title.lower())
            if similarity > 85:
                return index
    return None


def create_or_update_dashboard(sheet_client):
    sheet = sheet_client.open_by_key(GOOGLE_SHEET_ID)
    records = sheet.worksheet("Sheet1").get_all_records()
    df = pd.DataFrame(records)

    # Status Summary
    status_counts = Counter(df['Status'])
    status_section = [["Status", "Count"]] + [[k, v] for k, v in status_counts.items()]

    # Add spacing row
    spacer = [[], []]

    # Detection Mode Summary
    mode_counts = Counter(df['Detection Mode'])
    mode_section = [["Detection Mode", "Count"]] + [[k, v] for k, v in mode_counts.items()]

    # Combine all parts
    dashboard_data = status_section + spacer + mode_section

    try:
        dash = sheet.worksheet("Dashboard")
    except:
        dash = sheet.add_worksheet(title="Dashboard", rows="20", cols="2")

    dash.clear()
    dash.update(dashboard_data)
    sheet = sheet_client.open_by_key(GOOGLE_SHEET_ID)
    records = sheet.worksheet("Sheet1").get_all_records()
    df = pd.DataFrame(records)
    status_counts = Counter(df['Status'])

    dashboard_data = [["Status", "Count"]] + [[k, v] for k, v in status_counts.items()]

    try:
        dash = sheet.worksheet("Dashboard")
    except:
        dash = sheet.add_worksheet(title="Dashboard", rows="10", cols="2")

    dash.clear()
    dash.update(dashboard_data)


def update_google_sheet(sheet_client, data_rows):
    sheet = sheet_client.open_by_key(GOOGLE_SHEET_ID).sheet1
    records = sheet.get_all_records()
    df = pd.DataFrame(records)

    header = ['Date', 'Company', 'Job Title', 'Status', 'Detection Mode', 'Recruiter Email', 'Email Link', 'Account Email', 'Last Updated']
    if df.empty:
        sheet.append_row(header)
        df = pd.DataFrame(columns=header)

    for new_row in data_rows:
        date, company, job_title, status, mode, recruiter_email, email_link, account_email = new_row
        match_index = find_matching_row(df, company, job_title)
        if match_index is not None:
            df.at[match_index, 'Status'] = status
            df.at[match_index, 'Detection Mode'] = mode
            df.at[match_index, 'Recruiter Email'] = recruiter_email
            df.at[match_index, 'Last Updated'] = datetime.now().strftime('%Y-%m-%d %H:%M')
        else:
            new_entry = {
                'Date': date,
                'Company': company,
                'Job Title': job_title,
                'Status': status,
                'Detection Mode': mode,
                'Recruiter Email': recruiter_email,
                'Email Link': email_link,
                'Account Email': account_email,
                'Last Updated': datetime.now().strftime('%Y-%m-%d %H:%M')
            }
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

    df = df.fillna('')
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.values.tolist())
    create_or_update_dashboard(sheet_client)


def main():
    os.makedirs(TOKEN_DIR, exist_ok=True)
    account_names = os.getenv('GMAIL_ACCOUNTS', '').split(',')
    account_names = [acc.strip() for acc in account_names if acc.strip()]

    if not account_names:
        raise Exception("GMAIL_ACCOUNTS environment variable is required!")

    all_data = []
    last_timestamp = get_last_processed_timestamp()

    while True:
        for account_name in account_names:
            print(f"Connecting to {account_name}...")
            service = authenticate_gmail(account_name)
            messages = search_new_emails(service, last_timestamp)

            print(f"Found {len(messages)} new job-related emails in {account_name}")

            for m in messages:
                subject, body, from_email, link, internal_date = extract_email_data(service, m['id'])
                if USE_AI:
                    analysis = analyze_email_with_openai(subject, body)
                    mode = "GPT-3.5"
                else:
                    analysis = rule_based_classify(subject, body)
                    mode = "Rule-Based"

                if analysis.get("is_job_related"):
                    status = analysis.get("status", "Applied")
                    all_data.append([
                        datetime.fromtimestamp(internal_date, tz=timezone.utc).strftime('%Y-%m-%d'),
                        from_email.split('@')[0],
                        subject,
                        status,
                        mode,
                        from_email,
                        link,
                        account_name
                    ])
                    last_timestamp = max(last_timestamp, internal_date)

        if all_data:
            sheets_client = authenticate_google_sheets()
            update_google_sheet(sheets_client, all_data)
            save_last_processed_timestamp(last_timestamp)
            all_data = []

        print(f"Sleeping for {CHECK_INTERVAL_SECONDS/60} minutes...")
        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == '__main__':
    main()
