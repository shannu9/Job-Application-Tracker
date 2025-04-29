# Install required libraries (first time only)
# pip install --upgrade openai google-api-python-client google-auth-httplib2 google-auth-oauthlib gspread oauth2client pandas beautifulsoup4

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

openai.api_key = OPENAI_API_KEY

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
    query = f'newer_than:2d (subject:application OR subject:interview OR subject:offer OR subject:thank OR subject:hiring)'
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

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response['choices'][0]['message']['content']
    try:
        return json.loads(content)
    except:
        return {"is_job_related": False}


def update_google_sheet(sheet_client, data_rows):
    sheet = sheet_client.open_by_key(GOOGLE_SHEET_ID).sheet1
    records = sheet.get_all_records()
    df = pd.DataFrame(records)

    header = ['Date', 'Company', 'Job Title', 'Status', 'Recruiter Email', 'Email Link', 'Account Email', 'Last Updated']
    if df.empty:
        sheet.append_row(header)
        df = pd.DataFrame(columns=header)

    for new_row in data_rows:
        date, company, job_title, status, recruiter_email, email_link, account_email = new_row
        existing = df[(df['Company'] == company) & (df['Job Title'] == job_title)]
        if not existing.empty:
            index = existing.index[0]
            df.at[index, 'Status'] = status
            df.at[index, 'Recruiter Email'] = recruiter_email
            df.at[index, 'Last Updated'] = datetime.now().strftime('%Y-%m-%d %H:%M')
        else:
            new_entry = {
                'Date': date,
                'Company': company,
                'Job Title': job_title,
                'Status': status,
                'Recruiter Email': recruiter_email,
                'Email Link': email_link,
                'Account Email': account_email,
                'Last Updated': datetime.now().strftime('%Y-%m-%d %H:%M')
            }
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

    df = df.fillna('')
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.values.tolist())


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
                analysis = analyze_email_with_openai(subject, body)
                if analysis.get("is_job_related"):
                    status = analysis.get("status", "Applied")
                    all_data.append([
                        datetime.utcfromtimestamp(internal_date).strftime('%Y-%m-%d'),
                        from_email.split('@')[0],
                        subject,
                        status,
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
