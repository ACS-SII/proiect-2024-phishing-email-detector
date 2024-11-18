import re
import csv

input_file = '../datasets/fradulent_emails.txt'
output_file = './parsed_datasets/nigerian_frauds_cleaned_data.csv'

subject_re = re.compile(r"^Subject: (.+)$", re.MULTILINE)
body_re = re.compile(r"\n\n(.+)", re.MULTILINE | re.DOTALL)


def clean_body(body):
    metadata_patterns = [
        r"X-Sieve:.*",
        r"Message-Id:.*",
        r"X-Mailer:.*",
        r"Content-Transfer-Encoding:.*",
        r"X-MIME-Autoconverted:.*"
    ]

    combined_pattern = "|".join(metadata_patterns)
    cleaned_body = re.sub(combined_pattern, "", body, flags=re.MULTILINE).strip()

    return cleaned_body


emails = []

with open(input_file, 'r', encoding='latin-1') as f:
    email_sections = f.read().split("From r")

for email_text in email_sections:
    subject_match = subject_re.search(email_text)
    subject = subject_match.group(1) if subject_match else "No Subject"

    body_match = body_re.search(email_text)
    body = body_match.group(1).strip() if body_match else "No Body"

    emails.append({
        "subject": subject,
        "body": clean_body(body),
        "label": "phishing"
    })

with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ["subject", "body", "label"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(emails)