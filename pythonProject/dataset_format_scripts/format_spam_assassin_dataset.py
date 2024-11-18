import os
import email
import csv


def parse_email(file_path):
    with open(file_path, 'r', encoding='latin1') as file:
        msg = email.message_from_file(file)

        subject = msg['Subject']
        body = ""

        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode('latin1')
                    break
        else:
            body = msg.get_payload(decode=True).decode('latin1')

        return subject, body


def process_emails_to_csv(input_folder, output_csv):
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['subject', 'body', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for filename in os.listdir(input_folder):
            file_path = os.path.join(input_folder, filename)

            if os.path.isfile(file_path):
                subject, body = parse_email(file_path)

                writer.writerow({'subject': subject, 'body': body, 'label': 'ham'})


input_folder = '../datasets/spamassin'
output_csv = './parsed_datasets/spamassassin_dataset.csv'

process_emails_to_csv(input_folder, output_csv)
