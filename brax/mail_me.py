import os
import smtplib
from email.message import EmailMessage


EMAIL_USER = os.environ["MAIL_USER"]
EMAIL_PASS = os.environ["MAIL_PASS"]
EMAIL_TO = os.environ["MAIL_TO"]

SUBJECT = os.environ.get("MAIL_SUBJECT", "PROJET_CSI_4900")
BODY = os.environ.get(
    "MAIL_BODY",
    """Salut,

Votre pipeline a fini de s'executer sur Compute Canada.

SVP, allez voir vos resultats.

Message automatique.
""",
)

msg = EmailMessage()
msg["From"] = EMAIL_USER
msg["To"] = EMAIL_TO
msg["Subject"] = SUBJECT
msg.set_content(BODY)

with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as smtp:
    smtp.login(EMAIL_USER, EMAIL_PASS)
    smtp.send_message(msg)

print("Email envoyé avec succès.")