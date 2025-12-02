import os
import base64
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException

# IMPORTANT: Brevo API Key must be in environment variable BREVO_API_KEY

def send_report_email(text, tags, fen, orig_bytes, crop_bytes):
    try:
        # Configure API client
        configuration = sib_api_v3_sdk.Configuration()
        configuration.api_key['api-key'] = os.environ['SMTP_KEY']

        api_instance = sib_api_v3_sdk.TransactionalEmailsApi(
            sib_api_v3_sdk.ApiClient(configuration)
        )

        # Build HTML content
        html_content = f"""
        <h2>SnapFen Issue Report</h2>
        <p><strong>Tags:</strong> {tags}</p>
        <p><strong>Feedback:</strong> {text}</p>
        <p><strong>FEN:</strong> {fen}</p>
        """

        # Prepare attachments
        attachments = []

        if orig_bytes:
            attachments.append({
                "content": base64.b64encode(orig_bytes).decode(),
                "name": "original.png"
            })

        if crop_bytes:
            attachments.append({
                "content": base64.b64encode(crop_bytes).decode(),
                "name": "cropped.png"
            })

        email = sib_api_v3_sdk.SendSmtpEmail(
            to=[{"email": os.environ["EMAIL_RECEIVER"]}],
            sender={"email": os.environ["EMAIL_SENDER"]},
            subject="[SnapFen Report] " + str(tags),
            html_content=html_content,
            attachment=attachments
        )

        api_instance.send_transac_email(email)
        print("📧 Email sent successfully")

    except ApiException as e:
        print("Email API Exception:", e)
    except Exception as e:
        print("General Email Error:", e)
