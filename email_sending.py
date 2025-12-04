import os
import base64
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException

# Update function to accept the new 'attachment_bytes'
def send_report_email(text, tags, fen, orig_bytes, crop_bytes, attachment_bytes=None):
    try:
        # Configure API client
        configuration = sib_api_v3_sdk.Configuration()
        configuration.api_key['api-key'] = os.environ['SMTP_KEY']

        api_instance = sib_api_v3_sdk.TransactionalEmailsApi(
            sib_api_v3_sdk.ApiClient(configuration)
        )

        # Build HTML content
        html_content = f"""
        <h2>SnapFen Report</h2>
        <p><strong>Type:</strong> {tags}</p>
        <p><strong>Feedback:</strong> {text}</p>
        <p><strong>FEN (if applicable):</strong> {fen}</p>
        """

        # Prepare attachments
        attachments = []

        # 1. AI Feedback Images (Original & Crop)
        if orig_bytes:
            attachments.append({
                "content": base64.b64encode(orig_bytes).decode(),
                "name": "original_board.png"
            })
        if crop_bytes:
            attachments.append({
                "content": base64.b64encode(crop_bytes).decode(),
                "name": "cropped_board.png"
            })

        # 2. General Bug Screenshot (New Feature)
        if attachment_bytes:
            attachments.append({
                "content": base64.b64encode(attachment_bytes).decode(),
                "name": "bug_screenshot.png"
            })

        # Create Email Object
        email = sib_api_v3_sdk.SendSmtpEmail(
            to=[{"email": os.environ["EMAIL_RECEIVER"]}],
            sender={"email": os.environ["EMAIL_SENDER"]},
            subject=f"[SnapFen] {tags}",
            html_content=html_content,
            attachment=attachments if attachments else None
        )

        # Send
        api_instance.send_transac_email(email)
        print(f"📧 Email sent successfully! (Tags: {tags})")

    except ApiException as e:
        print("Email API Exception:", e)
    except Exception as e:
        print("General Email Error:", e)
