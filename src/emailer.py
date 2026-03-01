"""Email notification via QQ SMTP."""

import re
import smtplib
import time
from collections import OrderedDict
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape
from email.utils import formataddr
from urllib.parse import quote

import markdown

from . import config

_MD_EXTENSIONS = ["tables", "fenced_code", "nl2br", "sane_lists"]

_EMAIL_CSS = """\
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    font-size: 15px;
    line-height: 1.7;
    color: #1a1a1a;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}
h2 {
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 8px;
    margin-top: 32px;
}
h3 {
    color: #34495e;
    margin-top: 24px;
}
h3 small {
    color: #7f8c8d;
    font-weight: normal;
}
h4 { color: #555; margin-top: 18px; }
hr {
    border: none;
    border-top: 1px solid #e0e0e0;
    margin: 28px 0;
}
strong { color: #c0392b; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
}
th, td {
    border: 1px solid #ddd;
    padding: 8px 12px;
    text-align: left;
}
th {
    background: #f5f6fa;
    font-weight: 600;
}
tr:nth-child(even) { background: #fafafa; }
pre {
    background: #f4f4f4;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 12px 16px;
    overflow-x: auto;
    font-size: 13px;
    line-height: 1.5;
}
code {
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 13px;
}
p code {
    background: #f0f0f0;
    padding: 2px 5px;
    border-radius: 3px;
}
blockquote {
    border-left: 4px solid #3498db;
    margin: 12px 0;
    padding: 8px 16px;
    background: #f8f9fa;
    color: #555;
}
ul, ol { padding-left: 24px; }
li { margin-bottom: 4px; }
"""


def _md_to_html(md_text: str) -> str:
    """Convert Markdown to styled HTML, rendering LaTeX math as images.

    Processing order: extract LaTeX → markdown convert → restore as <img>.
    This prevents the markdown engine from corrupting backslash escapes in LaTeX.
    """
    # Step 1: Extract LaTeX expressions and replace with placeholders
    latex_map = {}
    counter = 0

    def _stash(match):
        nonlocal counter
        key = f"\x00LATEX{counter}\x00"
        counter += 1
        latex_map[key] = match.group(0)
        return key

    # Extract $$...$$ (block) before $...$ (inline) to avoid conflicts
    text = re.sub(r"\$\$(.+?)\$\$", _stash, md_text, flags=re.DOTALL)
    text = re.sub(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", _stash, text)

    # Step 2: Convert markdown (LaTeX is safely stashed away)
    html = markdown.markdown(text, extensions=_MD_EXTENSIONS)

    # Step 3: Replace placeholders with rendered <img> tags (PNG for email compat)
    for key, original in latex_map.items():
        if original.startswith("$$"):
            latex_content = original[2:-2]
            img = (
                f'<div style="text-align:center;margin:12px 0">'
                f'<img src="https://latex.codecogs.com/png.latex?\\dpi{{150}}%20{quote(latex_content)}"'
                f' alt="{escape(latex_content)}" style="vertical-align:middle"></div>'
            )
        else:
            latex_content = original[1:-1]
            img = (
                f'<img src="https://latex.codecogs.com/png.latex?\\dpi{{150}}\\inline%20{quote(latex_content)}"'
                f' alt="{escape(latex_content)}" style="vertical-align:middle">'
            )
        html = html.replace(key, img)

    return html


class Emailer:
    """Send course summary emails via QQ SMTP SSL."""

    def __init__(self):
        self.host = config.SMTP_HOST
        self.port = config.SMTP_PORT
        self.sender = config.SMTP_EMAIL
        self.password = config.SMTP_PASSWORD
        self.receiver = config.RECEIVER_EMAIL

    def send(self, items: list[dict]) -> bool:
        """Send a single email containing all lecture summaries.

        Args:
            items: List of dicts, each with keys:
                   course_title, sub_title, date, summary

        Returns:
            True if email was sent successfully, False otherwise.
        """
        if not items:
            return True

        # Group by course (preserve insertion order)
        courses: OrderedDict[str, list[dict]] = OrderedDict()
        for item in items:
            courses.setdefault(item["course_title"], []).append(item)

        # Subject
        parts = [f"{ct} ({len(lecs)})" for ct, lecs in courses.items()]
        subject = f"[iCourse 课程内容更新] {', '.join(parts)}"

        # Plain text (Markdown as-is, readable without rendering)
        plain_sections = []
        for course_title, lectures in courses.items():
            plain_sections.append(f"{'=' * 40}")
            plain_sections.append(f"课程：{course_title}")
            plain_sections.append(f"{'=' * 40}")
            for lec in lectures:
                plain_sections.append(
                    f"\n--- {lec['sub_title']} ({lec['date']}) ---\n"
                )
                plain_sections.append(lec["summary"])
        plain = "\n".join(plain_sections)

        # HTML (Markdown → styled HTML with LaTeX rendering)
        body_parts = []
        for course_title, lectures in courses.items():
            body_parts.append(f"<h2>{escape(course_title)}</h2>")
            for lec in lectures:
                body_parts.append(
                    f"<h3>{escape(lec['sub_title'])} "
                    f"<small>({escape(lec['date'])})</small></h3>"
                )
                body_parts.append(_md_to_html(lec["summary"]))
                body_parts.append("<hr>")

        html = (
            "<!DOCTYPE html>"
            "<html><head><meta charset='utf-8'>"
            f"<style>{_EMAIL_CSS}</style>"
            "</head><body>"
            + "\n".join(body_parts)
            + "</body></html>"
        )

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = formataddr(("iCourse Subscriber", self.sender))
        msg["To"] = self.receiver
        msg.attach(MIMEText(plain, "plain", "utf-8"))
        msg.attach(MIMEText(html, "html", "utf-8"))

        # Retry with exponential backoff
        for attempt in range(3):
            try:
                with smtplib.SMTP_SSL(self.host, self.port) as server:
                    server.login(self.sender, self.password)
                    server.sendmail(self.sender, self.receiver, msg.as_string())
                print(f"[Emailer] Sent: {subject}")
                return True
            except Exception as e:
                print(f"[Emailer] Attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)

        print("[Emailer] All send attempts failed.")
        return False
