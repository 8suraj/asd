from flask import (
    Flask,
    render_template,
    request,
    url_for,
    redirect,
    session,
    g,
    jsonify,
    send_from_directory,
    make_response,
    abort,
)
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    create_refresh_token,
    set_access_cookies,
    set_refresh_cookies,
    jwt_required,
    get_jwt_identity,
    get_jwt,
)
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from flask_session import Session
from flask.sessions import SecureCookieSessionInterface
from pymongo import MongoClient, DESCENDING
from vehicle_marking_system import get_details_from_code
from urllib.parse import parse_qs
from dotenv import load_dotenv
from dateutil import parser
import bcrypt
import datetime
import os
import re
import html
import math
import json
import logging
import bleach
import random
import requests
import secrets
import pytz
import uuid


# semantic search library imports
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


# Load environment variables from .env file
load_dotenv()

# Get the MongoDB URI from the environment variable
mongo_uri = os.getenv("MONGO_URI")


client = MongoClient(mongo_uri)
app = Flask(__name__)


app.config["SECRET_KEY"] = secrets.token_hex(40)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SECURE"] = True
app.config["SESSION_PERMANENT"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Strict"
app.config["SESSION_TYPE"] = "filesystem"
app.config["PERMANENT_SESSION_LIFETIME"] = datetime.timedelta(days=1)


# Initialize Flask-Session
app.session_interface = SecureCookieSessionInterface()
Session(app)

# Initialize Qdrant client
qdrant_client = QdrantClient(url="http://localhost:6333")

# Initialize your embeddings model
embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en")


# IPinfo API token
IPINFO_TOKEN = "c4586694fdba29"


## report generation library & function code starts here

# report generation library imports
from fpdf import FPDF
import torch
import spacy
import google.generativeai as genai
from collections import Counter
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from sentence_transformers import CrossEncoder
import ast
from math import cos, sin, radians


# ------------------ Gemini Setup ------------------
# genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
genai.configure(api_key=os.getenv("GOOGLE_API_KEY="))
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# ------------------ Load Models ------------------
nlp = spacy.load("en_core_web_trf")
embedding_model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-en",
    # model_kwargs={"device": "cuda:0"},
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
reranker_model = CrossEncoder(
    model_name_or_path="BAAI/bge-reranker-large", max_length=128
)


# ------------------ Helper Functions ------------------


def parse_date_safe(date_str):
    for fmt in ("%Y-%m-%d", "%B %d, %Y"):
        try:
            return datetime.datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def clean_invalid_json(s):
    return re.sub(r"[\x00-\x1F\x7F]", "", s)


def extract_named_entities(text):
    doc = nlp(text)
    return {
        "persons": [ent.text for ent in doc.ents if ent.label_ == "PERSON"],
        "organizations": [ent.text for ent in doc.ents if ent.label_ == "ORG"],
        "locations": list(
            set([ent.text for ent in doc.ents if ent.label_ in {"GPE", "LOC"}])
        ),
    }


def extract_topics(summary):
    """Extracts list items after 'Main Topics' section from summary text."""
    topics = re.findall(r"\*\s+(.*?)($|\n|\*)", summary)
    cleaned = [t[0].strip() for t in topics if len(t[0].strip()) > 2]
    return list(dict.fromkeys(cleaned))


def extract_key_events(summary):
    """Extracts events with dates from summary text."""
    events = []
    matches = re.findall(r"\*\*\s*(.*?)\s*\*\*:\s*(.*?)($|\n|\*)", summary, re.DOTALL)
    for m in matches:
        date = m[0].strip()
        event = m[1].strip()
        if date and event:
            events.append({"date": date, "event": event})
    return events


def extract_tenders(summary):
    """Extracts tender information from the summary."""
    tenders = []
    matches = re.findall(
        r"Tender Title: (.*?)\nIssue Date: (.*?)\nClosing Date: (.*?)\nLocation: (.*?)\nDescription: (.*?)\nStatus: (.*?)\nSource Link: (.*?)\n",
        summary,
        re.DOTALL,
    )
    for m in matches:
        tenders.append(
            {
                "title": m[0].strip(),
                "issue_date": m[1].strip(),
                "closing_date": m[2].strip(),
                "location": m[3].strip(),
                "description": m[4].strip(),
                "status": m[5].strip(),
                "source_link": m[6].strip(),
            }
        )
    return tenders


def load_reference_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    province_dict, current = {}, None
    for line in lines:
        line = line.strip()
        if re.match(r"^\d+\.\s*(\w+)", line):
            current = re.sub(r"^\d+\.\s*", "", line)
            province_dict[current] = [current]
        elif line and current:
            province_dict[current].append(re.split(r"\(", line)[0].strip())
    return province_dict


def rerank_docs(query, docs):
    pairs = [
        (query, d if isinstance(d, str) else getattr(d, "page_content", ""))
        for d in docs
    ]
    torch.cuda.empty_cache()
    scores = reranker_model.predict(pairs)
    torch.cuda.empty_cache()
    return sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)


# ------------------ Retrieval ------------------
def get_relevant_content(question, score_threshold=0.7, top_k=50, time_frame=None):
    store = QdrantVectorStore(
        client=qdrant_client,
        collection_name="report_generation_content",
        embedding=embedding_model,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
    )
    results = store.similarity_search_with_score(query=question, k=top_k)
    torch.cuda.empty_cache()

    if time_frame and len(time_frame) == 2:
        from_date, to_date = parse_date_safe(time_frame[0]), parse_date_safe(
            time_frame[1]
        )
    else:
        from_date = to_date = None

    final_docs, metadata = [], []
    for doc, score in results:
        if score <= score_threshold:
            continue
        try:
            parsed = ast.literal_eval(doc.page_content.strip())
            content = parsed.get("content: ", "")
            meta = parsed.get("metadata", {})
            doc_date = parse_date_safe(meta.get("Time_frame", ""))
            if (
                from_date
                and to_date
                and (not doc_date or not (from_date <= doc_date <= to_date))
            ):
                continue
            if content:
                final_docs.append(content)
                metadata.append(meta.get("source", ""))
        except Exception:
            continue

    if not final_docs:
        return "", set()

    ranked = rerank_docs(question, final_docs)
    top_docs, top_meta = [], []
    for d, s in ranked:
        if s > score_threshold:
            idx = final_docs.index(d)
            top_docs.append(d)
            top_meta.append(metadata[idx])

    if not top_docs:
        return "", set()

    return ", ".join(set(top_docs[:10])).replace("\\", ""), set(top_meta[:10])


# ------------------ Report Generation ------------------
def generate_report(question, content, sources, time_frame, report_title_value):
    prompt = f"""
You are an AI assistant tasked with summarizing and structuring a report.

Keyword(s): {question}
Content: {content}

Based on the content, provide a comprehensive report summary.
After the summary, list the main topics as a bulleted list, key events with dates, and any tender information.
For main topics, use a format like: `* Topic 1\n* Topic 2`.
For key events, use a format like: `**Date**: Event Description`.
For tender information, use a format like: `Tender Title: [title]\nIssue Date: [date]\nClosing Date: [date]\nLocation: [location]\nDescription: [description]\nStatus: [status]\nSource Link: [link]`.

Return the summary, main topics, key events with dates, tender info if available.
"""
    response = gemini_model.generate_content(prompt)
    summary = clean_invalid_json(response.text)

    ner = extract_named_entities(summary)

    top_topics = extract_topics(summary)
    key_events = extract_key_events(summary)
    tenders = extract_tenders(summary)

    return {
        "report_title": report_title_value.capitalize(),
        "keywords": question,
        "time_frame": (
            f"{time_frame[0]} - {time_frame[1]}" if time_frame else "Not specified"
        ),
        "generated_on": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sources": list(sources),
        "summary": summary,
        "total_relevant_items": len(content.split(",")),
        "geographical_coverage": ner.get("locations", []),
        "top_topics": top_topics,
        "key_events": key_events,
        "tenders": tenders,
        "top_locations": ner.get("locations", []),
        "top_entities": ner.get("organizations", []),
    }


# --------- Fix for long unbreakable words ---------
def safe_text_for_pdf(text):
    text = re.sub(r"(\*\*[^*]+:\*\*)", r"\1 ", text)  # Add space after **Subtopic:**
    text = text.replace("***", "** *")  # Fix triple asterisk runs
    return text


# --------- Detect subtopics ---------
def parse_subtopics(text):
    parts = re.split(r"\*\*(.*?)\*\*", text)
    result = []
    current_label = None
    for i, part in enumerate(parts):
        if i % 2 == 1:
            current_label = part.strip().rstrip(":")
        else:
            if current_label:
                result.append((current_label, part.strip()))
                current_label = None
    return result


# --------- Write subtopics & handle bullets ---------
def write_subtopics(pdf, text):
    text = safe_text_for_pdf(text)
    subtopics = parse_subtopics(text)
    for label, content in subtopics:
        # Subtopic heading
        pdf.set_font("DejaVu", "B", 11)
        pdf.multi_cell(0, 6, f"{label}:")
        pdf.set_font("DejaVu", "", 11)

        # Lines under subtopic
        for para in content.split("\n"):
            para = para.strip()
            if not para:
                continue
            if para.startswith("○"):
                parts = para[1:].strip().split(":", 1)
                if len(parts) == 2:
                    pdf.sub_bullet(parts[0] + ":", parts[1].strip())
                else:
                    pdf.sub_bullet(para[1:].strip())
            elif para.startswith("●"):
                parts = para[1:].strip().split(":", 1)
                if len(parts) == 2:
                    pdf.bullet(parts[0] + ":", parts[1].strip())
                else:
                    pdf.bullet(para[1:].strip())
            elif para.startswith("* "):
                pdf.numbered_list([para[2:].strip()])
            else:
                pdf.set_x(pdf.l_margin + 8)
                pdf.multi_cell(0, 6, para)
                pdf.ln(1)


# ---------- PDF Class ----------
class PDFReport(FPDF):
    def __init__(self, data, company_logo_path=None, product_logo_path=None):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)

        self.add_font(
            "DejaVu", "", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", uni=True
        )
        self.add_font(
            "DejaVu",
            "B",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            uni=True,
        )
        self.add_font(
            "DejaVu",
            "I",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
            uni=True,
        )
        self.set_font("DejaVu", "", 11)

        self.data = data  # store the report data
        self.company_logo_path = company_logo_path
        self.product_logo_path = product_logo_path

    # Transparency support
    def set_alpha(self, alpha, blend_mode="Normal"):
        """Set transparency for watermark"""
        # Add new ExtGState for alpha
        gs_id = len(getattr(self, "_extgstates", [])) + 1
        if not hasattr(self, "_extgstates"):
            self._extgstates = []
        self._extgstates.append((alpha, blend_mode))
        self._out(f"/GS{gs_id} gs")

    def _putextgstates(self):
        """Internal: write ExtGState definitions to PDF"""
        for i, (alpha, blend_mode) in enumerate(self._extgstates, start=1):
            self._newobj()
            self._out("<< /Type /ExtGState")
            self._out(f"/ca {alpha}")  # stroke alpha
            self._out(f"/CA {alpha}")  # fill alpha
            self._out(f"/BM /{blend_mode}")
            self._out(">>")
            self._out("endobj")
            self._extgstates[i - 1] = i

    def _putresourcedict(self):
        super()._putresourcedict()
        if hasattr(self, "_extgstates") and self._extgstates:
            self._out("/ExtGState <<")
            for i in range(len(self._extgstates)):
                self._out(f"/GS{i+1} {self._extgstates[i]} 0 R")
            self._out(">>")

    def _putresources(self):
        self._putextgstates()
        super()._putresources()

    # Rotation support
    def rotate(self, angle, x=None, y=None):
        """Rotate objects (for watermark slant)"""
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        angle = radians(angle)
        c = cos(angle)
        s = sin(angle)
        cx = x * self.k
        cy = (self.h - y) * self.k
        self._out(
            f"q {c:.5f} {s:.5f} {-s:.5f} {c:.5f} "
            f"{cx - c * cx + s * cy:.5f} {cy - s * cx - c * cy:.5f} cm"
        )

    # Header with watermark + logos
    def header(self):
        # Save state before watermark
        self._out("q")

        # Product logo watermark - big, slanted, transparent
        if self.product_logo_path and os.path.exists(self.product_logo_path):
            self.set_alpha(0.05)
            self.image(
                self.product_logo_path, x=self.w / 2 - 60, y=self.h / 2 - 60, w=120
            )
            self.set_alpha(0.08)

        # Restore state so rotation doesn't affect text
        self._out("Q")

        # Company logo - small top-right
        if self.company_logo_path and os.path.exists(self.company_logo_path):
            self.image(self.company_logo_path, x=self.w - 20, y=6, w=10)

        # Report title
        if self.page_no() == 1:
            self.set_font("DejaVu", "B", 16)
            self.cell(
                0,
                10,
                self.data.get("report_title", ""),
                align="C",
                new_x="LMARGIN",
                new_y="NEXT",
            )
            # self.cell(0, 10, self.data.get("report_title", ""), align="C")
            self.ln(5)
        else:
            self.ln(15)

    # Footer with page numbers
    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "I", 9)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    # Section title
    def section_title(self, title):
        self.set_font("DejaVu", "B", 14)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        # self.cell(0, 8, title)
        self.ln(2)
        self.set_font("DejaVu", "", 11)

    # Bullet point
    def bullet(self, label, value="", indent=0, bold_label=True):
        self.set_x(self.l_margin + indent)
        if bold_label:
            self.set_font("DejaVu", "B", 11)
            self.write(6, f"● {label}")
            self.set_font("DejaVu", "", 11)
            if value:
                self.write(6, f" {value}")
        else:
            self.set_font("DejaVu", "", 11)
            self.write(6, f"● {label} {value}")
        self.ln(6)

    # Sub bullet
    def sub_bullet(self, label, value="", indent=10, bold_label=True):
        self.set_x(self.l_margin + indent)
        if bold_label:
            self.set_font("DejaVu", "B", 11)
            self.write(6, f"○ {label}")
            self.set_font("DejaVu", "", 11)
            if value:
                self.write(6, f" {value}")
        else:
            self.set_font("DejaVu", "", 11)
            self.write(6, f"○ {label} {value}")
        self.ln(6)

    # Numbered list
    def numbered_list(self, items, indent=10):
        self.set_font("DejaVu", "", 11)
        for i, item in enumerate(items, start=1):
            self.set_x(self.l_margin + indent)
            self.multi_cell(0, 6, f"{i}. {item}")
            self.ln(1)


USER_REPORTS_DIR = "user_generated_reports"


# ---------- PDF Save Function ----------
def save_report_pdf(data, filename="generated_report.pdf"):

    # Ensure reports directory exists
    os.makedirs(USER_REPORTS_DIR, exist_ok=True)

    # Construct full file path
    filepath = os.path.join(USER_REPORTS_DIR, filename)

    company_logo = os.path.abspath(
        "static/imgs/report_gen_images/pinaca_logo.jpeg"
    )  # Path to small company logo (top-right)
    product_logo = os.path.abspath(
        "static/imgs/report_gen_images/inner_eye_logo.jpeg"
    )  # Path to product logo (watermark)

    if not os.path.exists(product_logo):
        raise FileNotFoundError(f"Product logo not found: {product_logo}")

    pdf = PDFReport(
        data, company_logo_path=company_logo, product_logo_path=product_logo
    )

    pdf.add_page()
    i = 0

    # Remove Gemini placeholder lines
    summary_text = data.get("summary", "")
    summary_text = re.sub(
        r"(Key Events with Dates:\s*No.*?(\n|$))", "", summary_text, flags=re.IGNORECASE
    )
    summary_text = re.sub(
        r"(Tender Information:\s*No.*?(\n|$))", "", summary_text, flags=re.IGNORECASE
    )

    # 1. Introduction
    i += 1
    pdf.section_title(f"{i}. Introduction")
    pdf.bullet("Keyword(s):", data.get("keywords", ""))
    pdf.bullet("Time Frame:", data.get("time_frame", ""))
    pdf.bullet("Report Generated On:", data.get("generated_on", ""))
    pdf.bullet("Data Sources:", ", ".join(data.get("sources", [])))

    # 2. Summary Overview
    i += 1
    pdf.section_title(f"{i}. Summary Overview")
    write_subtopics(pdf, summary_text)
    pdf.bullet("Total Relevant Items Found:", str(data.get("total_relevant_items", 0)))
    pdf.bullet(
        "Geographical Coverage:", ", ".join(data.get("geographical_coverage", []))
    )
    if data.get("top_topics"):
        pdf.bullet("Top Topics Identified:")
        pdf.numbered_list(data["top_topics"])

    # 3. Key Events and Updates
    if data.get("key_events"):
        i += 1
        pdf.section_title(f"{i}. Key Events and Updates (Chronological)")
        for e in data["key_events"]:
            pdf.sub_bullet("Date:", e.get("date", ""))
            pdf.sub_bullet("Event:", e.get("event", ""))

    # 4. Tender Information
    if data.get("tenders"):
        i += 1
        pdf.section_title(f"{i}. Tender Information (if applicable)")
        for t in data["tenders"]:
            pdf.sub_bullet("Title:", t.get("title", ""))
            pdf.sub_bullet("Issue Date:", t.get("issue_date", ""))
            pdf.sub_bullet("Closing Date:", t.get("closing_date", ""))
            pdf.sub_bullet("Location:", t.get("location", ""))
            pdf.sub_bullet("Description:", t.get("description", ""))
            pdf.sub_bullet("Status:", t.get("status", ""))
            pdf.sub_bullet("Source Link:", t.get("source_link", ""))
            pdf.ln(1)

    # 5. Data Insights
    if data.get("top_locations") or data.get("top_entities"):
        i += 1
        pdf.section_title(f"{i}. Data Insights")
        if data.get("top_locations"):
            pdf.bullet("Top Mentioned Locations:")
            pdf.numbered_list(list(set(data["top_locations"])))
        if data.get("top_entities"):
            pdf.bullet("Top Mentioned Entities (Companies, Agencies, etc.):")
            pdf.numbered_list(list(set(data["top_entities"])))

    pdf.output(filepath)
    return f"{filename}"


## report generation library & function code ends here


# private & public key loading for encription in JWT starts here
# =========================================================

with open("private_key_token.pem", "rb") as f:
    private_token_key = serialization.load_pem_private_key(
        f.read(), password=None, backend=default_backend()
    )

# Load the public key
with open("public_key_token.pem", "rb") as f:
    public_token_key = serialization.load_pem_public_key(
        f.read(), backend=default_backend()
    )

# private & public key loading for encription in JWT ends here
# =========================================================


# Initialize JWT Manager
app.config["JWT_SECRET_KEY"] = secrets.token_hex(40)
app.config["JWT_PRIVATE_KEY"] = private_token_key
app.config["JWT_PUBLIC_KEY"] = public_token_key
app.config["JWT_ALGORITHM"] = "RS256"
app.config["JWT_TOKEN_LOCATION"] = ["cookies"]
app.config["JWT_COOKIE_SECURE"] = True
app.config["JWT_COOKIE_HTTPONLY"] = True
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = datetime.timedelta(minutes=3)
app.config["JWT_REFRESH_TOKEN_EXPIRES"] = datetime.timedelta(days=1)
jwt = JWTManager(app)


db = client["ie_db"]
users = db["users"]
feeds_data = db["feeds_data"]
mucd_data = db["mucd_data"]
china_map_geo_json_data = db["china_geo_json_map_data"]
users_data = db["users_data"]
special_report_data = db["special_report_data"]
generate_report_data = db["generate_report_data"]
user_report_data = db["user_report_data"]
tenders_data = db["tenders_data"]
trending_keywords_data = db["trending_keywords_data"]
trending_keywords_pie_data = db["trending_keywords_pie_data"]
vehicles_data = db["vehicles_data"]
weapons_data = db["weapons_data"]
commanders_data = db["commanders_profile_data"]
social_media_data = db["social_media_data"]
users_auth_history = db["users_auth_history"]
satcom_data = db["satcom_data"]
indian_bases_data = db["indian_bases_data"]
satellite_tracking_data = db["satellite_tracking_data"]


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


# image directory of no image available image file
no_image_dir = "no_image"


@app.route("/nomg")
def no_image_available():
    if "email" not in session:
        return redirect(url_for("authentication"))

    filename = "no-image-found.jpg"
    return send_from_directory(no_image_dir, filename)


@app.route("/zhflgmg")
def china_flag_image():
    if "email" not in session:
        return redirect(url_for("authentication"))

    china_flag_image_name = "zh-flg.jpg"
    return send_from_directory(no_image_dir, china_flag_image_name)


# Set the directory where your images are stored
image_dir = "server_images"


@app.route("/res/<image_name>")
def serve_image(image_name):
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Validate the image name to prevent directory traversal
    if not re.match(
        r"^[\w\-. ]+$", image_name
    ):  # Allow only alphanumeric, dash, underscore, dot, and space
        return redirect(url_for("china_flag_image"))

    # Serve the image from the directory
    try:
        return send_from_directory(image_dir, image_name)
    except FileNotFoundError:
        return redirect(url_for("china_flag_image"))


# Set the directory where your reports are stored
special_report_dir = "special_reports"


# view special report
@app.route("/spcrpt/<report_number>")
def special_report_view(report_number):
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Validate the report_number to only allow valid characters (e.g., alphanumeric, hyphen, underscore)
    if not re.match(r"^[a-zA-Z0-9_-]+$", report_number):
        return jsonify({"error": "Invalid report number"}), 400

    filename = report_number + ".pdf"

    return send_from_directory(special_report_dir, filename)


# download special report
@app.route("/spcrptdwn/<report_number>", methods=["GET"])
def special_report_download(report_number):
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Validate the report_number to only allow valid characters (e.g., alphanumeric, hyphen, underscore)
    if not re.match(r"^[a-zA-Z0-9_-]+$", report_number):
        return jsonify({"error": "Invalid report number"}), 400

    filename = report_number + ".pdf"

    if report_number != "":
        report_file_name = filename
        return send_from_directory(
            special_report_dir, report_file_name, as_attachment=True
        )


# Set the directory where your reports are stored
generate_report_dir = "generate_reports"


# view generate report
@app.route("/gnrpt/<report_number>")
def generate_report_view(report_number):
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Validate the report_number to only allow valid characters (e.g., alphanumeric, hyphen, underscore)
    if not re.match(r"^[a-zA-Z0-9_-]+$", report_number):
        return jsonify({"error": "Invalid report number"}), 400

    filename = report_number + ".pdf"

    return send_from_directory(generate_report_dir, filename)


# download generate report
@app.route("/gnrptdwn/<report_number>", methods=["GET"])
def generate_report_download(report_number):
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Validate the report_number to only allow valid characters (e.g., alphanumeric, hyphen, underscore)
    if not re.match(r"^[a-zA-Z0-9_-]+$", report_number):
        return jsonify({"error": "Invalid report number"}), 400

    filename = report_number + ".pdf"

    if report_number != "":
        report_file_name = filename
        return send_from_directory(
            generate_report_dir, report_file_name, as_attachment=True
        )


# Set the directory where your reports are stored
user_generated_report_dir = "user_generated_reports"


# view generate report
@app.route("/usrgnrpt/<report_number>")
def user_generate_report_view(report_number):
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Validate the report_number to only allow valid characters (e.g., alphanumeric, hyphen, underscore)
    if not re.match(r"^[a-zA-Z0-9_-]+$", report_number):
        return jsonify({"error": "Invalid report number"}), 400

    filename = report_number + ".pdf"

    return send_from_directory(user_generated_report_dir, filename)


# download generate report
@app.route("/usrgnrptdwn/<report_number>", methods=["GET"])
def user_generate_report_download(report_number):
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Validate the report_number to only allow valid characters (e.g., alphanumeric, hyphen, underscore)
    if not re.match(r"^[a-zA-Z0-9_-]+$", report_number):
        return jsonify({"error": "Invalid report number"}), 400

    filename = report_number + ".pdf"

    if report_number != "":
        report_file_name = filename
        return send_from_directory(
            user_generated_report_dir, report_file_name, as_attachment=True
        )


# Set the directory where your tenders are stored
tenders_dir = "tenders"


# view tender
@app.route("/tndrs/<tender_number>")
def tender_view(tender_number):
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Validate the tender_number to only allow valid characters (e.g., alphanumeric, hyphen, underscore)
    if not re.match(r"^[a-zA-Z0-9_-]+$", tender_number):
        return jsonify({"error": "Invalid tender_number"}), 400

    tender_hash = tender_number
    if tender_hash:
        checking_tender_data_in_db = tenders_data.find_one(
            {"tender_hash": tender_hash}, {"_id": 0}
        )
        if checking_tender_data_in_db:
            tender_title = checking_tender_data_in_db["tender_title"]
            file_name = tender_title + ".pdf"
            if file_name:
                return send_from_directory(tenders_dir, file_name), 200
            else:
                return jsonify({"error": "File not found"}), 400
        else:
            # If tender_number not found, return a 400 error
            return jsonify({"error": "Invalid tender"}), 400
    else:
        # If tender_number is empty, return a 400 error
        return jsonify({"error": "Invalid input"}), 400


# tender new tab viewer
@app.route("/tndrsnwtbvw/<tender_number>", methods=["GET"])
def tender_new_tab_viewer(tender_number):
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Validate the tender_number to only allow valid characters (e.g., alphanumeric, hyphen, underscore)
    if not re.match(r"^[a-zA-Z0-9_-]+$", tender_number):
        return jsonify({"error": "Invalid tender_number"}), 400

    tender_hash = tender_number
    if tender_hash:
        checking_tender_data_in_db = tenders_data.find_one(
            {"tender_hash": tender_hash}, {"_id": 0}
        )
        if checking_tender_data_in_db:
            tender_title = checking_tender_data_in_db["tender_title"]
            file_name = tender_title + ".pdf"
            if file_name:
                return (
                    send_from_directory(tenders_dir, file_name, as_attachment=False),
                    200,
                )
            else:
                return jsonify({"error": "File not found"}), 400
        else:
            # If tender_number not found, return a 400 error
            return jsonify({"error": "Invalid tender"}), 400
    else:
        # If tender_number is empty, return a 400 error
        return jsonify({"error": "Invalid input"}), 400


# Set the directory where your reports are stored
mucd_doc_dir = "mucd_documents"


@app.route("/mddcdwn/<report_number>", methods=["POST"])
def mucd_doc_download(report_number):
    if "email" not in session:
        return redirect(url_for("authentication"))

    if report_number:
        # Check for both .doc and .docx extensions
        possible_extensions = [".doc", ".docx"]
        found_file = None

        for ext in possible_extensions:
            document_file_name = f"{report_number}{ext}"
            file_path = os.path.join(mucd_doc_dir, document_file_name)
            if os.path.isfile(file_path):
                found_file = document_file_name
                break

        if found_file:
            return send_from_directory(mucd_doc_dir, found_file, as_attachment=True)
        else:
            # If no file is found, return a 404 error
            abort(404)

    # If report_number is empty, return a 400 error
    return abort(400)


# functions to create image link
def gen_img_link(dictionary, image_dir=image_dir):
    # Check if 'image' key exists in the dictionary
    if "image" not in dictionary:
        return "NA"

    # Initialize image_name and image_path
    image_name = None
    image_path = None  # Initialize image_path

    # If 'image' is a string, use it directly
    if isinstance(dictionary["image"], str):
        image_name = dictionary["image"]
        image_path = os.path.join("res", image_name)
        if os.path.exists(os.path.join(image_dir, image_name)):

            # If the image exists, construct the complete image link
            domain_name = request.url_root
            complete_image_link = "{}{}".format(domain_name, image_path)
            return complete_image_link
        else:
            return "NA"  # Return "NA" if the image does not exist

    # If 'image' is a list, randomly select an image
    elif isinstance(dictionary["image"], list):
        for img in dictionary["image"]:
            image_name = img
            image_path = os.path.join("res", image_name)

            # Check if the image exists in the image directory
            if os.path.exists(os.path.join(image_dir, image_name)):
                break  # Exit the loop if a valid image is found
        else:
            # If no valid image was found, return "NA"
            return "NA"
    else:
        return "NA"  # Return "NA" if 'image' is neither a string nor a list

    # If a valid image name is found, construct the complete image link
    domain_name = request.url_root
    complete_image_link = "{}{}".format(domain_name, image_path)
    return complete_image_link


# function to split chinese paragraph on the bases of chinese dot '。'
def split_chinese_paragraphs(text):
    text = text.replace("\xa0", " ")
    paragraphs = re.split("。", text)
    paragraphs = [paragraph.strip() for paragraph in paragraphs]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    return paragraphs


# function to split english paragraph on the bases of dot '.'
def split_english_paragraphs(text):
    paragraphs = text.split(".")
    return paragraphs


# function to check dictionary with blank values and replace with "not available" string in those values
def replace_blank_and_NA_values(d):
    for key, value in d.items():
        if not value:
            d[key] = "NA"
        elif "NA" in value:
            d[key] = "NA"
    return d


# function to fixing theatre command spelling in mucd tracking
def fixing_theatre_command_spelling(data_dict):
    theatre_command = data_dict["theatre_command"]

    if theatre_command == "NA":
        return data_dict

    if "theatre" in theatre_command:
        data_dict["theatre_command"] = theatre_command
    else:
        splitting_theatre_command = data_dict["theatre_command"].split(" ")
        fixed_spelling = (
            splitting_theatre_command[0] + " theatre " + splitting_theatre_command[1]
        )
        data_dict["theatre_command"] = fixed_spelling

    return data_dict


# function to check NA and fix the feeds
def fixing_feeds(list_of_feeds):

    # checking if title and content are there in article
    feed_list_with_valid_tiitle_and_content = []
    for feed in list_of_feeds:

        if "title" in feed and "content" in feed:
            feed_list_with_valid_tiitle_and_content.append(feed)

    # checking how many feed have NA in title
    feed_with_valid_title = []
    for feed_title in feed_list_with_valid_tiitle_and_content:
        title_value = feed_title["title"]
        if title_value != "NA":
            feed_with_valid_title.append(feed_title)

    # splitting content in three paragraph if content is available
    feed_content_with_three_paragraph = []
    for fd_cnt in feed_with_valid_title:

        if isinstance(fd_cnt["content"], str):
            # splitting content in three paragraph only if it is a string
            if fd_cnt["content"] != "NA":
                splitting_feed_content = split_english_paragraphs(fd_cnt["content"])[:3]
                creating_string_for_content = "".join(splitting_feed_content).strip()
                fd_cnt["content"] = creating_string_for_content
                feed_content_with_three_paragraph.append(fd_cnt)
        elif isinstance(fd_cnt["content"], list):
            # merging content with three paragraph only if it is a list
            splitting_feed_content = fd_cnt["content"][:3]
            creating_string_for_content = " ".join(splitting_feed_content).strip()
            fd_cnt["content"] = creating_string_for_content
            feed_content_with_three_paragraph.append(fd_cnt)

    # splitting date from date time if it is available in feed
    feed_with_date_splitted = []
    for fd in feed_content_with_three_paragraph:
        if fd["date"] != "NA":
            fd["date"] = fd["date"].split(" ", 1)[0]
        else:
            fd["date"] = "NA"
        feed_with_date_splitted.append(fd)

    return feed_with_date_splitted


# function to validating & fixing commander name
def fix_commander_name(commander_name_value):
    final_dict = {}

    # Converting into string
    converted_in_string = str(commander_name_value)

    # Removing leading & trailing spaces from the name
    name_without_spaces = converted_in_string.strip()

    # checking allowed length
    if len(name_without_spaces) < 2 or len(name_without_spaces) > 30:
        return {"isvalid": "false", "name_value": name_without_spaces}

    # Regex pattern to allow valid characters (letters, digits, spaces, and certain special characters)
    valid_pattern = r"^[\w\s\.\-\'，。]+$"  # This pattern allows letters, digits, spaces, and some special characters

    # Regex pattern to detect potential malicious code patterns
    malicious_pattern = r"(\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bDROP\b|\b;|\'|\"|--|#|\bEXEC\b|\bUNION\b|\bOR\b|\bAND\b|<|>|%|\\|/|\\\*|\\\*.*?\\\*|;|&|\||\bscript\b|\balert\b|\bconsole\b|\bexec\b|\bimport\b|\bsubprocess\b|\bopen\b)"

    # Check for valid characters
    if re.match(valid_pattern, name_without_spaces) and not re.search(
        malicious_pattern, name_without_spaces, re.IGNORECASE
    ):
        sanitized_name = bleach.clean(name_without_spaces)
        final_data = {"isvalid": "true", "name_value": sanitized_name}
    else:
        final_data = {"isvalid": "false", "name_value": name_without_spaces}

    return final_data


# function to check commander profile id is valid or not before searching in db
def check_alphanumeric_with_hyphen(value):
    final_dict = {}

    # Converting into string
    converted_in_string = str(value)

    # Removing leading & trailing spaces from the name
    commander_profile_id_without_spaces = converted_in_string.strip()

    # Regex pattern to allow only alphanumeric characters and hyphens
    valid_pattern = (
        r"^[a-zA-Z0-9\-]+$"  # This pattern allows letters, digits, and hyphens
    )

    # Regex pattern to detect potential malicious code patterns
    malicious_pattern = r"(\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bDROP\b|\b;|\'|\"|--|#|\bEXEC\b|\bUNION\b|\bOR\b|\bAND\b|<|>|%|\\|/|\\\*|\\\*.*?\\\*|;|&|\||\bscript\b|\balert\b|\bconsole\b|\bexec\b|\bimport\b|\bsubprocess\b|\bopen\b)"

    # Check for valid characters
    if re.match(valid_pattern, commander_profile_id_without_spaces) and not re.search(
        malicious_pattern, commander_profile_id_without_spaces, re.IGNORECASE
    ):
        sanitized_string = bleach.clean(commander_profile_id_without_spaces)
        final_data = {"isvalid": "true", "commander_profile_id_value": sanitized_string}
    else:
        final_data = {
            "isvalid": "false",
            "commander_profile_id_value": commander_profile_id_without_spaces,
        }

    return final_data


# Sort all the articles of feeds in descending order using date
def parse_date(date_str):
    # Check for 'NA' and strip whitespace
    date_str = date_str.strip()
    if date_str == "NA":
        return datetime.datetime.min  # Use a very early date for sorting
    try:
        # Attempt to parse the date
        return datetime.datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        # If parsing fails, return a very early date
        return datetime.datetime.min


# function to fix all keys of commander profile dict into lowercase
def lowercase_keys(input_dict):
    lowercase_dict = {}

    # Create a new dictionary with lowercase keys
    for key, value in input_dict.items():
        lowercase_dict[key.lower()] = value

    return lowercase_dict


def validating_string_and_sanitizing_string(string_value):
    # Step 1: Strip leading and trailing whitespace
    sanitized_string = string_value.strip()

    # Step 2: Check if sanitized_string is a string and meets length requirements
    if (
        not isinstance(sanitized_string, str)
        or len(sanitized_string) < 2
        or len(sanitized_string) > 30
    ):
        return {"string_validated": "false", "string_value": sanitized_string}

    # Step 3: Allowed characters in string value
    if not re.match("^[a-zA-Z0-9_ ]*$", sanitized_string):
        return {"string_validated": "false", "string_value": sanitized_string}

    # Step 4: Sanitize the string to prevent XSS (Cross-Site Scripting)
    sanitized_string = html.escape(sanitized_string)

    # Step 5: Remove any non-alphanumeric characters except spaces and underscores
    sanitized_string = re.sub(r"[^a-zA-Z0-9_ ]", "", sanitized_string)

    # Step 6: Preventing multiple XSS attacks
    sanitized_string = bleach.clean(sanitized_string)

    return {"string_validated": "true", "string_value": sanitized_string}


# def validating_string_and_sanitizing_string_on_search_bar(string_value):
#     # Step 1: Strip leading and trailing whitespace
#     sanitized_string = string_value.strip()

#     # Step 2: Check if sanitized_string is a string and meets length requirements
#     if not isinstance(sanitized_string, str) or len(sanitized_string) < 2 or len(sanitized_string) > 70:
#         return {"string_validated": "false", "string_value": sanitized_string}

#     # Step 3: Allowed characters in string value
#     if not re.match("^[a-zA-Z0-9_ ]*$", sanitized_string):
#         return {"string_validated":"false", "string_value":sanitized_string}

#     # Step 4: Sanitize the string to prevent XSS (Cross-Site Scripting)
#     sanitized_string = html.escape(sanitized_string)

#     # Step 5: Remove any non-alphanumeric characters except spaces and underscores
#     sanitized_string = re.sub(r'[^a-zA-Z0-9_ ]', '', sanitized_string)

#     # Step 6: Preventing multiple XSS attacks
#     sanitized_string = bleach.clean(sanitized_string)

#     return {"string_validated":"true", "string_value":sanitized_string}


def validating_string_and_sanitizing_string_on_search_bar(string_value):
    # Step 1: Strip leading and trailing whitespace
    sanitized_string = string_value.strip()

    # Step 2: Check if sanitized_string is a string and meets length requirements
    if (
        not isinstance(sanitized_string, str)
        or len(sanitized_string) < 2
        or len(sanitized_string) > 150
    ):
        return {"string_validated": "false", "string_value": sanitized_string}

    # Step 3: Allowed characters in string value (including dot and hyphen)
    if not re.match("^[a-zA-Z0-9'-_,(). ]*$", sanitized_string):
        return {"string_validated": "false", "string_value": sanitized_string}

    # Step 4: Sanitize the string to prevent XSS (Cross-Site Scripting)
    sanitized_string = html.escape(sanitized_string)

    # Step 5: Remove any non-alphanumeric characters except spaces, underscores, dots, and hyphens
    sanitized_string = re.sub(r"[^a-zA-Z0-9\'-_,(). ]", "", sanitized_string)

    # Step 6: Preventing multiple XSS attacks
    sanitized_string = bleach.clean(sanitized_string)

    return {"string_validated": "true", "string_value": sanitized_string}


def validating_comment_and_sanitizing_comment(comment_value):
    # Step 1: Strip leading and trailing whitespace
    sanitized_comment = comment_value.strip()

    # Step 2: Check if sanitized_comment is a string and meets length requirements
    if (
        not isinstance(sanitized_comment, str)
        or len(sanitized_comment) < 2
        or len(sanitized_comment) > 100
    ):
        return {"comment_validation": "false", "comment_value": sanitized_comment}

    # Step 3: Allowed characters in string value
    if not re.match("^[a-zA-Z0-9_ ]*$", sanitized_comment):
        return {"comment_validation": "false", "comment_value": sanitized_comment}

    # Step 4: Sanitize the string to prevent XSS (Cross-Site Scripting)
    sanitized_comment = html.escape(sanitized_comment)

    # Step 5: Remove any non-alphanumeric characters except spaces and underscores
    sanitized_comment = re.sub(r"[^a-zA-Z0-9_ ]", "", sanitized_comment)

    # Step 6: Preventing multiple XSS attacks
    sanitized_comment = bleach.clean(sanitized_comment)

    return {"comment_validation": "true", "comment_value": sanitized_comment}


@app.route("/", methods=["GET"])
def index():

    # Get the real IP address for vps
    real_ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    # Call the IPinfo API to get location data
    response = requests.get(f"https://ipinfo.io/{real_ip}/json?token={IPINFO_TOKEN}")

    if response.status_code == 200:
        data = response.json()
        country = data.get("country")

        # Redirect based on the country
        if country == "IN":
            return render_template("index.html")
        else:
            return "", 404
    return "", 404


@app.route("/auth", methods=["GET"])
def authentication():
    if "email" in session:
        return redirect(url_for("home"))

    # Get the real IP address for vps
    real_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    # real_ip = "183.83.155.83"

    # Call the IPinfo API to get location data
    response = requests.get(f"https://ipinfo.io/{real_ip}/json?token={IPINFO_TOKEN}")

    if response.status_code == 200:
        data = response.json()
        country = data.get("country")

        # Redirect based on the country
        if country == "IN":
            return render_template("login.html")
        else:
            return "", 404
    return "", 404


@app.route("/rfactk", methods=["POST"])
@jwt_required(refresh=True)
def jwttokenrefresh():
    identity = get_jwt_identity()

    # Fetch the user from the database
    user = users.find_one({"email": identity})

    # Check if the refresh token is valid
    if user["refresh_token_status"] == "invalidated":
        return jsonify({"msg": "Refresh token is invalid"}), 401

    # Invalidate the old refresh token
    users.update_one(
        {"email": identity}, {"$set": {"refresh_token_status": "invalidated"}}
    )

    # Create new tokens
    access_token = create_access_token(identity=identity)
    refresh_token = create_refresh_token(identity=identity)

    # Set the new refresh token in an HTTP-only cookie
    response = jsonify({"tkRF": True})

    # Set expiration time for cookies
    max_age = datetime.timedelta(days=1)  # Set to your desired duration
    set_access_cookies(response, access_token, max_age=max_age)
    set_refresh_cookies(response, refresh_token, max_age=max_age)

    # Update the user document with the new refresh token and set its status to active
    users.update_one(
        {"email": identity},
        {"$set": {"refresh_token": refresh_token, "refresh_token_status": "active"}},
    )
    return response


@app.route("/login", methods=["POST"])
def login():
    email = request.json["email"]
    password = request.json["password"]

    # Get the real IP address for vps
    real_ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    # Call the IPinfo API to get location data
    response = requests.get(f"https://ipinfo.io/{real_ip}/json?token={IPINFO_TOKEN}")

    if response.status_code == 200:
        data = response.json()
        country = data.get("country")

        # Redirect based on the country
        if country == "IN":
            # regex pattern to validate email
            email_regex = r"^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})*(?:\.[a-zA-Z]{2,})*$"

            # Validate the email using the regex pattern
            if re.match(email_regex, email):
                user = users.find_one({"email": email})

                if user and bcrypt.checkpw(password.encode("utf-8"), user["password"]):
                    # Format the current date and time
                    formatted_time = datetime.datetime.now().strftime(
                        "%Y-%m-%d %I:%M %p"
                    )

                    # Get the real IP address for vps
                    real_ip = request.headers.get(
                        "X-Forwarded-For", request.remote_addr
                    )

                    # store all login and logout entries for seperate user
                    user_event_document = {
                        "type": "login",
                        "login_time": formatted_time,
                        "ip_address": real_ip,
                        "user_agent": request.user_agent.string,
                    }

                    # Update the user_event_document by pushing the new event into the events array at the beginning
                    users_auth_history.update_one(
                        {
                            "email": email,
                        },
                        {
                            "$setOnInsert": {
                                "email": email,
                            },
                            "$push": {
                                "events": {
                                    "$each": [user_event_document],
                                    "$position": 0,  # Insert at the beginning of the array
                                }
                            },
                        },
                        upsert=True,
                    )

                    access_token = create_access_token(identity=email)
                    refresh_token = create_refresh_token(identity=email)

                    session.permanent = True
                    session["email"] = email
                    session["login_time"] = datetime.datetime.now()

                    # Set the refresh token & access token in an HTTP-only cookie
                    response = jsonify({"success": True})

                    # Set expiration time for cookies
                    max_age = datetime.timedelta(days=1)  # Set to your desired duration
                    set_access_cookies(response, access_token, max_age=max_age)
                    set_refresh_cookies(response, refresh_token, max_age=max_age)

                    # Update the user document with the refresh token
                    users.update_one(
                        {"email": email},
                        {
                            "$set": {
                                "refresh_token": refresh_token,
                                "refresh_token_status": "active",
                            }
                        },
                    )
                    return response, 200
                else:
                    return (
                        jsonify({"success": False, "error": "Invalid credentials"}),
                        401,
                    )
            else:
                return "Invalid email", 401
        else:
            return "", 404
    else:
        return "", 404


@app.route("/logout", methods=["POST"])
@jwt_required()
def logout():
    current_user = get_jwt_identity()
    email = current_user

    # Format the current date and time
    formatted_time = datetime.datetime.now().strftime("%Y-%m-%d %I:%M %p")

    # Get the real IP address for vps
    real_ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    # store all login and logout entries for seperate user
    user_event_document = {
        "type": "logout",
        "logout_time": formatted_time,
        "ip_address": real_ip,
        "user_agent": request.user_agent.string,
    }

    # Update the user_event_document by pushing the new event into the events array at the beginning
    users_auth_history.update_one(
        {
            "email": email,
        },
        {
            "$setOnInsert": {
                "email": email,
            },
            "$push": {
                "events": {
                    "$each": [user_event_document],
                    "$position": 0,  # Insert at the beginning of the array
                }
            },
        },
        upsert=True,
    )

    users.update_one(
        {"email": current_user}, {"$set": {"refresh_token_status": "invalidated"}}
    )
    session.pop("email", None)

    # Create a response object
    response = make_response(jsonify({"msg": "Successfully logged out"}), 200)

    # Clear all cookies by setting them to expire
    cookies = request.cookies
    for cookie in cookies:
        response.set_cookie(cookie, "", expires=0)

    return response


@app.before_request
def check_session_and_assign_full_name_in_user_panel():
    if "email" in session:
        email = session.get("email")
        user_details = users.find_one({"email": email})
        g.full_name = user_details["full_name"]

        # Check if the session has expired
        login_time = session.get("login_time")
        if login_time:
            if (
                datetime.datetime.now() - login_time
                > app.config["PERMANENT_SESSION_LIFETIME"]
            ):

                # Session has expired, log the user out
                session.pop("email", None)
                session.pop("login_time", None)
                return redirect(url_for("authentication"))
    else:
        g.full_name = None


def creating_feeds_data_for_search_result(query):
    results = []
    query_vector = query

    # searching keyword in feeds data
    search_result = qdrant_client.search(
        collection_name="qdrant_feeds_data",
        query_vector=query_vector,
        limit=50,
        with_payload=True,
        score_threshold=0.8,
    )

    feeds_results = []
    unique_ids = set()  # Set to track unique article numbers
    for result in search_result:
        payload = result.payload
        article_data = payload

        # Check if the article number is already in the set
        if article_data["article_number"] not in unique_ids:
            unique_ids.add(article_data["article_number"])  # Add to the set
            feeds_results.append(article_data)
    fixed_list_of_feeds = fixing_feeds(feeds_results)

    # Sort the list based on the 'date' key
    sorted_data = sorted(
        fixed_list_of_feeds, key=lambda x: parse_date(x["date"]), reverse=True
    )

    # adding new key in feeds
    for feed in sorted_data:
        feed["found_in"] = "feeds_col"
        feed["article_link"] = "/fdarticle/{}".format(feed["article_number"])
        results.append(feed)

    return results


def creating_social_media_data_for_search_result(query):
    results = []
    query_vector = query

    # searching keyword in social media feeds data
    search_result = qdrant_client.search(
        collection_name="qdrant_social_media_data",
        query_vector=query_vector,
        limit=50,
        with_payload=True,
        score_threshold=0.8,
    )

    social_media_results = []
    unique_ids = set()  # Set to track unique article numbers
    for result in search_result:
        payload = result.payload
        # article_data = payload['metadata']
        article_data = payload

        # Check if the article number is already in the set
        if article_data["article_number"] not in unique_ids:
            unique_ids.add(article_data["article_number"])  # Add to the set
            social_media_results.append(article_data)
    fixed_list_of_social_feeds = fixing_feeds(social_media_results)

    # Sort the list based on the 'date' key
    sorted_data = sorted(
        fixed_list_of_social_feeds, key=lambda x: parse_date(x["date"]), reverse=True
    )

    # adding new key in social media feeds
    for social_feed in sorted_data:
        social_feed["found_in"] = "social_feed_col"
        splitting_social_feed_content = split_english_paragraphs(
            social_feed["content"]
        )[:3]
        creating_string_for_content = "".join(splitting_social_feed_content).strip()
        social_feed["content"] = creating_string_for_content
        social_feed["article_link"] = "/smhart/{}".format(social_feed["article_number"])
        results.append(social_feed)

    return results


def creating_tenders_data_for_search_result(query):
    results = []
    query_vector = query

    # searching keyword in social media feeds data
    search_result = qdrant_client.search(
        collection_name="qdrant_tenders_data",
        query_vector=query_vector,
        limit=50,
        with_payload=True,
        score_threshold=0.8,
    )

    tender_results = []
    unique_ids = set()  # Set to track unique article numbers
    for result in search_result:
        payload = result.payload
        # tender_data = payload['metadata']
        tender_data = payload

        # Check if the article number is already in the set
        if tender_data["tender_hash"] not in unique_ids:
            unique_ids.add(tender_data["tender_hash"])  # Add to the set
            tender_results.append(tender_data)

    # Sort the list based on the 'date' key
    sorted_data = sorted(
        tender_results, key=lambda x: parse_date(x["tender_date"]), reverse=True
    )

    # adding new key in tender
    for tender in sorted_data:
        tender["found_in"] = "tenders_col"
        tender["tender_view_link"] = "/tndrs/{}".format(tender["tender_hash"])
        results.append(tender)

    return results


def creating_commanders_profile_data_for_search_result(query):
    results = []
    query_vector = query

    # searching keyword in social media feeds data
    search_result = qdrant_client.search(
        collection_name="qdrant_commanders_profile_data",
        query_vector=query_vector,
        limit=20,
        with_payload=True,
        score_threshold=0.8,
    )

    commanders_results = []
    unique_ids = set()  # Set to track unique article numbers
    for result in search_result:
        payload = result.payload
        # commander_profile_data = payload['metadata']
        commander_profile_data = payload

        # Check if the article number is already in the set
        if commander_profile_data["person_uid"] not in unique_ids:
            unique_ids.add(commander_profile_data["person_uid"])  # Add to the set
            commanders_results.append(commander_profile_data)

    # adding new key in commander
    for commander in commanders_results:
        commander["found_in"] = "commanders_col"
        creating_image_link = creating_image_link_for_commander(commander)
        commander["image_link"] = creating_image_link
        results.append(commander)

    return results


@app.route("/search_results", methods=["GET"])
def search_result_on_new_tab():
    if "email" not in session:
        return redirect(url_for("authentication"))
    return render_template("search-new.html")


@app.route("/srhqry", methods=["POST"])
@jwt_required()
def search_query():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        data = request.get_json()

        # Validate that 'data' is a dictionary and contains the key 'search_query'
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid input"}), 400

        if "search_query" not in data:
            return jsonify({"error": "Missing required key"}), 400

        # If validation passes, you can proceed with your logic
        search_query = data["search_query"]

        # validatin & sanitizing the received data
        sanitized_search_query = validating_string_and_sanitizing_string_on_search_bar(
            search_query
        )

        if sanitized_search_query["string_validated"] == "true":
            search_keywords = sanitized_search_query["string_value"]

            # Convert the query to an embedding
            query_vector = embeddings.embed_query(search_keywords)

            user_email = session["email"]
            check_user_in_users_data = users_data.find_one({"email": user_email})

            if check_user_in_users_data:
                results = []

                # searching keyword in feeds data
                feeds_search_results = creating_feeds_data_for_search_result(
                    query_vector
                )
                results.extend(feeds_search_results)

                # searching keyword in social media feeds
                social_media_feeds_search_results = (
                    creating_social_media_data_for_search_result(query_vector)
                )
                results.extend(social_media_feeds_search_results)

                # searching keyword in tenders
                tenders_search_results = creating_tenders_data_for_search_result(
                    query_vector
                )
                results.extend(tenders_search_results)

                # searching keyword in commander profiles
                commander_profile_search_results = (
                    creating_commanders_profile_data_for_search_result(query_vector)
                )
                results.extend(commander_profile_search_results)

                return jsonify(results)
        else:
            return jsonify({"error": "Invalid input"}), 400
    else:
        return redirect(url_for("authentication"))


# fetching important feeds from feeds_data
def fetch_important_feeds_from_feeds_data():
    # fetching important feeds
    all_important_feeds = list(feeds_data.find({"type": "important"}, {"_id": 0}))
    important_feeds_fixed_list = fixing_feeds(all_important_feeds)

    # sorting by date
    all_important_feeds_sorted_data = sorted(
        important_feeds_fixed_list, key=lambda x: parse_date(x["date"]), reverse=True
    )

    # limited_sorted_important_feeds = all_important_feeds_sorted_data[:20]

    final_list_of_feeds = []
    for news in all_important_feeds_sorted_data:
        news["from"] = "news"
        news["title"] = news["title"].strip()
        news["view_link"] = "/fdarticle/{}".format(news["article_number"])
        final_list_of_feeds.append(news)
    return final_list_of_feeds


# fetching important feeds from feeds_data
def fetch_important_tenders_from_tenders_data():
    # fetching important tenders
    all_important_tenders = list(
        tenders_data.find({"type": "important"}, {"_id": 0}).sort("tender_date", -1)
    )

    final_list_of_all_tenders = []
    for tender in all_important_tenders:
        tender["title"] = tender["tender_title"].strip()
        tender["date"] = tender["tender_date"].split(" ", 1)[0]
        tender["from"] = "procurement"
        tender_hash = tender["tender_hash"]
        tender["view_link"] = "/tndrsnwtbvw/{}".format(tender_hash)
        final_list_of_all_tenders.append(tender)

    return final_list_of_all_tenders


# # fetching important feeds from social_media_data
# def fetch_important_feeds_from_social_media_data():
#     # fetching important social media feeds
#     all_important_social_media_feeds = list(social_media_data.find({"type":"important"}, {'_id': 0}));

#     # Sort articles in descending order by date
#     all_important_social_media_feeds_sorted_data = sorted(all_important_social_media_feeds, key=lambda x: datetime.datetime.strptime(x['date'], '%Y-%m-%d %I:%M %p'), reverse=True)

#     limited_sorted_important_social_media_feeds = all_important_social_media_feeds_sorted_data[:50]

#     final_list_of_feeds = []
#     for news in limited_sorted_important_social_media_feeds:
#         news['from'] = "social"
#         final_list_of_feeds.append(news)

#     return final_list_of_feeds


def extracting_latest_date_for_important_highlight(list_value):
    # extracting latest date only
    all_dates = []
    for data in list_value:
        date = data["date"]
        all_dates.append(date)

    # Remove duplicates from date list by converting to a set
    final_date_list = list(set(all_dates))

    # sorting by date
    final_list_of_sorted_date = sorted(
        final_date_list, key=lambda x: parse_date(x), reverse=True
    )

    latest_date_value = final_list_of_sorted_date[0]
    return latest_date_value


@app.route("/hghdat", methods=["GET"])
def show_highlight_data_on_new_tab():
    if "email" not in session:
        return redirect(url_for("authentication"))

    page = request.args.get("page", default=1, type=int)
    per_page = 15

    # fetching important data only
    important_feeds = fetch_important_feeds_from_feeds_data()
    important_tenders = fetch_important_tenders_from_tenders_data()

    # creating final list of highlight data by merging feeds & tenders
    list_of_highlight_data = important_feeds + important_tenders

    # extraction of latest date from important highlight
    latest_date_from_important_highlight = (
        extracting_latest_date_for_important_highlight(list_of_highlight_data)
    )

    # extracting important highlight of specific date
    final_important_highlight = []
    for data in list_of_highlight_data:
        if data["date"] == latest_date_from_important_highlight:
            final_important_highlight.append(data)

    # creating required key & values for feeds & tenders data
    final_list_of_important_highlights = []
    for entry in final_important_highlight:
        type_of_entry = entry["from"]

        # creating image_link for feeds & tenders data
        if type_of_entry == "procurement":
            image_link = "/zhflgmg"
        elif type_of_entry == "news":
            image_link = gen_img_link(entry)

        # content fixing for feeds & tenders data
        if "content" in entry:
            content_value = entry["content"]
        else:
            content_value = "NA"

        entry["image_link"] = image_link
        entry["content"] = content_value
        final_list_of_important_highlights.append(entry)

    # checking len of fetched feeds
    total_feeds = len(final_list_of_important_highlights)

    # creating feeds on per page bases
    paginated_feeds = final_list_of_important_highlights[
        (page - 1) * per_page : page * per_page
    ]

    if len(paginated_feeds) == 0:
        last_valid_page = max(
            1,
            math.floor(total_feeds / per_page)
            + (1 if total_feeds % per_page != 0 else 0),
        )
        if page > last_valid_page:
            return redirect(
                url_for("show_highlight_data_on_new_tab", page=last_valid_page),
                code=302,
            )

    pagination_links = []
    num_pages = 0
    if total_feeds > per_page:
        num_pages = math.ceil(total_feeds / per_page)
        for p in range(1, num_pages + 1):
            if p * per_page <= total_feeds:
                pagination_links.append(
                    {
                        "page": p,
                        "url": url_for("show_highlight_data_on_new_tab", page=p),
                    }
                )
            else:
                break  # stop generating links if there are no more feeds

    return render_template(
        "highlight_data.html",
        feedlist=paginated_feeds,
        pagination_links=pagination_links,
        page=page,
        num_pages=num_pages,
    )


@app.route("/home", methods=["GET"])
def home():
    if "email" not in session:
        return redirect(url_for("authentication"))

    domain_name = request.url_root

    # fetching important data only
    important_feeds = fetch_important_feeds_from_feeds_data()
    important_tenders = fetch_important_tenders_from_tenders_data()

    # creating final list of highlight data by merging feeds & tenders
    list_of_highlight_data = important_feeds + important_tenders

    # extraction of latest date from important highlight
    latest_date_from_important_highlight = (
        extracting_latest_date_for_important_highlight(list_of_highlight_data)
    )

    # extracting important highlight of specific date
    final_important_highlight = []
    for data in list_of_highlight_data:
        if data["date"] == latest_date_from_important_highlight:
            final_important_highlight.append(data)

    limited_important_highlight = final_important_highlight[:10]

    return render_template(
        "home.html",
        feeds=limited_important_highlight,
        latest_date=latest_date_from_important_highlight,
    )


@app.route("/recevehm", methods=["POST"])
@jwt_required()
def recent_event_on_home():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        data = request.get_json()

        # Validate that 'data' is a dictionary and contains the key 'prvncenme'
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid input"}), 400

        if "prvncenme" not in data:
            return jsonify({"error": "Missing required key"}), 400

        # If validation passes, you can proceed with your logic
        prvncenme = data["prvncenme"]

        # validatin & sanitizing the received data
        sanitized_province_name = validating_string_and_sanitizing_string(prvncenme)

        if sanitized_province_name["string_validated"] == "true":

            province_name = sanitized_province_name["string_value"]

            if "inner" and "mongolia" in province_name:
                province_name = province_name.replace(" ", "_")

            all_recent_event_feeds_of_specific_region = list(
                feeds_data.find({"region": province_name}, {"_id": 0})
                .sort("date", -1)
                .limit(20)
            )
            fixed_list_of_recent_event_feeds = fixing_feeds(
                all_recent_event_feeds_of_specific_region
            )

            final_list_of_region_articles = []
            for region_feed in fixed_list_of_recent_event_feeds:
                if '",' or '"' or "," in region_feed["date"]:
                    region_feed["date"] = region_feed["date"].strip('",')
                    region_feed["date"] = region_feed["date"].strip(",")

                region_feed["article_link"] = "/fdarticle/{}".format(
                    region_feed["article_number"]
                )
                final_list_of_region_articles.append(region_feed)

            sorting_all_recent_event_feeds = sorted(
                final_list_of_region_articles,
                key=lambda x: parse_date(x["date"]),
                reverse=True,
            )

            return jsonify(sorting_all_recent_event_feeds)
        else:
            return jsonify({"error": "Invalid input"})
    else:
        return redirect(url_for("authentication"))


@app.route("/recbrdact", methods=["GET"])
@jwt_required()
def recent_border_activity_on_home():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Check if the access token is expired
    jwt_data = get_jwt()

    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        list_of_region = ["tibet", "yunnan", "xinjiang"]

        final_list_of_border_activities = []
        for province_name in list_of_region:
            all_recent_event_feeds_of_specific_region = list(
                feeds_data.find({"region": province_name}, {"_id": 0})
                .sort("date", -1)
                .limit(10)
            )
            fixed_list_of_recent_event_feeds = fixing_feeds(
                all_recent_event_feeds_of_specific_region
            )

            for specific_province_dict in fixed_list_of_recent_event_feeds:
                specific_province_dict["article_link"] = "/fdarticle/{}".format(
                    specific_province_dict["article_number"]
                )

            # Extend the final list with the fixed list
            final_list_of_border_activities.extend(fixed_list_of_recent_event_feeds)

        sorting_all_border_activities_feeds = sorted(
            final_list_of_border_activities,
            key=lambda x: parse_date(x["date"]),
            reverse=True,
        )

        if sorting_all_border_activities_feeds:
            return jsonify(sorting_all_border_activities_feeds)
        else:
            return jsonify({"error": "Error while fetching data"})


@app.route("/trending", methods=["GET"])
def get_trending_keywords():
    date = request.args.get("dt")
    if not date:
        return jsonify({"error": "Date parameter is required"}), 400

    trending_doc = trending_keywords_pie_data.find_one({"date": date}, {"_id": 0})

    if trending_doc:
        all_trending_keywords_documents = trending_doc.get("trending_keywords", [])

        # seperating keywords list bases on mil & pol
        list_of_military_trending_keywords = []
        list_of_political_trending_keywords = []
        for document in all_trending_keywords_documents:
            if document["category"] == "military":
                list_of_military_trending_keywords.append(document)
            elif document["category"] == "political":
                list_of_political_trending_keywords.append(document)

        # creating final_dict of trending keyword
        final_dict_of_trending_keywords = {}
        final_dict_of_trending_keywords["mil"] = list_of_military_trending_keywords
        final_dict_of_trending_keywords["pol"] = list_of_political_trending_keywords

        return jsonify(final_dict_of_trending_keywords)
    else:
        return jsonify({"warning": "No data found for the specified date"}), 404


# Function to handle date conversion and "NA" values
def get_date(item):
    date_str = item["date"]
    date_str = date_str.strip().rstrip(",")
    if date_str == "NA":
        return float("inf")
    return -datetime.datetime.strptime(date_str, "%Y-%m-%d").timestamp()


# Function to handle image presence
def get_image_priority(item):
    return (
        item["image"] == "NA"
    )  # Returns True if image is 'NA', which will sort it to the end


def fetch_recent_articles_from_feeds_data():
    # fetching recent articles from feeds data
    all_recent_feeds = list(feeds_data.find({}, {"_id": 0}).sort("date", -1).limit(100))
    fixed_list_of_recent_feeds = fixing_feeds(all_recent_feeds)

    # creating new list of recent feeds
    new_list_of_recent_feeds = []
    for feed in fixed_list_of_recent_feeds:
        if '",' in feed["date"]:
            feed["date"] = feed["date"].strip('",')
        feed["date"] = feed["date"].split(" ")[0]
        feed["title"] = feed["title"].strip()
        creating_image_link_for_feed = gen_img_link(feed)
        feed["image_link"] = creating_image_link_for_feed
        article_number = feed["article_number"]
        feed["article_link"] = "/fdarticle/{}".format(article_number)
        new_list_of_recent_feeds.append(feed)

    # # sorting the recent feeds in descending order by date and image presence
    # new_list_of_recent_feeds.sort(key=lambda x: (get_image_priority(x), get_date(x)))
    # sorted_recent_feeds = new_list_of_recent_feeds

    # sorting the recent feeds in descending order by date and image presence
    new_list_of_recent_feeds.sort(key=lambda x: (get_date(x)))
    sorted_recent_feeds = new_list_of_recent_feeds

    # final list of recent news are sended in frontend
    # final_list_of_recent_news_articles = sorted_recent_feeds[0:5]
    final_list_of_recent_news_articles = sorted_recent_feeds[0:6]

    return final_list_of_recent_news_articles


def fetch_military_articles_from_feeds_data():
    military_article_category = [
        "ground_force",
        "air_force",
        "rocket_force",
        "navy",
        "jlsf",
        "isf",
        "armed_force",
    ]

    # fetching military articles from feeds data
    list_of_all_military_articles = []

    for military_category in military_article_category:
        specific_military_category_feed = list(
            feeds_data.find({"category": military_category}, {"_id": 0})
            .sort("date", -1)
            .limit(10)
        )
        fixed_list_of_military_category_feed = fixing_feeds(
            specific_military_category_feed
        )
        list_of_all_military_articles.extend(fixed_list_of_military_category_feed)

    # creating new list of recent feeds
    new_list_of_mil_cat_feeds = []
    for feed in list_of_all_military_articles:
        if '",' in feed["date"]:
            feed["date"] = feed["date"].strip('",')
        feed["date"] = feed["date"].split(" ")[0]
        feed["title"] = feed["title"].strip()
        creating_image_link_for_feed = gen_img_link(feed)
        feed["image_link"] = creating_image_link_for_feed
        article_number = feed["article_number"]
        feed["article_link"] = "/fdarticle/{}".format(article_number)
        new_list_of_mil_cat_feeds.append(feed)

    # # sorting the recent feeds in descending order by date and image presence
    # new_list_of_mil_cat_feeds.sort(key=lambda x: (get_image_priority(x), get_date(x)))
    # sorted_mil_cat_feeds = new_list_of_mil_cat_feeds

    # sorting the recent feeds in descending order by date and image presence
    new_list_of_mil_cat_feeds.sort(key=lambda x: (get_date(x)))
    sorted_mil_cat_feeds = new_list_of_mil_cat_feeds

    # final list of recent news are sended in frontend
    # final_list_of_military_articles = sorted_mil_cat_feeds[0:5]
    final_list_of_military_articles = sorted_mil_cat_feeds[0:6]

    return final_list_of_military_articles


def fetch_propaganda_articles_from_feeds_data():

    # fetching propanganda articles from feeds data
    all_propaganda_feeds = list(
        feeds_data.find(
            {
                "$or": [
                    {"category": "propaganda"},  # Match if category is a string
                    {
                        "category": {"$in": ["propaganda"]}
                    },  # Match if category is an array and contains the string ctr
                ]
            },
            {"_id": 0},
        )
        .sort("date", -1)
        .limit(100)
    )
    # all_propaganda_feeds = list(feeds_data.find({"category": "propaganda"}, {'_id': 0}).sort("date", -1).limit(100));
    fixed_list_of_propaganda_feeds = fixing_feeds(all_propaganda_feeds)

    # creating new list of recent feeds
    new_list_of_propaganda_feeds = []
    for feed in fixed_list_of_propaganda_feeds:
        if '",' in feed["date"]:
            feed["date"] = feed["date"].strip('",')
        feed["date"] = feed["date"].split(" ")[0]
        feed["title"] = feed["title"].strip()
        creating_image_link_for_feed = gen_img_link(feed)
        feed["image_link"] = creating_image_link_for_feed
        article_number = feed["article_number"]
        feed["article_link"] = "/fdarticle/{}".format(article_number)
        new_list_of_propaganda_feeds.append(feed)

    # # sorting the recent feeds in descending order by date and image presence
    # new_list_of_propaganda_feeds.sort(key=lambda x: (get_image_priority(x), get_date(x)))
    # sorted_propaganda_feeds = new_list_of_propaganda_feeds

    # sorting the recent feeds in descending order by date and image presence
    new_list_of_propaganda_feeds.sort(key=lambda x: (get_date(x)))
    sorted_propaganda_feeds = new_list_of_propaganda_feeds

    # final list of recent news are sended in frontend
    # final_list_of_propaganda_news_articles = sorted_propaganda_feeds[0:5]
    final_list_of_propaganda_news_articles = sorted_propaganda_feeds[0:6]

    return final_list_of_propaganda_news_articles


@app.route("/feeds", methods=["GET"])
def feeds():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # recent articles from feeds_data
    final_list_of_recent_news_articles = fetch_recent_articles_from_feeds_data()

    # fetching all military articles
    final_list_of_military_news_articles = fetch_military_articles_from_feeds_data()

    # propaganda articles from feeds_data
    final_list_of_propaganda_news_articles = fetch_propaganda_articles_from_feeds_data()

    return render_template(
        "feeds.html",
        recent_news_articles=final_list_of_recent_news_articles,
        mil_news=final_list_of_military_news_articles,
        propaganda_news=final_list_of_propaganda_news_articles,
    )


@app.route("/alnws", methods=["GET"])
def all_news_articles():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Get request arguments
    arguments = request.args.to_dict()

    # Set default values for region (rg)
    rg = arguments.get("rg", "all")

    page = int(arguments.get("page", 1))
    per_page = 15

    if "border_updates" in rg:
        region_to_search = ["tibet", "xinjiang", "yunnan"]
        indian_border_news_list = []
        for region in region_to_search:
            specific_region_feeds = list(
                feeds_data.find({"region": region}, {"_id": 0}).sort("date", -1)
            )
            for region_news in specific_region_feeds:
                indian_border_news_list.append(region_news)

        fixed_list_of_feeds = fixing_feeds(indian_border_news_list)
    elif rg == "all":
        all_region_feeds = list(feeds_data.find({}, {"_id": 0}).sort("date", -1))
        fixed_list_of_feeds = fixing_feeds(all_region_feeds)
    elif rg != "all":
        specific_region_feeds = list(
            feeds_data.find({"region": rg}, {"_id": 0}).sort("date", -1)
        )
        fixed_list_of_feeds = fixing_feeds(specific_region_feeds)

    # creating new list of feeds
    new_list_of_feeds = []
    for feed in fixed_list_of_feeds:
        if '",' in feed["date"]:
            feed["date"] = feed["date"].strip('",')
        feed["date"] = feed["date"].split(" ")[0]
        feed["title"] = feed["title"].strip()
        creating_image_link_for_feed = gen_img_link(feed)
        feed["image_link"] = creating_image_link_for_feed
        article_number = feed["article_number"]
        feed["article_link"] = "/fdarticle/{}".format(article_number)
        new_list_of_feeds.append(feed)

    # sorting the feeds in descending order
    new_list_of_feeds.sort(key=get_date)
    sorted_feeds = new_list_of_feeds

    # checking len of fetched feeds
    total_feeds = len(sorted_feeds)

    # creating feeds on per page bases
    paginated_feeds = sorted_feeds[(page - 1) * per_page : page * per_page]

    if len(paginated_feeds) == 0:
        last_valid_page = max(
            1,
            math.floor(total_feeds / per_page)
            + (1 if total_feeds % per_page != 0 else 0),
        )
        if page > last_valid_page:
            return redirect(
                url_for("all_news_articles", page=last_valid_page), code=302
            )

    pagination_links = []
    num_pages = 0
    if total_feeds > per_page:
        num_pages = math.ceil(total_feeds / per_page)
        for p in range(1, num_pages + 1):
            if p * per_page <= total_feeds:
                pagination_links.append(
                    {"page": p, "url": url_for("all_news_articles", page=p)}
                )
            else:
                break  # stop generating links if there are no more feeds

    return render_template(
        "alnwsfeeds.html",
        feedlist=paginated_feeds,
        pagination_links=pagination_links,
        page=page,
        num_pages=num_pages,
    )


# def convert_dict_values(d):
#     return {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}


def convert_dict_values(d):
    def convert_value(v):
        if isinstance(v, list):
            return tuple(v)
        elif isinstance(v, dict):
            return tuple(sorted((k, convert_value(val)) for k, val in v.items()))
        return v

    return {k: convert_value(v) for k, v in d.items()}


@app.route("/mlnws", methods=["GET"])
def all_mil_articles():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Get request arguments
    arguments = request.args.to_dict()

    # Set default values for category (ctr)
    ctr = arguments.get("ctr", "all")

    page = int(arguments.get("page", 1))
    per_page = 15  # adjust this value to set the number of feeds per page

    if ctr == "all":
        # military_article_category = ["ground_force","air_force","rocket_force","navy","jlsf","armed_force"]
        military_article_category = [
            "ground_force",
            "air_force",
            "rocket_force",
            "navy",
            "jlsf",
            "armed_force",
            "isf",
        ]

        # fetching military articles from feeds data
        list_of_all_military_articles = []

        for military_category in military_article_category:
            specific_military_category_feed = list(
                feeds_data.find({"category": military_category}, {"_id": 0}).sort(
                    "date", -1
                )
            )
            fixed_list_of_military_category_feed = fixing_feeds(
                specific_military_category_feed
            )
            list_of_all_military_articles.extend(fixed_list_of_military_category_feed)

        fixing_all_feeds = fixing_feeds(list_of_all_military_articles)
        unique_feeds = {
            tuple(sorted(convert_dict_values(d).items())) for d in fixing_all_feeds
        }
        fixed_list_of_feeds = [dict(t) for t in unique_feeds]
    elif ctr != "all":
        specific_military_feeds = list(
            feeds_data.find(
                {
                    "$or": [
                        {"category": ctr},  # Match if category is a string
                        {
                            "category": {"$in": [ctr]}
                        },  # Match if category is an array and contains the string ctr
                    ]
                },
                {"_id": 0},
            ).sort("date", -1)
        )
        fixed_list_of_feeds = fixing_feeds(specific_military_feeds)

    # creating new list of feeds
    new_list_of_feeds = []
    for feed in fixed_list_of_feeds:
        if '",' in feed["date"]:
            feed["date"] = feed["date"].strip('",')
        feed["date"] = feed["date"].split(" ")[0]
        feed["title"] = feed["title"].strip()
        creating_image_link_for_feed = gen_img_link(feed)
        feed["image_link"] = creating_image_link_for_feed
        article_number = feed["article_number"]
        feed["article_link"] = "/fdarticle/{}".format(article_number)
        new_list_of_feeds.append(feed)

    # sorting the feeds in descending order
    new_list_of_feeds.sort(key=get_date)
    sorted_feeds = new_list_of_feeds

    # checking len of fetched feeds
    total_feeds = len(sorted_feeds)

    # creating feeds on per page bases
    paginated_feeds = sorted_feeds[(page - 1) * per_page : page * per_page]

    if len(paginated_feeds) == 0:
        last_valid_page = max(
            1,
            math.floor(total_feeds / per_page)
            + (1 if total_feeds % per_page != 0 else 0),
        )
        if page > last_valid_page:
            return redirect(url_for("all_mil_articles", page=last_valid_page), code=302)

    pagination_links = []
    num_pages = 0
    if total_feeds > per_page:
        num_pages = math.ceil(total_feeds / per_page)
        for p in range(1, num_pages + 1):
            if p * per_page <= total_feeds:
                pagination_links.append(
                    {"page": p, "url": url_for("all_mil_articles", page=p)}
                )
            else:
                break  # stop generating links if there are no more feeds

    return render_template(
        "milnwsfeeds.html",
        feedlist=paginated_feeds,
        pagination_links=pagination_links,
        page=page,
        num_pages=num_pages,
    )


@app.route("/prpnws", methods=["GET"])
def all_propaganda_news():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Get request arguments
    arguments = request.args.to_dict()

    # Set default values for region (rg)
    rg = arguments.get("rg", "all")

    page = int(arguments.get("page", 1))
    per_page = 15  # adjust this value to set the number of feeds per page

    if rg == "all":
        all_region_feeds = list(
            feeds_data.find(
                {
                    "$or": [
                        {"category": "propaganda"},  # Match if category is a string
                        {
                            "category": {"$in": ["propaganda"]}
                        },  # Match if category is an array and contains the string ctr
                    ]
                },
                {"_id": 0},
            ).sort("date", -1)
        )

        # all_region_feeds = list(feeds_data.find({"category":"propaganda"}, {'_id': 0}).sort("date", -1));
        fixed_list_of_feeds = fixing_feeds(all_region_feeds)
    elif rg != "all":
        specific_region_feeds = list(
            feeds_data.find({"region": rg}, {"_id": 0}).sort("date", -1)
        )
        fixed_list_of_feeds = fixing_feeds(specific_region_feeds)

    # creating new list of feeds
    new_list_of_feeds = []
    for feed in fixed_list_of_feeds:
        if '",' in feed["date"]:
            feed["date"] = feed["date"].strip('",')
        feed["date"] = feed["date"].split(" ")[0]
        feed["title"] = feed["title"].strip()
        creating_image_link_for_feed = gen_img_link(feed)
        feed["image_link"] = creating_image_link_for_feed
        article_number = feed["article_number"]
        feed["article_link"] = "/fdarticle/{}".format(article_number)
        new_list_of_feeds.append(feed)

    # sorting the feeds in descending order
    new_list_of_feeds.sort(key=get_date)
    sorted_feeds = new_list_of_feeds

    # checking len of fetched feeds
    total_feeds = len(sorted_feeds)

    # creating feeds on per page bases
    paginated_feeds = sorted_feeds[(page - 1) * per_page : page * per_page]

    if len(paginated_feeds) == 0:
        last_valid_page = max(
            1,
            math.floor(total_feeds / per_page)
            + (1 if total_feeds % per_page != 0 else 0),
        )
        if page > last_valid_page:
            return redirect(
                url_for("all_propaganda_news", page=last_valid_page), code=302
            )

    pagination_links = []
    num_pages = 0
    if total_feeds > per_page:
        num_pages = math.ceil(total_feeds / per_page)
        for p in range(1, num_pages + 1):
            if p * per_page <= total_feeds:
                pagination_links.append(
                    {"page": p, "url": url_for("all_propaganda_news", page=p)}
                )
            else:
                break  # stop generating links if there are no more feeds

    return render_template(
        "prpnwsfeeds.html",
        feedlist=paginated_feeds,
        pagination_links=pagination_links,
        page=page,
        num_pages=num_pages,
    )


def combine_tags_dict_values(input_dict):
    final_list_of_all_tags = []

    combined_list = []
    for key in input_dict:
        combined_list.extend(input_dict[key])

    # removing " from values of combined_list
    for item in combined_list:
        if '"' in item:
            item = item.replace('"', "")

        if '"' in item:
            item = item.replace('"', "")

        if '\\"' in item:
            item = item.replace('\\"', "")

        final_list_of_all_tags.append(item)
    return final_list_of_all_tags


@app.route("/fdarticle/<article_number>", methods=["GET"])
def fdarticle(article_number):
    if "email" not in session:
        return redirect(url_for("authentication"))

    full_article = feeds_data.find_one({"article_number": article_number})

    creating_list_with_single_dict = []
    creating_list_with_single_dict.append(full_article)

    if full_article:
        final_list_of_article = []

        for article in creating_list_with_single_dict:
            article["date"] = article["date"].split(" ", 1)[0]

            if isinstance(article["content"], str):
                article_content = article["content"]
                formatted_article_content = article_content

            elif isinstance(article["content"], list):
                article_contents = article["content"]

                # Wrap each paragraph in <p> tags
                spaces_removed_paragraphs = []
                for paragraph in article_contents:
                    content_without_leading_trailing_spaces = paragraph.strip()
                    spaces_removed_paragraphs.append(
                        content_without_leading_trailing_spaces
                    )

                formatted_article_content = "".join(
                    f"<p>{paragraph}</p>" for paragraph in spaces_removed_paragraphs
                )

            article["content"] = formatted_article_content
            article_domain = article["domain"]
            article_external_link = article["url"]

            # Check if 'tags' exists in article and is not None
            if "tags" in article and article["tags"] is not None:
                article_tags = article["tags"]
                final_list_of_all_tags = combine_tags_dict_values(article_tags)
            else:
                final_list_of_all_tags = "NA"

            article["article_tags"] = final_list_of_all_tags
            article_number = article["article_number"]
            article_image_link = gen_img_link(article)
            article["article_image_link"] = article_image_link
            final_list_of_article.append(article)

        return render_template("feed_article.html", article_data=final_list_of_article)
    else:
        return redirect(url_for("home"))


@app.route("/fthothart", methods=["GET"])
@jwt_required()
def fetch_some_article_on_full_article_page():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        # Fetch the most recent date from the documents
        recent_date_doc = feeds_data.find_one(
            {"date": {"$ne": "NA"}}, sort=[("date", -1)], projection={"date": 1}
        )

        if not recent_date_doc:
            return jsonify({"error": "No articles found"}), 404

        recent_date = recent_date_doc["date"]
        random_articles = list(feeds_data.find({"date": recent_date}))

        # If we have fewer than 4 articles, we need to fetch from the previous date
        if len(random_articles) < 20:
            # Calculate the previous date
            previous_date_doc = feeds_data.find_one(
                {"date": {"$lt": recent_date}},
                sort=[("date", -1)],
                projection={"date": 1},
            )

            while len(random_articles) < 20 and previous_date_doc:
                previous_date = previous_date_doc["date"]
                additional_articles = list(feeds_data.find({"date": previous_date}))
                random_articles.extend(additional_articles)

                # Remove duplicates if any
                random_articles = list(
                    {article["title"]: article for article in random_articles}.values()
                )  # Assuming 'title' is unique

                # Check for the next previous date
                previous_date_doc = feeds_data.find_one(
                    {"date": {"$lt": previous_date}},
                    sort=[("date", -1)],
                    projection={"date": 1},
                )

        # Remove the MongoDB ID from the articles for the response
        for article in random_articles:
            article.pop("_id", None)

        num_random_items = 4
        new_random_list = random.sample(random_articles, num_random_items)

        final_list_of_article = new_random_list

        # creating new list of feeds
        new_list_of_random_articles = []
        for feed in final_list_of_article:
            if '",' in feed["date"]:
                feed["date"] = feed["date"].strip('",')
            feed["date"] = feed["date"].split(" ")[0]
            feed["title"] = feed["title"].strip()
            creating_image_link_for_feed = gen_img_link(feed)
            feed["image_link"] = creating_image_link_for_feed
            article_number = feed["article_number"]
            feed["article_link"] = "/fdarticle/{}".format(article_number)
            new_list_of_random_articles.append(feed)

        return jsonify(new_list_of_random_articles)
    else:
        return jsonify({"error": "User not found"}), 404


@app.route("/fav_feeds", methods=["GET"])
def fav_feeds():
    if "email" not in session:
        return redirect(url_for("authentication"))

    page = int(request.args.get("page", 1))  # default to page 1
    per_page = 15  # adjust this value to set the number of feeds per page

    email = session["email"]
    check_user = users_data.find_one({"email": email}, {"_id": 0})

    if check_user:
        all_article_numbers = check_user["fav_feeds_article_number"]

        if len(all_article_numbers) == 0:
            return render_template(
                "fav_feeds.html", message="You don't have any favourite feeds yet."
            )
        else:
            # Sort the articles by serial_number in descending order
            sorted_article_numbers = sorted(
                all_article_numbers, key=lambda x: x["serial_number"], reverse=True
            )

            new_list_of_feeds = []
            for (
                article
            ) in sorted_article_numbers:  # Iterate over the sorted list of dictionaries
                article_number = article["article_number"]  # Extract article_number

                # checking article number in feeds and social media feed
                feed = feeds_data.find_one(
                    {"article_number": article_number}, {"_id": 0}
                )
                social_media_feed = social_media_data.find_one(
                    {"article_number": article_number}, {"_id": 0}
                )

                if feed:
                    feed["date"] = feed["date"].split(" ", 1)[0]

                    if isinstance(feed["content"], str):
                        # splitting content in three paragraph only if it is a string
                        if feed["content"] != "NA":
                            splitting_feed_content = split_english_paragraphs(
                                feed["content"]
                            )[:3]
                            creating_string_for_content = "".join(
                                splitting_feed_content
                            ).strip()
                            feed["content"] = creating_string_for_content
                    elif isinstance(feed["content"], list):
                        # merging content with three paragraph only if it is a list
                        splitting_feed_content = feed["content"][:3]
                        creating_string_for_content = " ".join(
                            splitting_feed_content
                        ).strip()
                        feed["content"] = creating_string_for_content

                    creating_image_link_for_feed = gen_img_link(feed)
                    feed["img_lnk"] = creating_image_link_for_feed
                    feed["arc_nm"] = article_number  # Use extracted article_number
                    feed["article_link"] = "/fdarticle/{}".format(article_number)
                    new_list_of_feeds.append(feed)
                elif social_media_feed:

                    if isinstance(social_media_feed["content"], str):
                        # splitting content in three paragraph only if it is a string
                        if social_media_feed["content"] != "NA":
                            splitting_feed_content = split_english_paragraphs(
                                social_media_feed["content"]
                            )[:3]
                            creating_string_for_content = "".join(
                                splitting_feed_content
                            ).strip()
                            social_media_feed["content"] = creating_string_for_content
                    elif isinstance(social_media_feed["content"], list):
                        # merging content with three paragraph only if it is a list
                        splitting_feed_content = social_media_feed["content"][:3]
                        creating_string_for_content = " ".join(
                            splitting_feed_content
                        ).strip()
                        social_media_feed["content"] = creating_string_for_content

                    creating_image_link_for_social_media_feed = (
                        creating_image_link_for_social_media(social_media_feed)
                    )
                    social_media_feed["img_lnk"] = (
                        creating_image_link_for_social_media_feed
                    )
                    social_media_feed["arc_nm"] = (
                        article_number  # Use extracted article_number
                    )
                    social_media_feed["article_link"] = "/smhart/{}".format(
                        article_number
                    )
                    new_list_of_feeds.append(social_media_feed)

            total_feeds = len(new_list_of_feeds)
            paginated_feeds = new_list_of_feeds[(page - 1) * per_page : page * per_page]

            if len(paginated_feeds) == 0:
                last_valid_page = max(
                    1,
                    math.floor(total_feeds / per_page)
                    + (1 if total_feeds % per_page != 0 else 0),
                )
                if page > last_valid_page:
                    return redirect(
                        url_for("fav_feeds", page=last_valid_page), code=302
                    )

            pagination_links = []
            if total_feeds > per_page:
                num_pages = math.ceil(total_feeds / per_page)
                for p in range(1, num_pages + 1):
                    if p * per_page <= total_feeds:
                        pagination_links.append(
                            {"page": p, "url": url_for("fav_feeds", page=p)}
                        )
                    else:
                        break  # stop generating links if there are no more feeds

            return render_template(
                "fav_feeds.html",
                favfeedlist=paginated_feeds,
                total_feeds_count=total_feeds,
                pagination_links=pagination_links if total_feeds > per_page else [],
                page=page,
                num_pages=num_pages if total_feeds > per_page else 1,
            )


@app.route("/ad_fav_fd", methods=["POST"])
@jwt_required()
def add_fav_feed():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        data = request.get_json()
        article_number = data["article_number"]
        user_email = session["email"]
        check_user_in_users_data = users_data.find_one({"email": user_email})

        if check_user_in_users_data:
            # Get the existing fav feeds
            existing_fav_feeds = check_user_in_users_data.get(
                "fav_feeds_article_number", []
            )

            # Check if the article_number already exists in fav feeds
            if any(
                feed["article_number"] == article_number for feed in existing_fav_feeds
            ):
                return jsonify({"warning": "Article already in favorites"})

            # Determine the next serial number
            if not existing_fav_feeds:
                serial_number = 1  # Start from 1 if no existing feeds
            else:
                # Extract existing serial numbers and find the maximum
                existing_serial_numbers = [
                    feed["serial_number"] for feed in existing_fav_feeds
                ]
                serial_number = (
                    max(existing_serial_numbers) + 1
                )  # Start from the next available number

            filter_query = {"email": user_email}
            update = {
                "$addToSet": {
                    "fav_feeds_article_number": {
                        "article_number": article_number,
                        "serial_number": serial_number,
                    }
                }
            }
            result = users_data.find_one_and_update(filter_query, update)

            if result:
                return jsonify({"success": "Favourite feed added successfully"})
            else:
                return jsonify({"error": "Failed to update user data"})
    return jsonify({"error": "Something went wrong"})


@app.route("/rm_fav_feeds", methods=["POST"])
@jwt_required()
def remove_fav_feeds():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        data = request.get_json()
        article_number_to_remove = data.get("article_Number")
        user_email = session["email"]
        check_user_in_users_data = users_data.find_one({"email": user_email})

        if check_user_in_users_data:
            # Update the filter query to match the structure
            filter_query = {
                "email": user_email,
                "fav_feeds_article_number.article_number": article_number_to_remove,
            }
            # Use $pull to remove the object with the specific article_number
            update = {
                "$pull": {
                    "fav_feeds_article_number": {
                        "article_number": article_number_to_remove
                    }
                }
            }
            result = users_data.update_one(filter_query, update)
            if result.modified_count > 0:
                return jsonify({"success": "Favourite feed removed successfully"})
            else:
                return jsonify({"error": "Favourite feed not found"})
    return jsonify({"error": "Something went wrong"})


@app.route("/chk_fav_feeds", methods=["POST"])
def check_fav_feeds_and_update_in_topbar():
    if "email" not in session:
        return redirect(url_for("authentication"))

    if request.method == "POST":
        user_email = session["email"]
        check_user_in_users_data = users_data.find_one({"email": user_email})

        if check_user_in_users_data:
            fav_feeds = check_user_in_users_data.get("fav_feeds_article_number", [])

            # Sort by serial_number in descending order
            sorted_fav_feeds = sorted(
                fav_feeds, key=lambda x: x["serial_number"], reverse=True
            )

            # Get the latest 5 entries
            latest_articles = sorted_fav_feeds[:5]

            # Getting the articles from feed
            final_data = []

            for article in latest_articles:
                article_number = article["article_number"]
                checking_article_in_feed = feeds_data.find_one(
                    {"article_number": article_number}, {"_id": 0}
                )
                checking_article_in_social_media_feed = social_media_data.find_one(
                    {"article_number": article_number}, {"_id": 0}
                )

                if checking_article_in_feed:
                    article_number = checking_article_in_feed["article_number"]
                    article_title = checking_article_in_feed["title"]
                    article_date = checking_article_in_feed["date"].split(" ")[0]
                    article_link = "fdarticle/" + article_number
                    final_data.append(
                        {
                            "article_number": article_number,
                            "article_title": article_title,
                            "article_date": article_date,
                            "article_link": article_link,
                        }
                    )

                elif checking_article_in_social_media_feed:
                    article_number = checking_article_in_social_media_feed[
                        "article_number"
                    ]
                    article_title = checking_article_in_social_media_feed.get("title")
                    article_date = checking_article_in_social_media_feed["date"].split(
                        " "
                    )[0]
                    article_link = "smhart/" + article_number

                    # If title is missing, use content and limit to 3 paragraphs
                    if not article_title:
                        content = checking_article_in_social_media_feed.get(
                            "content", ""
                        )

                        if isinstance(content, str):
                            # splitting content in three paragraph only if it is a string
                            if content != "NA":
                                splitting_feed_content = split_english_paragraphs(
                                    content
                                )[:3]
                                creating_string_for_content = "".join(
                                    splitting_feed_content
                                ).strip()
                                content = creating_string_for_content
                        elif isinstance(content, list):
                            # merging content with three paragraph only if it is a list
                            splitting_feed_content = content[:3]
                            creating_string_for_content = " ".join(
                                splitting_feed_content
                            ).strip()
                            content = creating_string_for_content

                        paragraphs = content.split(".")
                        limited_content = (
                            ". ".join(paragraphs[:3]) if paragraphs else content
                        )  # Join first 3 paragraphs
                        article_title = (
                            limited_content.strip()
                        )  # Use trimmed content as title

                    final_data.append(
                        {
                            "article_number": article_number,
                            "article_title": article_title,
                            "article_date": article_date,
                            "article_link": article_link,
                        }
                    )
            return jsonify(final_data)
    return jsonify({"error": "User  not found or no favorite feeds available."})


@app.route("/commented_feeds", methods=["GET"])
def commented_feeds():
    if "email" not in session:
        return redirect(url_for("authentication"))

    page = int(request.args.get("page", 1))  # default to page 1
    per_page = 15  # adjust this value to set the number of feeds per page

    email = session["email"]
    check_user = users_data.find_one({"email": email}, {"_id": 0})

    if check_user:
        all_article_numbers_with_comments_dict = check_user[
            "commented_feeds_article_number"
        ]

        if len(all_article_numbers_with_comments_dict) == 0:
            return render_template(
                "commented_feeds.html",
                message="You don't have any feeds with comment yet.",
            )
        elif len(all_article_numbers_with_comments_dict) >= 1:
            new_list_of_feeds = []
            for item in all_article_numbers_with_comments_dict:
                article_number = item["article_number"]
                comment_value = item["comment_value"]

                # trying to get article from feeds & social media feed
                feed = feeds_data.find_one(
                    {"article_number": article_number}, {"_id": 0}
                )
                social_media_feed = social_media_data.find_one(
                    {"article_number": article_number}, {"_id": 0}
                )

                # checking favourite feed article in both sections
                if feed:
                    feed["date"] = feed["date"].split(" ", 1)[0]

                    if isinstance(feed["content"], str):
                        # splitting content in three paragraph only if it is a string
                        if feed["content"] != "NA":
                            splitting_feed_content = split_english_paragraphs(
                                feed["content"]
                            )[:3]
                            creating_string_for_content = "".join(
                                splitting_feed_content
                            ).strip()
                            feed["content"] = creating_string_for_content
                    elif isinstance(feed["content"], list):
                        # merging content with three paragraph only if it is a list
                        splitting_feed_content = feed["content"][:3]
                        creating_string_for_content = " ".join(
                            splitting_feed_content
                        ).strip()
                        feed["content"] = creating_string_for_content

                    feed["user_comment"] = comment_value
                    creating_image_link_for_feed = gen_img_link(feed)
                    feed["img_lnk"] = creating_image_link_for_feed
                    feed["article_link"] = "/fdarticle/{}".format(article_number)
                    new_list_of_feeds.append(feed)
                elif social_media_feed:

                    if isinstance(social_media_feed["content"], str):
                        # splitting content in three paragraph only if it is a string
                        if social_media_feed["content"] != "NA":
                            splitting_feed_content = split_english_paragraphs(
                                social_media_feed["content"]
                            )[:3]
                            creating_string_for_content = "".join(
                                splitting_feed_content
                            ).strip()
                            social_media_feed["content"] = creating_string_for_content
                    elif isinstance(social_media_feed["content"], list):
                        # merging content with three paragraph only if it is a list
                        splitting_feed_content = social_media_feed["content"][:3]
                        creating_string_for_content = " ".join(
                            splitting_feed_content
                        ).strip()
                        social_media_feed["content"] = creating_string_for_content

                    social_media_feed["user_comment"] = comment_value
                    creating_image_link_for_social_media_feed = (
                        creating_image_link_for_social_media(social_media_feed)
                    )
                    social_media_feed["img_lnk"] = (
                        creating_image_link_for_social_media_feed
                    )
                    social_media_feed["article_link"] = "/smhart/{}".format(
                        article_number
                    )
                    new_list_of_feeds.append(social_media_feed)

            total_feeds = len(new_list_of_feeds)
            paginated_feeds = new_list_of_feeds[(page - 1) * per_page : page * per_page]

            if len(paginated_feeds) == 0:
                # if the page has no feeds, redirect to the last valid page
                last_valid_page = max(
                    1,
                    math.floor(total_feeds / per_page)
                    + (1 if total_feeds % per_page != 0 else 0),
                )
                if page > last_valid_page:
                    return redirect(
                        url_for("commented_feeds", page=last_valid_page), code=302
                    )

            pagination_links = []
            if total_feeds > per_page:
                num_pages = math.ceil(total_feeds / per_page)
                for p in range(1, num_pages + 1):
                    if p * per_page <= total_feeds:
                        pagination_links.append(
                            {"page": p, "url": url_for("commented_feeds", page=p)}
                        )
                    else:
                        break  # stop generating links if there are no more feeds

            return render_template(
                "commented_feeds.html",
                cmmntfeedlist=paginated_feeds,
                total_feeds_count=total_feeds,
                pagination_links=pagination_links if total_feeds > per_page else [],
                page=page,
                num_pages=num_pages if total_feeds > per_page else 1,
            )


@app.route("/ad_cmt_fd", methods=["POST"])
@jwt_required()
def add_comment_feed():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        data = request.get_json()
        article_number = data["article_number"]
        comment_value = data["comment_value"]

        if not article_number or not comment_value:
            return jsonify({"error": "Missing required keys"}), 400

        # Sanitize comment value
        comment_validation = validating_comment_and_sanitizing_comment(comment_value)

        if comment_validation["comment_validation"] == "true":
            comment_value = comment_validation["comment_value"]
            user_email = user["email"]
            check_user_in_users_data = users_data.find_one({"email": user_email})

            if check_user_in_users_data:
                # Check if the comment already exists
                existing_comment = users_data.find_one(
                    {
                        "email": user_email,
                        "commented_feeds_article_number.article_number": article_number,
                    }
                )

                if existing_comment:
                    # Update the existing comment
                    filter_query = {
                        "email": user_email,
                        "commented_feeds_article_number.article_number": article_number,
                    }
                    update = {
                        "$set": {
                            "commented_feeds_article_number.$.comment_value": comment_value
                        }
                    }
                    users_data.update_one(filter_query, update)
                    return jsonify({"warning": "Comment updated successfully"})
                else:
                    # Add a new comment if it doesn't exist
                    filter_query = {"email": user_email}
                    update = {
                        "$addToSet": {
                            "commented_feeds_article_number": {
                                "article_number": article_number,
                                "comment_value": comment_value,
                            }
                        }
                    }
                    users_data.find_one_and_update(filter_query, update)
                    return jsonify({"success": "Comment added successfully"})
        else:
            return jsonify({"error": "Invalid comment value"}), 404
    else:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"error": "Something went wrong"})


@app.route("/up_cmt_fd", methods=["POST"])
@jwt_required()
def update_comment_feed():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        data = request.get_json()
        article_number = data["article_number"]
        new_comment_value = data["comment_value"]

        if not article_number or not new_comment_value:
            return jsonify({"error": "Missing required keys"}), 400

        # Sanitize comment value
        comment_validation = validating_comment_and_sanitizing_comment(
            new_comment_value
        )

        if comment_validation["comment_validation"] == "true":
            new_comment_value = comment_validation["comment_value"]

            user_email = session["email"]
            check_user_in_users_data = users_data.find_one({"email": user_email})

            if check_user_in_users_data:
                filter_query = {
                    "email": user_email,
                    "commented_feeds_article_number.article_number": article_number,
                }
                update = {
                    "$set": {
                        "commented_feeds_article_number.$.comment_value": new_comment_value
                    }
                }
                result = users_data.find_one_and_update(filter_query, update)
                return jsonify({"success": "Comment updated succesfully"})
        else:
            return jsonify({"error": "Invalid comment value"}), 404
    else:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"error": "Something went wrong"})


@app.route("/rm_cmt_feeds", methods=["POST"])
@jwt_required()
def remove_comment_feed():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        data = request.get_json()
        article_number_to_remove = data.get("article_Number")
        user_email = session["email"]
        check_user_in_users_data = users_data.find_one({"email": user_email})

        if check_user_in_users_data:
            filter_query = {"email": user_email}
            update = {
                "$pull": {
                    "commented_feeds_article_number": {
                        "article_number": article_number_to_remove
                    }
                }
            }
            result = users_data.update_one(filter_query, update)
            if result.modified_count > 0:
                return jsonify({"success": "Commented feed removed successfully"})
            else:
                return jsonify({"error": "Commented feed not found"})
    else:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"error": "Something went wrong"})


@app.route("/search", methods=["GET"])
def search():
    if "email" not in session:
        return redirect(url_for("authentication"))
    return render_template("search.html")


# image dir of social_media
social_media_image_dir = "social_media"


@app.route("/smhres/<image_name>")
def serve_social_media_image(image_name):
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Validate the image name to prevent directory traversal
    if not re.match(
        r"^[\w\-. ]+$", image_name
    ):  # Allow only alphanumeric, dash, underscore, dot, and space
        return redirect(url_for("china_flag_image"))

    # Search for the image in all subdirectories
    for root, dirs, files in os.walk(social_media_image_dir):
        if image_name in files:
            # If the image is found, return it using send_from_directory
            return send_from_directory(root, image_name)

    return redirect(url_for("china_flag_image"))


# def creating_image_link_for_social_media(image_name_value):
#     image_name = image_name_value
#     image_path = os.path.join('smhres', image_name_value)

#     # Search for the image in all subdirectories
#     for root, dirs, files in os.walk(social_media_image_dir):
#         if image_name in files:
#             domain_name = request.url_root
#             complete_image_link = "{}{}".format(domain_name, image_path)
#             return complete_image_link
#     # If the image is not found, return "NA"
#     return "NA"


def creating_image_link_for_social_media(dictionary, image_dir=social_media_image_dir):
    # Check if 'image' key exists in the dictionary
    if "image" not in dictionary:
        return "NA"

    # If 'image' is a string, use it directly
    if isinstance(dictionary["image"], str):
        image_name = dictionary["image"]
        image_path = os.path.join("smhres", image_name)

        # Search for the image in all subdirectories
        for root, dirs, files in os.walk(image_dir):
            if image_name in files:
                domain_name = request.url_root
                complete_image_link = "{}{}".format(domain_name, image_path)
                return complete_image_link
        return "NA"  # Return "NA" if the image does not exist

    # If 'image' is a list, randomly select an image
    elif isinstance(dictionary["image"], list):
        for img in dictionary["image"]:
            image_name = img
            image_path = os.path.join("smhres", image_name)

            # Search for the image in all subdirectories
            for root, dirs, files in os.walk(image_dir):
                if image_name in files:
                    domain_name = request.url_root
                    complete_image_link = "{}{}".format(domain_name, image_path)
                    return complete_image_link
        return "NA"  # Return "NA" if no valid image was found

    return "NA"  # Return "NA" if 'image' is neither a string nor a list


@app.route("/smh", methods=["GET"])
def social_media_highlights():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Get request arguments
    arguments = request.args.to_dict()

    # Set default values for ctr and dt
    ctr = arguments.get("ctr", "all")
    dt = arguments.get("dt", "this month")
    # dt = arguments.get('dt', 'last 7 days')

    if " " in dt:
        space_count = dt.count(" ")
        if space_count == 1:
            splitting_dt = dt.strip().split(" ")
            first_value = splitting_dt[0].lower()
            second_value = splitting_dt[1].lower()
            final_dt_value = " ".join([first_value, second_value])
        elif space_count == 2:
            splitting_dt = dt.strip().split(" ")
            first_value = splitting_dt[0].lower()
            second_value = splitting_dt[1]
            third_value = splitting_dt[2].lower()
            final_dt_value = " ".join([first_value, second_value, third_value])
        dt = final_dt_value
    else:
        dt = dt.lower()

    # Initialize date range
    start_date = None
    end_date = datetime.datetime.now()

    if dt == "today":
        start_date = (end_date - datetime.timedelta(days=1)).replace(
            hour=23, minute=59, second=59, microsecond=0
        )
        end_date = start_date + datetime.timedelta(days=1)
    elif dt == "yesterday":
        start_date = (end_date - datetime.timedelta(days=2)).replace(
            hour=23, minute=59, second=59, microsecond=0
        )
        end_date = start_date + datetime.timedelta(days=1)
    elif dt == "last 7 days":
        start_date = (end_date - datetime.timedelta(days=7)).replace(
            hour=23, minute=59, second=59, microsecond=0
        )
    elif dt == "last 15 days":
        start_date = (end_date - datetime.timedelta(days=15)).replace(
            hour=23, minute=59, second=59, microsecond=0
        )
    elif dt == "this month":
        start_date = end_date.replace(day=1)
    elif dt == "last month":
        # Get the first day of the current month
        first_day_of_current_month = end_date.replace(day=1)

        # Get the last day of the last month
        last_day_of_last_month = first_day_of_current_month - datetime.timedelta(days=1)

        # Set the start date to the first day of the last month
        start_date = last_day_of_last_month.replace(day=1)

        # Set the end date to the last day of the last month
        end_date = last_day_of_last_month
    elif dt == "custom range":
        # Handle custom date range from request arguments
        start_date_str = arguments.get("start_date")
        end_date_str = arguments.get("end_date")
        if start_date_str and end_date_str:
            start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = (
                datetime.datetime.strptime(end_date_str, "%Y-%m-%d")
                + datetime.timedelta(days=1)
                - datetime.timedelta(seconds=1)
            )

    # # Default to 7 days if no valid date range is found
    # if start_date is None:
    #     start_date = end_date - datetime.timedelta(days=7)

    # Default to the beginning of the current month if no valid date range is found
    if start_date is None:
        start_date = end_date.replace(day=1)

    page = int(arguments.get("page", 1))
    per_page = 15

    # Query MongoDB based on ctr and date range
    if ctr == "twitter":
        all_articles_of_specific_category = list(
            social_media_data.find(
                {
                    "domain": "twitter.com",
                    "date": {
                        "$gte": start_date.strftime("%Y-%m-%d %I:%M %p"),
                        "$lte": end_date.strftime("%Y-%m-%d %I:%M %p"),
                    },
                },
                {"_id": 0},
            )
        )

    elif ctr == "weibo":
        all_articles_of_specific_category = list(
            social_media_data.find(
                {
                    "domain": "weibo.cn",
                    "date": {
                        "$gte": start_date.strftime("%Y-%m-%d %I:%M %p"),
                        "$lte": end_date.strftime("%Y-%m-%d %I:%M %p"),
                    },
                },
                {"_id": 0},
            )
        )

    elif ctr == "bilibili":
        all_articles_of_specific_category = list(
            social_media_data.find(
                {
                    "domain": "bilibili.com",
                    "date": {
                        "$gte": start_date.strftime("%Y-%m-%d %I:%M %p"),
                        "$lte": end_date.strftime("%Y-%m-%d %I:%M %p"),
                    },
                },
                {"_id": 0},
            )
        )

    elif ctr == "sohu":
        all_articles_of_specific_category = list(
            social_media_data.find(
                {
                    "domain": "sohu.com",
                    "date": {
                        "$gte": start_date.strftime("%Y-%m-%d %I:%M %p"),
                        "$lte": end_date.strftime("%Y-%m-%d %I:%M %p"),
                    },
                },
                {"_id": 0},
            )
        )

    elif ctr == "all":
        all_articles_of_specific_category = list(
            social_media_data.find(
                {
                    "date": {
                        "$gte": start_date.strftime("%Y-%m-%d %I:%M %p"),
                        "$lte": end_date.strftime("%Y-%m-%d %I:%M %p"),
                    }
                },
                {"_id": 0},
            )
        )

    # Sort articles in descending order by date
    sorted_data = sorted(
        all_articles_of_specific_category,
        key=lambda x: datetime.datetime.strptime(x["date"], "%Y-%m-%d %I:%M %p"),
        reverse=True,
    )

    list_of_all_social_media_articles = []
    for articles in sorted_data:

        # if isinstance(articles['image'], str):
        #     image_name = articles['image']
        # elif isinstance(articles['image'], list):
        #     image_name = articles['image'][0]

        if isinstance(articles["content"], str):
            # splitting content in three paragraph only if it is a string
            if articles["content"] != "NA":
                splitting_feed_content = split_english_paragraphs(articles["content"])[
                    :3
                ]
                creating_string_for_content = "".join(splitting_feed_content).strip()
                articles["content"] = creating_string_for_content
        elif isinstance(articles["content"], list):
            # merging content with three paragraph only if it is a list
            splitting_feed_content = articles["content"][:3]
            creating_string_for_content = " ".join(splitting_feed_content).strip()
            articles["content"] = creating_string_for_content

        # articles['image'] = image_name
        creating_image_link = creating_image_link_for_social_media(articles)
        # creating_image_link = creating_image_link_for_social_media(image_name)
        articles["image_link"] = creating_image_link
        articles["article_link"] = "/smhart/{}".format(articles["article_number"])
        list_of_all_social_media_articles.append(articles)

    # checking len of fetched feeds
    total_feeds = len(list_of_all_social_media_articles)

    # creating feeds on per page bases
    paginated_feeds = list_of_all_social_media_articles[
        (page - 1) * per_page : page * per_page
    ]

    if len(paginated_feeds) == 0:
        last_valid_page = max(
            1,
            math.floor(total_feeds / per_page)
            + (1 if total_feeds % per_page != 0 else 0),
        )
        if page > last_valid_page:
            return redirect(
                url_for("social_media_highlights", page=last_valid_page), code=302
            )

    pagination_links = []
    num_pages = 0
    if total_feeds > per_page:
        num_pages = math.ceil(total_feeds / per_page)
        for p in range(1, num_pages + 1):
            if p * per_page <= total_feeds:
                pagination_links.append({"page": p, "url": url_for("feeds", page=p)})
            else:
                break  # stop generating links if there are no more feeds
    return render_template(
        "smh.html",
        articles=paginated_feeds,
        pagination_links=pagination_links,
        page=page,
        num_pages=num_pages,
        total_feeds=total_feeds,
    )


@app.route("/smhart/<article_number>", methods=["GET"])
def social_media_article(article_number):
    if "email" not in session:
        return redirect(url_for("authentication"))

    full_article = social_media_data.find_one({"article_number": article_number})

    creating_list_with_single_list = []
    creating_list_with_single_list.append(full_article)

    if full_article:
        final_list_of_article = []
        for article in creating_list_with_single_list:

            if isinstance(article["content"], str):
                article_content = article["content"]
                formatted_article_content = article_content

            elif isinstance(article["content"], list):
                article_contents = article["content"]

                # Wrap each paragraph in <p> tags
                spaces_removed_paragraphs = []
                for paragraph in article_contents:
                    content_without_leading_trailing_spaces = paragraph.strip()
                    spaces_removed_paragraphs.append(
                        content_without_leading_trailing_spaces
                    )

                formatted_article_content = "".join(
                    f"<p>{paragraph}</p>" for paragraph in spaces_removed_paragraphs
                )

            # Check if 'tags' exists in article and is not None
            if "tags" in article and article["tags"] is not None:
                article_tags = article["tags"]
                final_list_of_all_tags = combine_tags_dict_values(article_tags)
            else:
                final_list_of_all_tags = "NA"

            article["article_tags"] = final_list_of_all_tags
            article["content"] = formatted_article_content
            article_image_link = creating_image_link_for_social_media(article)
            article["article_image_link"] = article_image_link
            final_list_of_article.append(article)
        return render_template(
            "social_media_feed_article.html", article_data=final_list_of_article
        )
    else:
        return redirect(url_for("home"))


@app.route("/fthothsmhart", methods=["GET"])
@jwt_required()
def fetch_some_article_on_full_article_page_of_social_media():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        # Fetch the most recent date from the documents
        recent_date_doc = social_media_data.find_one(
            {"date": {"$ne": "NA"}}, sort=[("date", -1)], projection={"date": 1}
        )

        if not recent_date_doc:
            return jsonify({"error": "No articles found"}), 404

        recent_date = recent_date_doc["date"]
        random_articles = list(social_media_data.find({"date": recent_date}))

        if len(random_articles) < 20:
            # Calculate the previous date
            previous_date_doc = social_media_data.find_one(
                {"date": {"$lt": recent_date}},
                sort=[("date", -1)],
                projection={"date": 1},
            )

            while len(random_articles) < 20 and previous_date_doc:
                previous_date = previous_date_doc["date"]
                additional_articles = list(
                    social_media_data.find({"date": previous_date})
                )
                random_articles.extend(additional_articles)

                # Remove duplicates if any
                random_articles = list(
                    {
                        ",".join(article["content"]): article
                        for article in random_articles
                    }.values()
                )

                # Check for the next previous date
                previous_date_doc = social_media_data.find_one(
                    {"date": {"$lt": previous_date}},
                    sort=[("date", -1)],
                    projection={"date": 1},
                )

        # Remove the MongoDB ID from the articles for the response
        for article in random_articles:
            article.pop("_id", None)

        num_random_items = 4
        new_random_list = random.sample(random_articles, num_random_items)

        final_list_of_article = new_random_list

        # creating new list of feeds
        new_list_of_random_articles = []
        for feed in final_list_of_article:

            if '",' in feed["date"]:
                feed["date"] = feed["date"].strip('",')
            feed["date"] = feed["date"].split(" ")[0]

            if "title" in feed:
                feed["title"] = feed["title"].strip()

            if isinstance(feed["content"], str):
                # splitting content in three paragraph only if it is a string
                if feed["content"] != "NA":
                    splitting_feed_content = split_english_paragraphs(feed["content"])[
                        :1
                    ]
                    creating_string_for_content = "".join(
                        splitting_feed_content
                    ).strip()
                    feed["content"] = creating_string_for_content
            elif isinstance(feed["content"], list):
                # merging content with three paragraph only if it is a list
                splitting_feed_content = feed["content"][:1]
                creating_string_for_content = " ".join(splitting_feed_content).strip()
                feed["content"] = creating_string_for_content

            creating_image_link_for_feed = creating_image_link_for_social_media(feed)
            feed["image_link"] = creating_image_link_for_feed
            feed["article_link"] = "/smhart/{}".format(feed["article_number"])
            new_list_of_random_articles.append(feed)

        return jsonify(new_list_of_random_articles)
    else:
        return jsonify({"error": "User not found"}), 404


@app.route("/tenders", methods=["GET"])
def tenders():
    if "email" not in session:
        return redirect(url_for("authentication"))

    all_tenders = list(tenders_data.find({}, {"_id": 0}).sort("tender_date", -1))
    final_list_of_all_tenders = []
    for tenders in all_tenders:
        tenders["tender_date"] = tenders["tender_date"].split(" ", 1)[0]
        tenders_hash = tenders["tender_hash"]
        tenders["tender_view_link"] = "/tndrs/{}".format(tenders_hash)
        tenders["tender_download_link"] = "/tndrsdwn/{}".format(tenders_hash)

        suggested_topic = tenders.get("suggested_topic", None)

        if suggested_topic is not None and isinstance(suggested_topic, str):
            perfect_suggested_topic = suggested_topic.strip()
        else:
            perfect_suggested_topic = None

        if perfect_suggested_topic and (
            perfect_suggested_topic != ""
            and perfect_suggested_topic != "NA"
            and perfect_suggested_topic != "na"
        ):
            tenders["s_top"] = perfect_suggested_topic
        else:
            tenders["s_top"] = "NA"

        final_list_of_all_tenders.append(tenders)

    # Extracting all suggested topics and store in a temp list
    templist_all_topics = []
    for t in all_tenders:
        suggested_topic_name = t.get("suggested_topic", None)

        if suggested_topic_name is not None and isinstance(suggested_topic_name, str):
            perfect_suggested_topic_name = suggested_topic_name.strip()
        else:
            perfect_suggested_topic_name = None

        if perfect_suggested_topic_name and (
            perfect_suggested_topic_name != ""
            and perfect_suggested_topic_name != "NA"
            and perfect_suggested_topic_name != "na"
        ):
            templist_all_topics.append(perfect_suggested_topic_name)

    # removing duplicates from templist of all topics
    topics_in_tuple = tuple(templist_all_topics)

    # Convert the tuple back to a list, using set to remove duplicates
    final_list_of_all_topics = list(set(topics_in_tuple))

    return render_template(
        "tenders.html",
        tndrs_data=final_list_of_all_tenders,
        all_suggested_topics=final_list_of_all_topics,
    )


# @app.route("/tenders", methods=['GET'])
# def tenders():
#    if 'email' not in session:
#        return redirect(url_for('authentication'))
#
#    all_tenders = list(tenders_data.find({}, {'_id': 0}).sort("tender_date", -1));
#    final_list_of_all_tenders = []
#    for tenders in all_tenders:
#        tenders['tender_date'] = tenders['tender_date'].split(" ", 1)[0]
#        tenders_hash = tenders['tender_hash']
#        tenders['tender_view_link'] = "/tndrs/{}".format(tenders_hash)
#        tenders['tender_download_link'] = "/tndrsdwn/{}".format(tenders_hash)
#        final_list_of_all_tenders.append(tenders)
#    return render_template('tenders.html', tndrs_data=final_list_of_all_tenders)


# image dir of mucd
mucd_image_dir = "mucd_images"


@app.route("/mcdres/<image_name>")
def serve_mucd_image(image_name):
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Validate the image name to prevent directory traversal
    if not re.match(
        r"^[\w\-. ]+$", image_name
    ):  # Allow only alphanumeric, dash, underscore, dot, and space
        return redirect(url_for("no_image_available"))

    # Search for the image in all subdirectories
    for root, dirs, files in os.walk(mucd_image_dir):
        if image_name in files:
            # If the image is found, return it using send_from_directory
            return send_from_directory(root, image_name)
    return redirect(url_for("no_image_available"))


def creating_image_link_for_mucd(image_name_value):
    image_name = image_name_value
    image_path = os.path.join("mcdres", image_name_value)

    # Search for the image in all subdirectories
    for root, dirs, files in os.walk(mucd_image_dir):
        if image_name in files:
            domain_name = request.url_root
            complete_image_link = "{}{}".format(domain_name, image_path)
            return complete_image_link

    # If the image is not found, return "NA"
    return "NA"


def validate_and_sanitize_numeric_data(input_data):
    # Ensure input_data is a string
    if isinstance(input_data, str):
        # Strip whitespace and check if it consists only of digits
        sanitized_input = input_data.strip()

        if sanitized_input.isdigit():
            # Check the length of the sanitized input
            if 3 <= len(sanitized_input) <= 6:
                return {"string_validated": "true", "string_value": sanitized_input}
            else:
                return {"string_validated": "false", "string_value": sanitized_input}
        else:
            return {"string_validated": "false", "string_value": input_data}
    else:
        return {"string_validated": "false", "string_value": input_data}


@app.route("/mucd_track", methods=["GET"])
def mucd_track():
    if "email" not in session:
        return redirect(url_for("authentication"))
    return render_template("mucd_track.html")


@app.route("/mucd_check", methods=["POST"])
@jwt_required()
def mucd_checking_mucd_number():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        mucd_number = request.form["mdtknm"]
        mucd_code_validation = validate_and_sanitize_numeric_data(mucd_number)

        if mucd_code_validation["string_validated"] == "true":
            mucd_code = mucd_code_validation["string_value"]

            latest_mucd_entry = mucd_data.find(
                {"mucd_number": str(mucd_code)}, {"_id": 0}
            ).sort("date", DESCENDING)

            # Convert the cursor to a list
            latest_mucd_entry = list(latest_mucd_entry)

            if latest_mucd_entry:
                if len(latest_mucd_entry) >= 2:
                    checking_mucd_number_in_db = latest_mucd_entry[0]
                    checking_mucd_number_in_db["history"] = "yes"
                else:
                    checking_mucd_number_in_db = latest_mucd_entry[0]
                    checking_mucd_number_in_db["history"] = "no"

                if checking_mucd_number_in_db:
                    final_dict = replace_blank_and_NA_values(checking_mucd_number_in_db)
                    fixing_theatre_command_spell = fixing_theatre_command_spelling(
                        final_dict
                    )

                    # creating image link for mucd entry
                    if "image" in fixing_theatre_command_spell:
                        main_image_value = fixing_theatre_command_spell["image"]
                        creating_main_image_link = creating_image_link_for_mucd(
                            main_image_value
                        )
                        fixing_theatre_command_spell["main_image_link"] = (
                            creating_main_image_link
                        )
                    else:
                        fixing_theatre_command_spell["image"] = "NA"

                    # creating force name using force_type
                    if "force_type" in fixing_theatre_command_spell:
                        splitting_force_name = fixing_theatre_command_spell[
                            "force_type"
                        ].split("_")
                        force_name = " ".join(splitting_force_name).title()
                        fixing_theatre_command_spell["force_name"] = force_name

                    # checking doc_uid for mucd entry
                    if "doc_uid" in fixing_theatre_command_spell:
                        fixing_theatre_command_spell["doc_uid"] = (
                            fixing_theatre_command_spell["doc_uid"]
                        )
                    else:
                        fixing_theatre_command_spell["doc_uid"] = "NA"

                    # creating sub images link for mucd entry
                    if "sub_images" in fixing_theatre_command_spell:
                        sub_images = fixing_theatre_command_spell["sub_images"]

                        final_list_of_sub_images_of_mucd = []
                        if isinstance(sub_images, list):
                            for img in sub_images:
                                image_link = creating_image_link_for_mucd(img)
                                final_list_of_sub_images_of_mucd.append(image_link)
                        elif isinstance(sub_images, str):
                            image_link = creating_image_link_for_mucd(sub_images)
                            final_list_of_sub_images_of_mucd.append(image_link)

                        fixing_theatre_command_spell["sub_images_link"] = (
                            final_list_of_sub_images_of_mucd
                        )

                    result = fixing_theatre_command_spell
                    return jsonify(result)
            else:
                result = {"error": "number not available"}
                return jsonify(result)
        else:
            return jsonify({"error": "Invalid input"}), 404
    else:
        return redirect(url_for("authentication"))


@app.route("/mcd_sngl", methods=["POST"])
@jwt_required()
def single_mucd_number_checking():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        mucd_number = request.form["mdtknm"]
        mucd_code_validation = validate_and_sanitize_numeric_data(mucd_number)

        if mucd_code_validation["string_validated"] == "true":
            mucd_code = mucd_code_validation["string_value"]
            mucd_code = int(mucd_code)

            latest_mucd_entry = (
                mucd_data.find({"mucd_number": str(mucd_code)}, {"_id": 0})
                .sort("date", DESCENDING)
                .limit(1)
            )

            # Convert the cursor to a list and get the first (and only) entry
            latest_mucd_entry = list(latest_mucd_entry)

            if latest_mucd_entry:
                checking_mucd_number_in_db = latest_mucd_entry[0]

                if checking_mucd_number_in_db:
                    final_dict = replace_blank_and_NA_values(checking_mucd_number_in_db)
                    fixing_theatre_command_spell = fixing_theatre_command_spelling(
                        final_dict
                    )

                    # creating image link for mucd entry
                    if "image" in fixing_theatre_command_spell:
                        main_image_value = fixing_theatre_command_spell["image"]
                        creating_main_image_link = creating_image_link_for_mucd(
                            main_image_value
                        )
                        fixing_theatre_command_spell["main_image_link"] = (
                            creating_main_image_link
                        )
                    else:
                        fixing_theatre_command_spell["image"] = "NA"

                    # creating force name using force_type
                    if "force_type" in fixing_theatre_command_spell:
                        splitting_force_name = fixing_theatre_command_spell[
                            "force_type"
                        ].split("_")
                        force_name = " ".join(splitting_force_name).title()
                        fixing_theatre_command_spell["force_name"] = force_name

                    # checking doc_uid for mucd entry
                    if "doc_uid" in fixing_theatre_command_spell:
                        fixing_theatre_command_spell["doc_uid"] = (
                            fixing_theatre_command_spell["doc_uid"]
                        )
                    else:
                        fixing_theatre_command_spell["doc_uid"] = "NA"

                    # creating sub images link for mucd entry
                    if "sub_images" in fixing_theatre_command_spell:
                        sub_images = fixing_theatre_command_spell["sub_images"]

                        final_list_of_sub_images_of_mucd = []
                        if isinstance(sub_images, list):
                            for img in sub_images:
                                image_link = creating_image_link_for_mucd(img)
                                final_list_of_sub_images_of_mucd.append(image_link)
                        elif isinstance(sub_images, str):
                            image_link = creating_image_link_for_mucd(sub_images)
                            final_list_of_sub_images_of_mucd.append(image_link)

                        fixing_theatre_command_spell["sub_images_link"] = (
                            final_list_of_sub_images_of_mucd
                        )

                    result = fixing_theatre_command_spell
                    return jsonify(result)
            else:
                result = {"error": "number not available"}
                return jsonify(result)
        else:
            return jsonify({"error": "Invalid input"}), 404
    else:
        return redirect(url_for("authentication"))


def create_region_for_mucd(mucd_location):
    # List of provinces in China
    provinces = [
        "anhui",
        "beijing",
        "chongqing",
        "fujian",
        "gansu",
        "guangdong",
        "guangxi",
        "guizhou",
        "hainan",
        "hebei",
        "heilongjiang",
        "henan",
        "hubei",
        "hunan",
        "jiangsu",
        "jiangxi",
        "liaoning",
        "ningxia",
        "qinghai",
        "shaanxi",
        "shandong",
        "shanxi",
        "shanghai",
        "sichuan",
        "tianjin",
        "tibet",
        "xinjiang",
        "yunnan",
        "zhejiang",
        "hong_Kong",
        "macau",
        "taiwan",
    ]

    # Convert the location to lowercase for consistent comparison
    converting_mucd_value_in_lowercase = mucd_location.lower()

    # Split the location at the last comma
    parts = converting_mucd_value_in_lowercase.rsplit(",", 1)

    # Extract the region (the part after the last comma)
    if len(parts) > 1:
        region_extract = parts[1].strip()
        if " " in region_extract:
            region_extract = region_extract.split(" ")[0]
        elif "^" in region_extract:
            region_extract = region_extract.split("^")[0]
        else:
            region_extract = region_extract
    else:
        region_extract = parts[0].strip()

    # Check if the extracted region matches any province
    final_region = "NA"
    for province in provinces:
        if region_extract.lower() == province.lower():
            final_region = province
            break

    return final_region


def validate_and_sanitize_string_values_in_mucd(string_value):
    # Step 1: Strip leading and trailing whitespace
    sanitized_string = string_value.strip()

    # Step 2: Check if sanitized_string is a string and meets length requirements
    if (
        not isinstance(sanitized_string, str)
        or len(sanitized_string) < 2
        or len(sanitized_string) > 40
    ):
        return {"string_validated": "false", "string_value": string_value}

    # Step 3: Allowed characters in string value
    if not re.match("^[a-zA-Z0-9_ ]*$", sanitized_string):
        return {"string_validated": "false", "string_value": sanitized_string}

    # Step 4: Remove any non-alphanumeric characters except spaces and underscores
    sanitized_string = re.sub(r"[^a-zA-Z0-9_ ]", "", sanitized_string)

    # Step 5: Sanitize the string to prevent XSS (Cross-Site Scripting)
    sanitized_string = html.escape(sanitized_string)

    # Step 6: Preventing multiple XSS attacks
    sanitized_string = bleach.clean(sanitized_string)

    return {"string_validated": "true", "string_value": sanitized_string}


def validate_and_sanitize_mucd_fetch_arguments(input_dict):
    if isinstance(input_dict, dict):
        list_of_sanitized_strings = []
        for key, value in input_dict.items():
            dict_value = value
            sanitized_string = validate_and_sanitize_string_values_in_mucd(dict_value)
            if sanitized_string["string_validated"] == "true":
                input_dict[key] = sanitized_string["string_value"]
            list_of_sanitized_strings.append(sanitized_string)

        # Check if all dictionaries have 'string_validated' set to 'true'
        if all(
            item["string_validated"] == "true" for item in list_of_sanitized_strings
        ):
            return {"arguments_dict_validated": "true", "arguments_dict": input_dict}
        else:
            return {"arguments_dict_validated": "false", "arguments_dict": input_dict}
    else:
        return {"arguments_dict_validated": "false", "arguments_dict": input_dict}


@app.route("/mcdlstft", methods=["POST"])
@jwt_required()
def mucd_latest_entries_fetch():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        data = request.get_json()

        # Convert the query string to a dictionary
        if isinstance(data, str):
            arguments = parse_qs(data)

            # Convert lists to single values
            arguments = {k: v[0] for k, v in arguments.items()}
        else:
            arguments = {}

        arguments_validation = validate_and_sanitize_mucd_fetch_arguments(arguments)

        if arguments_validation["arguments_dict_validated"] == "true":

            # Set default values for rg and frtp
            thr = arguments.get("thr", "all")
            frtp = arguments.get("frtp", "all")
            rg = arguments.get("rg", "all")

            if thr == "all" and frtp == "all" and rg == "all":
                all_mucds = list(mucd_data.find({}, {"_id": 0}))

                # fixing date in mucd entries
                fixed_date_of_mucd = []
                for mucd_entry in all_mucds:
                    if "date" in mucd_entry:
                        mucd_entry["date"] = mucd_entry["date"]
                    else:
                        mucd_entry["date"] = "NA"

                    if '",' in mucd_entry["date"]:
                        mucd_entry["date"] = mucd_entry["date"].strip('",')
                    mucd_entry["date"] = mucd_entry["date"].split(" ")[0]

                    fixed_date_of_mucd.append(mucd_entry)

                # sorting mucd entry bases on date
                fixed_date_of_mucd.sort(key=get_date)
                sorted_recent_mucds = fixed_date_of_mucd

                final_list_of_mucd_item = []
                for mucd_item in sorted_recent_mucds:
                    mucd_number = mucd_item["mucd_number"]
                    mucd_location = mucd_item["location"]

                    # Check if latitude and longitude are None or blank
                    mucd_latitude = (
                        mucd_item["latitude"]
                        if mucd_item["latitude"] not in [None, ""]
                        else "NA"
                    )
                    mucd_longitude = (
                        mucd_item["longitude"]
                        if mucd_item["longitude"] not in [None, ""]
                        else "NA"
                    )

                    if "\t" in mucd_latitude:
                        mucd_latitude = mucd_latitude.strip().replace("\t", "")

                    if "\t" in mucd_longitude:
                        mucd_longitude = mucd_longitude.strip().replace("\t", "")

                    # creating region field if it's not there using location
                    if "region" not in mucd_item:
                        created_region = create_region_for_mucd(mucd_location)
                        mucd_item["region"] = created_region
                    else:
                        mucd_item["region"] = mucd_item["region"]

                    final_dict = {
                        "mucd": mucd_number,
                        "location": mucd_location,
                        "latitude": mucd_latitude,
                        "longitude": mucd_longitude,
                        "region": mucd_item["region"],
                    }
                    final_list_of_mucd_item.append(final_dict)
                return jsonify(final_list_of_mucd_item)
            else:
                return jsonify({"error": "Error in fetching latest tracking details"})
        else:
            return jsonify({"error": "Invalid input"}), 404
    else:
        return redirect(url_for("authentication"))


def fixing_theatre_command_search_query(search_string):
    theatre_command_code = {
        "sthrtcmd": "Southern Theatre Command",
        "nthrtcmd": "Northern Theatre Command",
        "wthrtcmd": "Western Theatre Command",
        "ethrtcmd": "Eastern Theatre Command",
        "cthrtcmd": "Central Theatre Command",
    }

    for key, value in theatre_command_code.items():
        if search_string in key:
            return value


@app.route("/mcdftch", methods=["POST"])
@jwt_required()
def fetch_mucd_data():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        data = request.get_json()

        # Convert the query string to a dictionary
        if isinstance(data, str):
            arguments = parse_qs(data)

            # Convert lists to single values
            arguments = {k: v[0] for k, v in arguments.items()}
        else:
            arguments = {}

        arguments_validation = validate_and_sanitize_mucd_fetch_arguments(arguments)

        if arguments_validation["arguments_dict_validated"] == "true":

            # Set default values for rg and frtp
            thr = arguments.get("thr", "all")
            frtp = arguments.get("frtp", "all")
            rg = arguments.get("rg", "all")

            def construct_final_list(mucd_items):
                final_list = []
                for mucd_item in mucd_items:
                    mucd_number = mucd_item["mucd_number"]
                    mucd_location = mucd_item["location"]

                    # Check if latitude and longitude are None or blank
                    mucd_latitude = (
                        mucd_item["latitude"]
                        if mucd_item["latitude"] not in [None, ""]
                        else "NA"
                    )
                    mucd_longitude = (
                        mucd_item["longitude"]
                        if mucd_item["longitude"] not in [None, ""]
                        else "NA"
                    )

                    if "\t" in mucd_latitude:
                        mucd_latitude = mucd_latitude.replace("\t", "")

                    if "\t" in mucd_longitude:
                        mucd_longitude = mucd_longitude.replace("\t", "")

                    # creating region field if it's not there using location
                    if "region" not in mucd_item:
                        created_region = create_region_for_mucd(mucd_location)
                        mucd_item["region"] = created_region
                    else:
                        mucd_item["region"] = mucd_item["region"]

                    final_dict = {
                        "mucd": mucd_number,
                        "location": mucd_location,
                        "latitude": mucd_latitude,
                        "longitude": mucd_longitude,
                        "region": mucd_item["region"],
                    }
                    final_list.append(final_dict)
                return final_list

            # Initialize the query dictionary
            query = {}

            # Build the query based on the values of thr, frtp, and rg
            if thr != "all":
                query["theatre_command"] = re.compile(
                    f"^{fixing_theatre_command_search_query(thr)}$", re.IGNORECASE
                )
            if frtp != "all":
                query["force_type"] = re.compile(f"^{frtp}$", re.IGNORECASE)
            if rg != "all":
                query["region"] = re.compile(f"^{rg}$", re.IGNORECASE)

            # If all values are 'all', query all documents
            if not query:  # This means thr, frtp, and rg are all 'all'
                all_mucds = list(mucd_data.find({}, {"_id": 0}))

                # fixing date in mucd entries
                fixed_date_of_mucd = []
                for mucd_entry in all_mucds:
                    if "date" in mucd_entry:
                        mucd_entry["date"] = mucd_entry["date"]
                    else:
                        mucd_entry["date"] = "NA"

                    if '",' in mucd_entry["date"]:
                        mucd_entry["date"] = mucd_entry["date"].strip('",')
                    mucd_entry["date"] = mucd_entry["date"].split(" ")[0]

                    fixed_date_of_mucd.append(mucd_entry)

                    # sorting mucd entry bases on date
                    fixed_date_of_mucd.sort(key=get_date)
                    sorted_recent_mucds = fixed_date_of_mucd
                final_list_of_mucds = sorted_recent_mucds
                final_list_of_mucd_item = construct_final_list(final_list_of_mucds)
                return jsonify(final_list_of_mucd_item)
            else:
                all_mucds = list(mucd_data.find(query, {"_id": 0}))

                if all_mucds:
                    # fixing date in mucd entries
                    fixed_date_of_mucd = []
                    for mucd_entry in all_mucds:
                        if "date" in mucd_entry:
                            mucd_entry["date"] = mucd_entry["date"]
                        else:
                            mucd_entry["date"] = "NA"

                        if '",' in mucd_entry["date"]:
                            mucd_entry["date"] = mucd_entry["date"].strip('",')
                        mucd_entry["date"] = mucd_entry["date"].split(" ")[0]

                        fixed_date_of_mucd.append(mucd_entry)

                        # sorting mucd entry bases on date
                        fixed_date_of_mucd.sort(key=get_date)
                        sorted_recent_mucds = fixed_date_of_mucd
                    final_list_of_mucds = sorted_recent_mucds

                    final_list_of_mucd_item = construct_final_list(final_list_of_mucds)
                    return jsonify(final_list_of_mucd_item)
                else:
                    return jsonify({"error": "No data found"})
        else:
            return jsonify({"error": "Invalid input"}), 404
    else:
        return redirect(url_for("authentication"))


# @app.route('/mcdrcnt', methods=['POST'])
# @jwt_required()
# def fetch_recent_mucd_data():
#     if 'email' not in session:
#         return redirect(url_for('authentication'))


#     # Access token is still active
#     current_user = get_jwt_identity()
#     user = users.find_one({"email": current_user}, {"_id": 0})

#     if user:
#         data = request.get_json()

#         # Convert the query string to a dictionary
#         if isinstance(data, str):
#             arguments = parse_qs(data)

#             # Convert lists to single values
#             arguments = {k: v[0] for k, v in arguments.items()}
#         else:
#             arguments = {}

#         arguments_validation = validate_and_sanitize_mucd_fetch_arguments(arguments)

#         if arguments_validation['arguments_dict_validated'] == 'true':

#             # Set default values for rg
#             rcnt = arguments.get('rcnt', 'all')

#             # Find all documents that contain the 'doc_uid' field
#             all_mucds = list(mucd_data.find({"doc_uid": {"$exists": True}}, {"_id": 0}))

#             if all_mucds:
#                 # fixing date in mucd entries
#                 fixed_date_of_mucd = []
#                 for mucd_entry in all_mucds:
#                     if "date" in mucd_entry:
#                         mucd_entry['date'] = mucd_entry['date']
#                     else:
#                         mucd_entry['date'] = "NA"

#                     if '",' in mucd_entry['date']:
#                         mucd_entry['date'] = mucd_entry['date'].strip('",')
#                     mucd_entry['date'] = mucd_entry['date'].split(" ")[0]

#                     fixed_date_of_mucd.append(mucd_entry)

#                 # sorting mucd entry bases on date
#                 fixed_date_of_mucd.sort(key=get_date)
#                 sorted_recent_mucds = fixed_date_of_mucd

#                 final_list_of_mucd_item = []
#                 for mucd_item in sorted_recent_mucds:
#                     mucd_number = mucd_item['mucd_number']
#                     mucd_location = mucd_item['location']

#                     # Check if latitude and longitude are None or blank
#                     mucd_latitude = mucd_item['latitude'] if mucd_item['latitude'] not in [None, ''] else 'NA'
#                     mucd_longitude = mucd_item['longitude'] if mucd_item['longitude'] not in [None, ''] else 'NA'

#                     if "\t" in mucd_latitude:
#                         mucd_latitude = mucd_latitude.replace('\t', '')

#                     if "\t" in mucd_longitude:
#                         mucd_longitude = mucd_longitude.replace('\t', '')


#                     # creating region field if it's not there using location
#                     if 'region' not in mucd_item:
#                         created_region = create_region_for_mucd(mucd_location)
#                         mucd_item['region'] = created_region
#                     else:
#                         mucd_item['region'] = mucd_item['region']

#                     final_dict = {"mucd":mucd_number, "location":mucd_location, "latitude":mucd_latitude, "longitude":mucd_longitude, "region":mucd_item['region']}
#                     final_list_of_mucd_item.append(final_dict)
#                 return jsonify(final_list_of_mucd_item)
#             else:
#                 return jsonify({"error": "something went wrong"})
#         else:
#             return jsonify({'error': 'Invalid input'}), 404
#     else:
#         return redirect(url_for('authentication'))


def extracting_latest_date_for_mucd(list_value):
    # extracting latest date only
    all_dates = []
    for data in list_value:
        if "date" in data:
            date = data["date"]
            all_dates.append(date)

    # Remove duplicates from date list by converting to a set
    final_date_list = list(set(all_dates))

    # sorting by date
    final_list_of_sorted_date = sorted(
        final_date_list, key=lambda x: parse_date(x), reverse=True
    )

    latest_date_value = final_list_of_sorted_date[0]
    return latest_date_value


@app.route("/mcdrcnt", methods=["POST"])
@jwt_required()
def fetch_recent_mucd_data():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        data = request.get_json()

        # Convert the query string to a dictionary
        if isinstance(data, str):
            arguments = parse_qs(data)

            # Convert lists to single values
            arguments = {k: v[0] for k, v in arguments.items()}
        else:
            arguments = {}

        arguments_validation = validate_and_sanitize_mucd_fetch_arguments(arguments)

        if arguments_validation["arguments_dict_validated"] == "true":

            # Set default values for rg
            rcnt = arguments.get("rcnt", "all")

            # Find all documents that contain the 'doc_uid' field
            # all_mucds = list(mucd_data.find({"doc_uid": {"$exists": True}}, {"_id": 0}))
            all_entries_of_mucd = list(mucd_data.find({}, {"_id": 0}))

            # extraction of latest date from mucd data
            latest_date_from_mucd_data = extracting_latest_date_for_mucd(
                all_entries_of_mucd
            )

            all_mucds = list(
                mucd_data.find({"date": latest_date_from_mucd_data}, {"_id": 0})
            )

            if all_mucds:
                # fixing date in mucd entries
                fixed_date_of_mucd = []
                for mucd_entry in all_mucds:
                    if "date" in mucd_entry:
                        mucd_entry["date"] = mucd_entry["date"]
                    else:
                        mucd_entry["date"] = "NA"

                    if '",' in mucd_entry["date"]:
                        mucd_entry["date"] = mucd_entry["date"].strip('",')
                    mucd_entry["date"] = mucd_entry["date"].split(" ")[0]

                    fixed_date_of_mucd.append(mucd_entry)

                # sorting mucd entry bases on date
                fixed_date_of_mucd.sort(key=get_date)
                sorted_recent_mucds = fixed_date_of_mucd

                final_list_of_mucd_item = []
                for mucd_item in sorted_recent_mucds:
                    mucd_number = mucd_item["mucd_number"]
                    mucd_location = mucd_item["location"]
                    mucd_date = mucd_item["date"]

                    # Check if latitude and longitude are None or blank
                    mucd_latitude = (
                        mucd_item["latitude"]
                        if mucd_item["latitude"] not in [None, ""]
                        else "NA"
                    )
                    mucd_longitude = (
                        mucd_item["longitude"]
                        if mucd_item["longitude"] not in [None, ""]
                        else "NA"
                    )

                    if "\t" in mucd_latitude:
                        mucd_latitude = mucd_latitude.replace("\t", "")

                    if "\t" in mucd_longitude:
                        mucd_longitude = mucd_longitude.replace("\t", "")

                    # creating region field if it's not there using location
                    if "region" not in mucd_item:
                        created_region = create_region_for_mucd(mucd_location)
                        mucd_item["region"] = created_region
                    else:
                        mucd_item["region"] = mucd_item["region"]

                    final_dict = {
                        "mucd": mucd_number,
                        "location": mucd_location,
                        "latitude": mucd_latitude,
                        "longitude": mucd_longitude,
                        "region": mucd_item["region"],
                        "date": mucd_date,
                    }
                    final_list_of_mucd_item.append(final_dict)
                return jsonify(final_list_of_mucd_item)
            else:
                return jsonify({"error": "something went wrong"})
        else:
            return jsonify({"error": "Invalid input"}), 404
    else:
        return redirect(url_for("authentication"))


@app.route("/mcdunnmsug", methods=["GET"])
@jwt_required()
def unit_number_suggestions_for_mucd():
    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        query = request.args.get("query", "")
        if query:
            suggestions = mucd_data.find(
                {"mucd_number": {"$regex": query, "$options": "i"}},
                {"_id": 0, "mucd_number": 1},
            )

            # Convert cursor to list
            suggestions_list = [suggestion["mucd_number"] for suggestion in suggestions]
            convert_to_tuple = set(suggestions_list)
            final_suggestions_list = list(convert_to_tuple)
            return jsonify(final_suggestions_list)
        return jsonify([])
    else:
        return "Invalid user", 404


def creating_final_dict_of_mucd_history(data_dict):
    final_dict = {}

    final_dict["mucd_number"] = data_dict["mucd_number"]

    # date management in entry
    if "date" in data_dict:
        if data_dict["date"] != "" or data_dict["date"] != "NA":
            final_dict["date"] = data_dict["date"]
        else:
            final_dict["date"] = "Not available"
    else:
        final_dict["date"] = "Not available"

    # location management in entry
    if "location" in data_dict:
        if data_dict["location"] != "" or data_dict["location"] != "NA":
            final_dict["location"] = data_dict["location"]
        else:
            final_dict["location"] = "Not available"
    else:
        final_dict["location"] = "Not available"

    # latitude management in entry
    if "latitude" in data_dict:
        if data_dict["latitude"] != "" or data_dict["latitude"] != "NA":
            final_dict["latitude"] = data_dict["latitude"]
        elif (
            data_dict["latitude"] == ""
            or data_dict["latitude"] == "NA"
            or data_dict["latitude"] == "na"
        ):
            final_dict["latitude"] = "Not available"
        else:
            final_dict["latitude"] = "Not available"
    else:
        final_dict["latitude"] = "Not available"

    # longitude management in entry
    if "longitude" in data_dict:
        if data_dict["longitude"] != "" or data_dict["longitude"] != "NA":
            final_dict["longitude"] = data_dict["longitude"]
        elif (
            data_dict["longitude"] == ""
            or data_dict["longitude"] == "NA"
            or data_dict["longitude"] == "na"
        ):
            final_dict["longitude"] = "Not available"
        else:
            final_dict["longitude"] = "Not available"
    else:
        final_dict["longitude"] = "Not available"

    # region management in entry
    if "region" in data_dict:
        if data_dict["region"] != "" or data_dict["region"] != "NA":
            final_dict["region"] = data_dict["region"]
        else:
            final_dict["region"] = "Not available"
    else:
        final_dict["region"] = "Not available"

    # doc_name management in entry
    if "doc_name" in data_dict:
        if data_dict["doc_name"] != "" or data_dict["doc_name"] != "NA":
            final_dict["doc_name"] = data_dict["doc_name"]
        else:
            final_dict["doc_name"] = "Not available"
    else:
        final_dict["doc_name"] = "Not available"

    # doc_uid management in entry
    if "doc_uid" in data_dict:
        if data_dict["doc_uid"] != "" or data_dict["doc_uid"] != "NA":
            final_dict["doc_uid"] = data_dict["doc_uid"]
        else:
            final_dict["doc_uid"] = "Not available"
    else:
        final_dict["doc_uid"] = "Not available"

    # tracking status management in entry
    if "tracked_status" in data_dict:
        final_dict["tracked_status"] = data_dict["tracked_status"]

    return final_dict


@app.route("/mcdldhis", methods=["POST"])
@jwt_required()
def mucd_number_old_history():
    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        data = request.get_json()

        # Convert the query string to a dictionary
        if isinstance(data, str):
            arguments = parse_qs(data)

            # Convert lists to single values
            arguments = {k: v[0] for k, v in arguments.items()}
        else:
            arguments = {}

        arguments_validation = validate_and_sanitize_mucd_fetch_arguments(arguments)

        if arguments_validation["arguments_dict_validated"] == "true":
            mucd_code = arguments.get("mcdnm", "")
            latest_mucd_entry = list(
                mucd_data.find({"mucd_number": str(mucd_code)}, {"_id": 0}).sort(
                    "date", DESCENDING
                )
            )

            if latest_mucd_entry:
                final_list_of_all_history = []

                # adding tracked status in mucd entry
                for index, data in enumerate(latest_mucd_entry):
                    if index == 0:
                        data["tracked_status"] = "recent"
                    else:
                        data["tracked_status"] = "previous"

                for mucd in latest_mucd_entry:
                    fixed_final_dict = creating_final_dict_of_mucd_history(mucd)
                    final_list_of_all_history.append(fixed_final_dict)
                return jsonify(final_list_of_all_history)
            else:
                return jsonify([])
        else:
            return jsonify({"error": "mucd number is not valid"})
        return jsonify({"error": "something went wrong"})
    else:
        return jsonify({"error": "invalid user"})


@app.route("/profiler", methods=["GET"])
def profiler():
    if "email" not in session:
        return redirect(url_for("authentication"))
    return render_template("profiler.html")


# image dir of commander profile
weapon_image_dir = "weapon_images"


@app.route("/wpnres/<image_name>")
def serve_weapon_images(image_name):
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Validate the image name to prevent directory traversal
    if not re.match(
        r"^[\w\-. ]+$", image_name
    ):  # Allow only alphanumeric, dash, underscore, dot, and space
        return redirect(url_for("no_image_available"))

    # Serve the image from the directory
    try:
        return send_from_directory(weapon_image_dir, image_name)
    except FileNotFoundError:
        return redirect(url_for("no_image_available"))


# functions to create image link for weapon
def creating_image_link_for_weapon(dictionary):
    if isinstance(dictionary["image"], str):
        image_name_value = dictionary["image"]
    elif isinstance(dictionary["image"], list):
        image_name_value = dictionary["image"][0]

    image_path = os.path.join("wpnres", image_name_value)

    # Check if the image exists in the image directory
    check_data = os.path.exists(os.path.join(weapon_image_dir, image_name_value))

    if check_data:
        domain_name = request.url_root
        complete_image_link = "{}{}".format(domain_name, image_path)
        return complete_image_link
    else:
        complete_image_link = "NA"
        return complete_image_link


# Function to check if the input might represent a weapon name
def is_weapon_name_or_vehicle_name(input_string):
    # This regex allows for alphanumeric characters, spaces, and hyphens
    pattern = r"^[A-Za-z0-9\s\-\/\(\)]+$"
    return re.match(pattern, input_string) is not None


# Function to search for vehicle names and weapon names
def search_vehicle_name_or_weapon_name(query_value):
    final_list = []
    # checking in vehicles data
    # vehicle_result = list(vehicles_data.find({"name": {"$regex": query_value, "$options": "i"}}))  # Case-insensitive search

    # checking in weapons data
    weapon_results = list(
        weapons_data.find(
            {
                "$or": [
                    {"name": {"$regex": query_value, "$options": "i"}},
                    {
                        "variants": {
                            "$elemMatch": {
                                "variant_name": {"$regex": query_value, "$options": "i"}
                            }
                        }
                    },
                ]
            },
            {"_id": 0},
        )
    )

    if len(weapon_results) != 0:
        # New list to store the filtered results
        filtered_weapons = []

        # Iterate through the fetched results
        for weapon in weapon_results:
            # Check if the name matches
            if re.search(query_value, weapon["name"], re.IGNORECASE):
                creating_image_link = creating_image_link_for_weapon(weapon)
                weapon["image_link"] = creating_image_link
                # If matched by name, add the whole document to the filtered list
                filtered_weapons.append(weapon)

            # Check if variants is a list and not empty, or if it's not 'NA' and not an empty string
            if isinstance(weapon["variants"], list) and weapon["variants"]:
                # Check for matches in the variants
                for variant in weapon["variants"]:
                    if re.search(query_value, variant["variant_name"], re.IGNORECASE):
                        creating_image_link = creating_image_link_for_weapon(variant)
                        # Create a new dict with only the matched variant
                        filtered_variant = {
                            "name": variant["variant_name"],
                            "specifications": variant["specifications"],
                            "image": variant["image"],
                            "weapon_uid": weapon["weapon_uid"],
                            "image_link": creating_image_link,
                            "research_links": variant.get("research_links", []),
                            "related_articles": variant.get("related_articles", []),
                        }
                        filtered_weapons.append(filtered_variant)

        for final_data in filtered_weapons:
            final_list.append(final_data)
    return final_list


# @app.route('/danm', methods=['POST'])
# def check_da_number():
#     if 'email' not in session:
#         return redirect(url_for('authentication'))


#     if request.method == 'POST':
#         da_query = request.form['da_vlve']

#         if is_license_plate(da_query):
#             fixed_code = da_query.upper()
#             vehicle_details = get_details_from_code(fixed_code)

#             if vehicle_details != 'Invalid event or theatre_commond character.':
#                 vehicle_details['found_in'] = 'vhc_lg'
#                 vehicle_details['lic_nm'] = fixed_code
#                 return jsonify(vehicle_details)
#             else:
#                 return jsonify({"error": "number not available"})
#         elif is_weapon_name_or_vehicle_name(da_query):
#             checking_data_in_vehicle_and_weapon = search_vehicle_name_or_weapon_name(da_query)

#             if len(checking_data_in_vehicle_and_weapon) != 0:
#                 return jsonify(checking_data_in_vehicle_and_weapon)
#             else:
#                 return jsonify({"error":"weapon or vehicle not found"})
#         return jsonify({"error":"Not available"})


@app.route("/wpnprfdn", methods=["GET"])
def fetching_some_weapon_for_predefined_box():
    if "email" not in session:
        return redirect(url_for("authentication"))

    #    all_weapons = list(weapons_data.find({}, {"_id": 0}).sort("_id", -1).limit(12))
    all_weapons = list(
        weapons_data.aggregate([{"$sample": {"size": 12}}, {"$project": {"_id": 0}}])
    )

    if all_weapons:

        final_list = []
        for weapon in all_weapons:
            creating_image_link = creating_image_link_for_weapon(weapon)
            weapon["image_link"] = creating_image_link
            final_list.append(weapon)
        return jsonify(final_list)
    return jsonify({"error": "something went wrong"}), 400


@app.route("/danm", methods=["POST"])
@jwt_required()
def check_da_number():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        da_query = request.form["da_vlve"]

        if is_weapon_name_or_vehicle_name(da_query):
            weapon_result = search_vehicle_name_or_weapon_name(da_query)

            if len(weapon_result) != 0:
                weapon = weapon_result[0]
                return jsonify(weapon)
        else:
            return jsonify({"error": "weapon or vehicle not found"})
        return jsonify({"error": "Not available"})
    else:
        return jsonify({"error": "Invalid input"}), 404


@app.route("/wpnsg", methods=["GET"])
@jwt_required()
def defence_assets_name_suggestions():

    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        query = request.args.get("query", "")
        if query:
            suggestions = search_vehicle_name_or_weapon_name(query)

            final_list_of_defence_assets = []
            for defence_asset in suggestions:
                final_dict_of_defence_asset = {}

                final_dict_of_defence_asset["weapon_name"] = defence_asset[
                    "name"
                ].strip()
                final_dict_of_defence_asset["image_link"] = defence_asset[
                    "image_link"
                ].strip()
                final_list_of_defence_assets.append(final_dict_of_defence_asset)
            return jsonify(final_list_of_defence_assets)
        return jsonify([])
    else:
        return "Invalid user", 404


# image dir of commander profile
commander_image_dir = "commanders_images"


@app.route("/cmdres/<image_name>")
def serve_commander_images(image_name):
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Validate the image name to prevent directory traversal
    if not re.match(
        r"^[\w\-. ]+$", image_name
    ):  # Allow only alphanumeric, dash, underscore, dot, and space
        return redirect(url_for("no_image_available"))

    # Serve the image from the directory
    try:
        return send_from_directory(commander_image_dir, image_name)
    except FileNotFoundError:
        return redirect(url_for("no_image_available"))


# functions to create image link for commander profile
def creating_image_link_for_commander(dictionary):
    image_name_value = dictionary["image"]
    image_path = os.path.join("cmdres", image_name_value)

    # Check if the image exists in the image directory
    check_data = os.path.exists(os.path.join(commander_image_dir, image_name_value))

    if check_data:
        domain_name = request.url_root
        complete_image_link = "{}{}".format(domain_name, image_path)
        return complete_image_link
    else:
        complete_image_link = "NA"
        return complete_image_link


@app.route("/cmd_pro", methods=["GET"])
def commander_profile():
    if "email" not in session:
        return redirect(url_for("authentication"))

    return render_template("commander_profile.html")


# @app.route('/cmdprfdn', methods=['GET'])
# def fetching_some_commanders_profile_for_predefined_box():
#     if 'email' not in session:
#         return redirect(url_for('authentication'))


#     all_commanders_profile = list(commanders_data.find({}, {"_id": 0}).sort("_id", -1).limit(9))

#     if all_commanders_profile:

#         final_list = []
#         for commander in all_commanders_profile:
#             creating_image_link = creating_image_link_for_commander(commander)
#             commander['image_link'] = creating_image_link
#             final_list.append(commander)
#         return jsonify(final_list)
#     return jsonify({"error": "something went wrong"})


@app.route("/cmdprfdn", methods=["GET"])
def fetching_some_commanders_profile_for_predefined_box():
    if "email" not in session:
        return redirect(url_for("authentication"))

    all_commanders_profile = list(
        commanders_data.aggregate([{"$sample": {"size": 9}}, {"$project": {"_id": 0}}])
    )

    if all_commanders_profile:
        final_list = []
        for commander in all_commanders_profile:
            creating_image_link = creating_image_link_for_commander(commander)
            commander["image_link"] = creating_image_link
            final_list.append(commander)
        return jsonify(final_list)
    return jsonify({"error": "something went wrong"})


# Remove duplicates by (eng_name, native_name) key combination before selecting random samples
def get_unique_profiles(profiles):
    seen = set()
    unique_profiles = []
    for profile in profiles:
        # Create a unique key based on relevant fields to identify duplicates
        key = (
            profile.get("eng_name").strip().lower(),
            profile.get("native_name").strip().lower(),
        )
        if key not in seen:
            seen.add(key)
            unique_profiles.append(profile)
    return unique_profiles


def remove_non_alpha_characters(input_string: str) -> str:
    # Use regular expression to replace non-alphabetic characters
    cleaned_string = re.sub(r"[^a-zA-Z ]", "", input_string)
    return cleaned_string


def creating_feeds_data_for_commander_profile(query):
    splitting_query_keywords = query.split("|")

    final_list_of_keywords = []
    for keyword in splitting_query_keywords:
        fixed_string = remove_non_alpha_characters(keyword)
        final_list_of_keywords.append(fixed_string.strip().lower())

    final_query = " ".join(final_list_of_keywords)

    results = []
    query_vector = embeddings.embed_query(final_query)

    # Assuming final_list_of_keywords is already defined and populated
    commander_name = final_list_of_keywords[0]
    commander_designation = final_list_of_keywords[1]

    # Construct the filter for the search
    filter_condition = {
        "must": [{"key": "page_content", "match": {"value": commander_name}}]
    }

    # # Construct the filter for the search
    # filter_condition = {
    #     "must": [
    #         {
    #             "key": "page_content",
    #             "match": {
    #                 "value": commander_name
    #             }
    #         },
    #         {
    #             "key": "page_content",
    #             "match": {
    #                 "value": commander_designation
    #             }
    #         }
    #     ]
    # }

    # searching keyword in feeds data
    search_result = qdrant_client.search(
        collection_name="qdrant_feeds_data",
        query_vector=query_vector,
        limit=50,
        with_payload=True,
        score_threshold=0.7,
        query_filter=filter_condition,
    )

    feeds_results = []
    unique_ids = set()  # Set to track unique article numbers
    for result in search_result:
        payload = result.payload
        article_data = payload

        # Check if the article number is already in the set
        if article_data["article_number"] not in unique_ids:
            unique_ids.add(article_data["article_number"])  # Add to the set
            feeds_results.append(article_data)
    fixed_list_of_feeds = fixing_feeds(feeds_results)

    # Sort the list based on the 'date' key
    sorted_data = sorted(
        fixed_list_of_feeds, key=lambda x: parse_date(x["date"]), reverse=True
    )

    # adding new key in feeds
    for feed in sorted_data:
        feed["article_link"] = "/fdarticle/{}".format(feed["article_number"])
        results.append(feed)

    return results


def check_news_regarding_commander_profile(commander_profile):
    commander_data = commander_profile

    # extracting required values for news searching
    commander_name = commander_data["eng_name"]
    latest_entry_of_career_details = commander_data["career_details"][0]
    designation = latest_entry_of_career_details["role"]
    department = latest_entry_of_career_details["department"]

    # sending query to qdrant db for searching

    # Convert the query to an embedding
    query = f"{commander_name.strip()} | {designation.strip()} | {department.strip()}"

    # searching query in feeds data
    feeds_search_results = creating_feeds_data_for_commander_profile(query)

    final_list_of_articles = []
    for feed in feeds_search_results:
        final_dict = {}
        final_dict["title"] = feed["title"]
        final_dict["date"] = feed["date"]
        final_dict["article_number"] = feed["article_number"]
        final_dict["article_link"] = feed["article_link"]
        final_list_of_articles.append(final_dict)

    # sorting the feeds in descending order
    final_list_of_articles.sort(key=get_date)
    sorted_feeds = final_list_of_articles

    return sorted_feeds


@app.route("/cmd_profth/<commander_profile_id>", methods=["GET"])
def fetch_commander_profile_on_new_tab(commander_profile_id):
    if "email" not in session:
        return redirect(url_for("authentication"))

    fixing_profile_ID = check_alphanumeric_with_hyphen(commander_profile_id)

    if fixing_profile_ID["isvalid"] == "true":
        fixed_profileID = fixing_profile_ID["commander_profile_id_value"]
        checking_commander_profileID_in_db = commanders_data.find_one(
            {"person_uid": fixed_profileID}, {"_id": 0}
        )

        # converting every dict key in lowercase of career details list
        career_details_list = checking_commander_profileID_in_db["career_details"]

        if isinstance(career_details_list, list):
            career_details_lowercase_dict_list = []
            for item in career_details_list:
                converting_dict_key_in_lowercase = lowercase_keys(item)
                career_details_lowercase_dict_list.append(
                    converting_dict_key_in_lowercase
                )

        # Sort the career details in descending order based on the start year
        sorted_career_details = sorted(
            career_details_lowercase_dict_list, key=extract_years, reverse=True
        )
        checking_commander_profileID_in_db["career_details"] = sorted_career_details

        # Extract departments with "current" or "present" in the years
        current_departments = []
        for i, item in enumerate(sorted_career_details):
            # if 'current' in item['years'].lower() or 'present' in item['years'].lower():
            if "years" in item and (
                "current" in item["years"].lower() or "present" in item["years"].lower()
            ):
                # Check if the department is blank
                department_value = (
                    item["department"].strip() if item["department"].strip() else None
                )

                # If department is blank, take the value from the next dict if available
                if not department_value and i + 1 < len(sorted_career_details):
                    department_value = sorted_career_details[i + 1]["department"]

                # Append the department value to the list
                current_departments.append(department_value)

        # Add the current departments list to the commander profile data
        checking_commander_profileID_in_db["current_departments"] = current_departments

        # adding image link key with value in dict
        checking_commander_profileID_in_db["image_link"] = (
            creating_image_link_for_commander(checking_commander_profileID_in_db)
        )

        # extracting commander name for new tab title
        commander_name_in_english_for_title = checking_commander_profileID_in_db[
            "eng_name"
        ]

        # creating final list with single dict
        final_list_of_commander = [checking_commander_profileID_in_db]

        # checking news which include this commander profile
        commander_in_news = check_news_regarding_commander_profile(
            final_list_of_commander[0]
        )

        # merging commander profile suggestions in a single list
        merging_all_suggestion_in_a_single_list = []
        for department_based_search in current_departments:
            query = department_based_search
            if query:
                commander_profile_suggestions = list(
                    commanders_data.find(
                        {
                            "$or": [
                                {"occupation": {"$regex": query, "$options": "i"}},
                                {
                                    "career_details.department": {
                                        "$regex": query,
                                        "$options": "i",
                                    }
                                },
                            ]
                        },
                        {
                            "_id": 0,
                        },
                    )
                )
                merging_all_suggestion_in_a_single_list.extend(
                    commander_profile_suggestions
                )

        # creating a commanders profile suggestion final list
        final_list_of_commanders_profile_suggestions = []
        for commander_entry in merging_all_suggestion_in_a_single_list:
            final_dict_of_commander_profile_suggestion = {}
            commander_name_in_english = commander_entry["eng_name"]
            commander_name_in_chinese = commander_entry["native_name"]
            commander_designation = commander_entry["occupation"]
            commander_profile_id = commander_entry["person_uid"]
            commander_image_link = creating_image_link_for_commander(commander_entry)

            # updating new dict
            final_dict_of_commander_profile_suggestion["eng_name"] = (
                commander_name_in_english.strip()
            )
            final_dict_of_commander_profile_suggestion["native_name"] = (
                commander_name_in_chinese.strip()
            )
            final_dict_of_commander_profile_suggestion["occupation"] = (
                commander_designation.strip()
            )
            final_dict_of_commander_profile_suggestion["person_uid"] = (
                commander_profile_id.strip()
            )
            final_dict_of_commander_profile_suggestion["image_link"] = (
                commander_image_link.strip()
            )
            final_list_of_commanders_profile_suggestions.append(
                final_dict_of_commander_profile_suggestion
            )

        # creating list of 3 commanders profile suggestion in random order
        # removing duplicate entries from the list
        unique_commanders = get_unique_profiles(
            final_list_of_commanders_profile_suggestions
        )

        # Select 3 unique random profiles
        num_to_select = 3
        if len(unique_commanders) >= num_to_select:
            random_commanders_profile = random.sample(unique_commanders, num_to_select)
        else:
            random_commanders_profile = unique_commanders  # Return all if less than 3

        return render_template(
            "commander_full_profile.html",
            commander_full_profile_data=final_list_of_commander,
            commander_name=commander_name_in_english_for_title,
            profile_suggestions=random_commanders_profile,
            commander_profile_news_details=commander_in_news,
        )
    else:
        return "", 404


@app.route("/cmd_check", methods=["POST"])
@jwt_required()
def commander_profile_check():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        cmd_name = request.form["cmd_name"]
        fixing_commander_name = fix_commander_name(cmd_name)

        if fixing_commander_name["isvalid"] == "true":
            commander_name = fixing_commander_name["name_value"]

            # Determine if the name is in English or native language
            if re.match(r"^[A-Za-z\s]+$", commander_name):  # English name check
                query_field = "eng_name"
            else:  # Assume it's in native language
                query_field = "native_name"

            checking_commander_name_in_db = list(
                commanders_data.find(
                    {
                        query_field: re.compile(
                            f"^{re.escape(commander_name)}$", re.IGNORECASE
                        )
                    },
                    {"_id": 0},
                )
            )
            if checking_commander_name_in_db:

                # creating a final list of commanders found
                final_list = []
                for commander in checking_commander_name_in_db:
                    final_dict = {}
                    final_dict["engnme"] = commander["eng_name"]
                    final_dict["ntvnme"] = commander["native_name"]
                    final_dict["occp"] = commander["occupation"]
                    final_dict["cpid"] = commander["person_uid"]
                    creating_image_link = creating_image_link_for_commander(commander)
                    final_dict["image_link"] = creating_image_link
                    final_list.append(final_dict)
                return jsonify(final_list)
            return jsonify({"error": "profile not found"}), 201
        else:
            return jsonify({"error": "Invalid input"}), 404
    else:
        return redirect(url_for("authentication")), 302


# Function to extract the start year from the "Years" or "years" field of career details in the commander's profile
def extract_years(career):
    # Attempt to get the "Years" field, checking both cases
    years = career.get("Years") or career.get("years", "")

    # Split the years string by the '-' character
    year_parts = years.split("-")

    # Extract the start year, handling cases where the format may vary
    if year_parts:
        start_year_str = year_parts[
            0
        ].strip()  # Get the first part and strip whitespace

        # Check if the start year is a valid year or contains "Present"
        if start_year_str.isdigit():
            return start_year_str  # Return as integer if it's a valid year

        # Check if the start year contains alphanumeric characters using regex
        elif re.search(r"\w", start_year_str):

            # Extract numeric values from start_year_str
            numeric_values = re.findall(r"\d+", start_year_str)

            if numeric_values:
                return numeric_values[0]  # Return the first numeric value found
            else:
                return ""  # Return empty string if no numeric values found
        else:
            return ""
    return ""


@app.route("/cmd_prof", methods=["POST"])
@jwt_required()
def commander_profile_single_check():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        data = request.get_json()
        commander_profile_ID = data.get("cmd_sngl_nme")
        fixing_profile_ID = check_alphanumeric_with_hyphen(commander_profile_ID)

        if fixing_profile_ID["isvalid"] == "true":
            fixed_profileID = fixing_profile_ID["commander_profile_id_value"]
            checking_commander_profileID_in_db = commanders_data.find_one(
                {"person_uid": fixed_profileID}, {"_id": 0}
            )

            # converting every dict key in lowercase of career details list
            career_details_list = checking_commander_profileID_in_db["career_details"]

            if isinstance(career_details_list, list):
                career_details_lowercase_dict_list = []
                for item in career_details_list:
                    converting_dict_key_in_lowercase = lowercase_keys(item)
                    career_details_lowercase_dict_list.append(
                        converting_dict_key_in_lowercase
                    )

            # Sort the career details in descending order based on the start year
            sorted_career_details = sorted(
                career_details_lowercase_dict_list, key=extract_years, reverse=True
            )
            checking_commander_profileID_in_db["career_details"] = sorted_career_details

            checking_commander_profileID_in_db["image_link"] = (
                creating_image_link_for_commander(checking_commander_profileID_in_db)
            )
            final_result = checking_commander_profileID_in_db
            return jsonify(final_result)
        else:
            return jsonify({"error": "profile not found"})
    else:
        return jsonify({"error": "something went wrong"})


@app.route("/cmdprosg", methods=["GET"])
@jwt_required()
def commander_name_suggestions():

    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        query = request.args.get("query", "")
        if query:
            suggestions = list(
                commanders_data.find(
                    {
                        "$or": [
                            {"eng_name": {"$regex": query, "$options": "i"}},
                            {"native_name": {"$regex": query, "$options": "i"}},
                            {"occupation": {"$regex": query, "$options": "i"}},
                            {
                                "personal_details.birth_place": {
                                    "$regex": query,
                                    "$options": "i",
                                }
                            },
                            {
                                "career_details.location": {
                                    "$regex": query,
                                    "$options": "i",
                                }
                            },
                            {
                                "career_details.department": {
                                    "$regex": query,
                                    "$options": "i",
                                }
                            },
                        ]
                    },
                    {
                        "_id": 0,
                    },
                )
            )

            final_list_of_commanders = []
            for commander_entry in suggestions:
                final_dict_of_commander = {}
                commander_name_in_english = commander_entry["eng_name"]
                commander_name_in_chinese = commander_entry["native_name"]
                commander_designation = commander_entry["occupation"]
                commander_image_link = creating_image_link_for_commander(
                    commander_entry
                )

                # updating new dict
                final_dict_of_commander["eng_name"] = commander_name_in_english.strip()
                final_dict_of_commander["native_name"] = (
                    commander_name_in_chinese.strip()
                )
                final_dict_of_commander["occupation"] = commander_designation.strip()
                final_dict_of_commander["image_link"] = commander_image_link.strip()
                final_list_of_commanders.append(final_dict_of_commander)
            return jsonify(final_list_of_commanders)
        return jsonify([])
    else:
        return "Invalid user", 404


@app.route("/gen_report", methods=["GET"])
def gen_report():
    if "email" not in session:
        return redirect(url_for("authentication"))

    email = session["email"]

    all_generated_reports = list(
        generate_report_data.find({}, {"_id": 0}).sort("report_date", -1)
    )
    final_list_of_all_reports = []
    for report_link in all_generated_reports:
        report_hash = report_link["report_hash"]
        report_link["report_link"] = "/gnrpt/{}".format(report_hash)
        report_link["report_download_link"] = "/gnrptdwn/{}".format(report_hash)
        final_list_of_all_reports.append(report_link)

    return render_template("generate_report.html", gn_rpt=final_list_of_all_reports)


@app.route("/shwgnrpt", methods=["POST"])
@jwt_required()
def show_generated_report():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        email = user["email"]
        all_user_generated_reports = list(
            user_report_data.find({"user_email": email}, {"_id": 0}).sort(
                "report_date", -1
            )
        )
        final_list_of_all_user_reports = []
        for user_report_link in all_user_generated_reports:
            report_hash = user_report_link["report_hash"]
            user_report_link["report_link"] = "/usrgnrpt/{}".format(report_hash)
            user_report_link["report_download_link"] = "/usrgnrptdwn/{}".format(
                report_hash
            )
            final_list_of_all_user_reports.append(user_report_link)
        return jsonify(final_list_of_all_user_reports)
    else:
        return redirect(url_for("authentication"))


@app.route("/create_report", methods=["POST"])
def create_report():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Retrieve form data
    report_title = request.form.get("report_title")
    report_keyword = request.form.get("report_keyword")
    start_date = request.form.get("date_from")
    end_date = request.form.get("date_to")
    time_frame = [start_date, end_date]

    # Basic validation
    if not report_title or not report_keyword or not start_date or not end_date:
        return jsonify({"error": "All fields are required!"})

    content, sources = get_relevant_content(report_keyword, time_frame=time_frame)

    if not content:
        return jsonify({"warning": "No relevant data found for the keyword!"})
    else:
        structured_data = generate_report(
            report_keyword, content, sources, time_frame, report_title
        )

        email = session["email"]

        # Add metadata
        report_hash = str(uuid.uuid4())
        structured_data.update(
            {
                "report_hash": report_hash,
                "report_date": structured_data["generated_on"].split(" ")[0],
                "user_email": email,
            }
        )
        save_report_pdf(structured_data, f"{report_hash}.pdf")

        # Prepare user report data for user_data & user_report_data
        final_dict = {
            "report_title": report_title,
            "report_date": structured_data["report_date"],
            "report_hash": structured_data["report_hash"],
        }

        # Store full report data (only if hash doesn't exist)
        existing_doc = user_report_data.find_one({"report_hash": report_hash})

        if not existing_doc:
            user_report_data.insert_one(structured_data)

        # Update the user's document in MongoDB
        user_document = users_data.find_one({"email": email})
        if user_document:
            # Check if 'user_reports' key exists, if not, create it
            if "user_reports" not in user_document:
                users_data.update_one({"email": email}, {"$set": {"user_reports": []}})
            # Append the final_dict to the user_reports array
            users_data.update_one(
                {"email": email}, {"$push": {"user_reports": final_dict}}
            )

        return jsonify({"success": f"Report generated successfully!"})


# @app.route("/spc_report", methods=['GET'])
# def special_report():
#     if 'email' not in session:
#         return redirect(url_for('authentication'))

#     all_special_reports = list(special_report_data.find({}, {'_id': 0}).sort("report_date", -1));
#     final_list_of_all_reports = []
#     for report_link in all_special_reports:
#         report_hash = report_link['report_hash']
#         report_link['report_link'] = "/spcrpt/{}".format(report_hash)
#         report_link['report_download_link'] = "/spcrptdwn/{}".format(report_hash)
#         final_list_of_all_reports.append(report_link)
#     return render_template('special_report.html', spc_rpt=final_list_of_all_reports)


# Set the directory where your reports are stored
sat_report_dir = "satcom_reports"


# view sat report
@app.route("/satrpt/<report_number>")
def sat_report_view(report_number):
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Validate the report_number to only allow valid characters (e.g., alphanumeric, hyphen, underscore)
    if not re.match(r"^[a-zA-Z0-9_-]+$", report_number):
        return jsonify({"error": "Invalid report number"}), 400

    filename = report_number + ".pdf"

    return send_from_directory(sat_report_dir, filename)


# download sat report
@app.route("/satrptdwn/<report_number>", methods=["GET"])
def sat_report_download(report_number):
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Validate the report_number to only allow valid characters (e.g., alphanumeric, hyphen, underscore)
    if not re.match(r"^[a-zA-Z0-9_-]+$", report_number):
        return jsonify({"error": "Invalid report number"}), 400

    filename = report_number + ".pdf"

    if report_number != "":
        report_file_name = filename
        return send_from_directory(sat_report_dir, report_file_name, as_attachment=True)


@app.route("/satcom", methods=["GET"])
def satellite_communication():
    if "email" not in session:
        return redirect(url_for("authentication"))

    all_satcom_reports = list(satcom_data.find({}, {"_id": 0}).sort("report_date", -1))
    final_list_of_all_reports = []
    for report_link in all_satcom_reports:
        report_hash = report_link["report_hash"]
        report_link["report_link"] = "/satrpt/{}".format(report_hash)
        report_link["report_download_link"] = "/satrptdwn/{}".format(report_hash)
        final_list_of_all_reports.append(report_link)
    return render_template("/satcom.html", satrpt=final_list_of_all_reports)


@app.route("/satsdata", methods=["GET"])
def satellite_data():
    if "email" not in session:
        return redirect(url_for("authentication"))

    return render_template("satsdata.html")


@app.route("/allbsent", methods=["GET"])
@jwt_required()
def all_bases_entries():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        all_indian_bases = list(indian_bases_data.find({}, {"_id": 0}))

        # excluding bases which don't have lat long details
        final_list_of_all_indian_bases = []
        for base in all_indian_bases:
            if (
                base["latitude"] != "not available"
                or base["longitude"] != "not available"
            ):
                final_list_of_all_indian_bases.append(base)

        return jsonify(final_list_of_all_indian_bases)
    else:
        return "Invalid user", 404


# image dir of indian bases
indian_base_image_dir = "indian_bases_image"


@app.route("/bseimg/<image_name>")
def serve_indian_base_image(image_name):
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Validate the image name to prevent directory traversal
    if not re.match(
        r"^[\w\-. ]+$", image_name
    ):  # Allow only alphanumeric, dash, underscore, dot, and space
        return redirect(url_for("no_image_available"))

    # Search for the image in all subdirectories
    for root, dirs, files in os.walk(indian_base_image_dir):
        if image_name in files:
            # If the image is found, return it using send_from_directory
            return send_from_directory(root, image_name)
    return redirect(url_for("no_image_available"))


def creating_image_link_for_indian_base(image_name_value):
    image_name = image_name_value
    image_path = os.path.join("bseimg", image_name_value)

    # Search for the image in all subdirectories
    for root, dirs, files in os.walk(indian_base_image_dir):
        if image_name in files:
            domain_name = request.url_root
            complete_image_link = "{}{}".format(domain_name, image_path)
            return complete_image_link

    # If the image is not found, return "NA"
    return "NA"


def validate_and_sanitize_base_name_data(string_value):
    # Step 1: Strip leading and trailing whitespace
    sanitized_string = string_value.strip()

    # Step 2: Check if sanitized_string is a string and meets length requirements
    if (
        not isinstance(sanitized_string, str)
        or len(sanitized_string) < 2
        or len(sanitized_string) > 80
    ):
        return {"string_validated": "false", "string_value": sanitized_string}

    # Step 3: Allowed characters in string value (including spaces)
    if not re.match("^[a-zA-Z ]*$", sanitized_string):
        return {"string_validated": "false", "string_value": sanitized_string}

    # Step 4: Sanitize the string to prevent XSS (Cross-Site Scripting)
    sanitized_string = html.escape(sanitized_string)

    # Step 5: Remove any non-alphanumeric characters except spaces
    sanitized_string = re.sub(r"[^a-zA-Z ]", "", sanitized_string)

    # Step 6: Preventing multiple XSS attacks
    sanitized_string = bleach.clean(sanitized_string)

    return {"string_validated": "true", "string_value": sanitized_string}


@app.route("/bse_check", methods=["POST"])
@jwt_required()
def base_name_checking():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        base_name = request.form["bse-nm"]
        base_name_validation = validate_and_sanitize_base_name_data(base_name)

        if base_name_validation["string_validated"] == "true":
            final_base_name = base_name_validation["string_value"]

            query = final_base_name.lower().strip()
            base_entry = indian_bases_data.find_one({"name": query}, {"_id": 0})
            satellite_information = list(
                satellite_tracking_data.find(
                    {"history.closest_base": query}, {"_id": 0}
                )
            )

            # creating list with contains all satellite history information
            list_of_all_satellite_history = []
            for satellite in satellite_information:
                satellite_history = satellite["history"]
                for history in satellite_history:
                    list_of_all_satellite_history.append(history)

            # creating list which contains history only which is related to queried indian base
            final_list_of_satellite_query_based = []
            for sats in list_of_all_satellite_history:
                closest_base = sats["closest_base"]
                if closest_base == query:
                    final_list_of_satellite_query_based.append(sats)

            # Assuming final_list_of_satellite_query_based is your list of satellite data entries
            unique_entries = []
            seen_entries = (
                set()
            )  # Tracks (satellite_name, tle_timestamp) to avoid duplicates
            for entry in final_list_of_satellite_query_based:
                key = (
                    entry["satellite_name"],
                    entry["tle_timestamp"],
                )  # Unique combination

                if key not in seen_entries:
                    seen_entries.add(key)
                    unique_entries.append(entry)

            # Sort the unique entries in descending order based on 'tle_timestamp'
            sorted_data = sorted(
                unique_entries, key=lambda x: x["tle_timestamp"], reverse=True
            )

            # Get the top two unique entries
            past_two_satellite = sorted_data[:2]

            # adding satellite pass timeline stamp
            satellite_pass_timeline_data = creating_satellite_pass_timeline(
                past_two_satellite
            )

            if base_entry:
                final_dict = replace_blank_and_NA_values(base_entry)

                # creating image link for base entry
                if "image" in final_dict:
                    image_value = final_dict["image"]
                    creating_image_link = creating_image_link_for_indian_base(
                        image_value
                    )
                    final_dict["image_link"] = creating_image_link
                else:
                    final_dict["image_link"] = "NA"
                    final_dict["image"] = "NA"

                if len(satellite_pass_timeline_data) != 0:
                    final_dict["pass_timeline"] = satellite_pass_timeline_data
                else:
                    final_dict["pass_timeline"] = "NA"

                result = final_dict
                return jsonify(result)
            else:
                result = {"warning": "base name not available"}
                return jsonify(result)
        else:
            return jsonify({"error": "Invalid input"}), 404
    else:
        return redirect(url_for("authentication"))


@app.route("/insbsesg", methods=["GET"])
@jwt_required()
def indian_bases_name_suggestions():

    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        query = request.args.get("query", "")
        if query:
            suggestions = list(
                indian_bases_data.find(
                    {"name": {"$regex": query, "$options": "i"}}, {"_id": 0}
                )
            )

            final_list_of_indian_bases = []
            for bases_entry in suggestions:
                final_dict_of_base = {}
                base_name = bases_entry["name"]

                image_value = bases_entry["image"]

                if image_value != "NA":
                    base_image_link = creating_image_link_for_indian_base(image_value)
                else:
                    base_image_link = "NA"

                # updating new dict
                final_dict_of_base["name"] = base_name.strip()
                final_dict_of_base["image_link"] = base_image_link.strip()
                final_list_of_indian_bases.append(final_dict_of_base)
            return jsonify(final_list_of_indian_bases)
        return jsonify([])
    else:
        return "Invalid user", 404


def convert_utc_to_ist(utc_time_str):
    try:
        # Parse the input string using dateutil's flexible parser
        utc_time = parser.isoparse(utc_time_str)

        # Convert to IST (UTC+5:30)
        ist_time = utc_time + datetime.timedelta(hours=5, minutes=30)
        proper_IST_time = str(ist_time).split("+")[0]

        return proper_IST_time
    except ValueError as e:
        print(f"Error parsing time string: {e}")
        return "NA"


def creating_satellite_pass_timeline(data_value):
    final_list = []
    for data in data_value:
        final_dict = {}
        satellite = data
        satellite_timestamp = satellite["tle_timestamp"]

        # Parse the satellite_timestamp
        parsed_satellite_timestamp = datetime.datetime.fromisoformat(
            satellite_timestamp
        )

        # Define your timezone (UTC+5:30)
        indian_timezone = pytz.timezone("Asia/Kolkata")

        # Convert the given timestamp to your timezone
        satellite_timestamp_according_to_indian_timezone = (
            parsed_satellite_timestamp.astimezone(indian_timezone)
        )

        # adding satellite_timestamp according to our timezone in dict
        data["local_timestamp"] = convert_utc_to_ist(satellite_timestamp)

        # Get the current time in your timezone
        current_indian_time = datetime.datetime.now(indian_timezone)

        # Check if satellite is directly overhead
        if satellite_timestamp_according_to_indian_timezone == current_indian_time:
            data["passed_time"] = "NA"
            data["timeline_status"] = "current"
        else:
            # Calculate the time difference
            time_difference = (
                current_indian_time - satellite_timestamp_according_to_indian_timezone
            )

            if time_difference:
                # Time difference exists (past event)
                time_passed = time_difference.total_seconds()
                hours_passed = int(time_passed // 3600)
                minutes_passed = int((time_passed % 3600) // 60)
                final_time_passed = f"{hours_passed} hours and {minutes_passed} minutes"
                data["passed_time"] = final_time_passed
                data["timeline_status"] = "past"

        final_dict["distance"] = data["distance_km"]
        final_dict["passed_time"] = data["passed_time"]
        final_dict["satellite_name"] = data["satellite_name"]
        final_dict["timeline_status"] = data["timeline_status"]
        final_dict["local_timestamp"] = data["local_timestamp"]
        final_list.append(final_dict)
    return final_list


@app.route("/snglbse", methods=["POST"])
@jwt_required()
def single_bases_entry():
    if "email" not in session:
        return redirect(url_for("authentication"))

    # Access token is still active
    current_user = get_jwt_identity()
    user = users.find_one({"email": current_user}, {"_id": 0})

    if user:
        base_name = request.get_json()
        if base_name:
            query = base_name.lower()
            base_information = indian_bases_data.find_one({"name": query}, {"_id": 0})
            satellite_information = list(
                satellite_tracking_data.find(
                    {"history.closest_base": query}, {"_id": 0}
                )
            )

            # creating list with contains all satellite history information
            list_of_all_satellite_history = []
            for satellite in satellite_information:
                satellite_history = satellite["history"]
                for history in satellite_history:
                    list_of_all_satellite_history.append(history)

            # creating list which contains history only which is related to queried indian base
            final_list_of_satellite_query_based = []
            for sats in list_of_all_satellite_history:
                closest_base = sats["closest_base"]
                if closest_base == query:
                    final_list_of_satellite_query_based.append(sats)

            # Assuming final_list_of_satellite_query_based is your list of satellite data entries
            unique_entries = []
            seen_entries = (
                set()
            )  # Tracks (satellite_name, tle_timestamp) to avoid duplicates
            for entry in final_list_of_satellite_query_based:
                key = (
                    entry["satellite_name"],
                    entry["tle_timestamp"],
                )  # Unique combination

                if key not in seen_entries:
                    seen_entries.add(key)
                    unique_entries.append(entry)

            # Sort the unique entries in descending order based on 'tle_timestamp'
            sorted_data = sorted(
                unique_entries, key=lambda x: x["tle_timestamp"], reverse=True
            )

            # Get the top two unique entries
            past_two_satellite = sorted_data[:2]

            # adding satellite pass timeline stamp
            satellite_pass_timeline_data = creating_satellite_pass_timeline(
                past_two_satellite
            )

            if base_information:
                # creating image link for base entry
                if "image" in base_information:
                    image_value = base_information["image"]
                    creating_image_link = creating_image_link_for_indian_base(
                        image_value
                    )
                    base_information["image_link"] = creating_image_link
                else:
                    base_information["image_link"] = "NA"
                    base_information["image"] = "NA"

                if len(satellite_pass_timeline_data) != 0:
                    base_information["pass_timeline"] = satellite_pass_timeline_data
                else:
                    base_information["pass_timeline"] = "NA"
                return jsonify(base_information)
            else:
                result = {"warning": "base name not available"}
                return jsonify(result)
        else:
            return jsonify({"warning": "Missing query value"})
    else:
        return "Invalid user", 404


def creating_human_readable_time(data_value):
    satellite = data_value
    satellite_timestamp = satellite["local_timestamp"]
    satellite_epoch = satellite["local_epoch"]
    satellite_propagation = satellite["local_propagation"]

    # Parse the input timestamp
    dt = datetime.datetime.strptime(satellite_timestamp, "%Y-%m-%d %H:%M:%S.%f")
    # Format the datetime object into the desired format
    formatted_timestamp = dt.strftime("%b %d, %Y, %I:%M %p")

    # Parse the input epoch
    dt = datetime.datetime.strptime(satellite_epoch, "%Y-%m-%d %H:%M:%S.%f")
    # Format the datetime object into the desired format
    formatted_epoch = dt.strftime("%b %d, %Y, %I:%M %p")

    # Parse the input propagation
    dt = datetime.datetime.strptime(satellite_propagation, "%Y-%m-%d %H:%M:%S")
    # Format the datetime object into the desired format
    formatted_propagation = dt.strftime("%b %d, %Y, %I:%M %p")

    satellite["formatted_timestamp"] = formatted_timestamp
    satellite["formatted_epoch"] = formatted_epoch
    satellite["formatted_propagation"] = formatted_propagation

    return satellite


def creating_satellite_pass_full_timeline(data_value):
    final_list = []
    for data in data_value:
        satellite = data
        satellite_timestamp = satellite["tle_timestamp"]
        satellite_epoch = satellite["tle_epoch"]
        satellite_propagation = satellite["propagation_time"]

        # Parse the satellite_timestamp
        parsed_satellite_timestamp = datetime.datetime.fromisoformat(
            satellite_timestamp
        )

        # Define your timezone (UTC+5:30)
        indian_timezone = pytz.timezone("Asia/Kolkata")

        # Convert the given timestamp to your timezone
        satellite_timestamp_according_to_indian_timezone = (
            parsed_satellite_timestamp.astimezone(indian_timezone)
        )

        # adding satellite_timestamp according to our timezone in dict
        data["local_timestamp"] = convert_utc_to_ist(satellite_timestamp)
        data["local_epoch"] = convert_utc_to_ist(satellite_epoch)
        data["local_propagation"] = convert_utc_to_ist(satellite_propagation)

        # adding human readable time format
        human_readable_time = creating_human_readable_time(satellite)
        data = human_readable_time

        # Get the current time in your timezone
        current_indian_time = datetime.datetime.now(indian_timezone)

        # Check if satellite is directly overhead
        if satellite_timestamp_according_to_indian_timezone == current_indian_time:
            data["passed_time"] = "NA"
            data["timeline_status"] = "current"
        else:
            # Calculate the time difference
            time_difference = (
                current_indian_time - satellite_timestamp_according_to_indian_timezone
            )

            if time_difference:
                # Time difference exists (past event)
                time_passed = time_difference.total_seconds()
                hours_passed = int(time_passed // 3600)
                minutes_passed = int((time_passed % 3600) // 60)
                final_time_passed = f"{hours_passed} hours and {minutes_passed} minutes"
                data["passed_time"] = final_time_passed
                data["timeline_status"] = "past"
        final_list.append(data)
    return final_list


@app.route("/satshisdata/<satellite_uid>", methods=["GET"])
def satellite_full_history_data(satellite_uid):
    if "email" not in session:
        return redirect(url_for("authentication"))

    satellite_ID = satellite_uid
    base_information = indian_bases_data.find_one(
        {"base_uid": satellite_ID}, {"_id": 0}
    )

    # extracting base name
    base_name = base_information["name"]

    # processing satellite all history
    satellite_information = list(
        satellite_tracking_data.find({"history.closest_base": base_name}, {"_id": 0})
    )

    # creating list with contains all satellite history information
    list_of_all_satellite_history = []
    for satellite in satellite_information:
        satellite_history = satellite["history"]
        for history in satellite_history:
            list_of_all_satellite_history.append(history)

    # creating list which contains history only which is related to queried indian base
    final_list_of_satellite_query_based = []
    for sats in list_of_all_satellite_history:
        closest_base = sats["closest_base"]
        if closest_base == base_name:
            final_list_of_satellite_query_based.append(sats)

    # Assuming final_list_of_satellite_query_based is your list of satellite data entries
    unique_entries = []
    seen_entries = set()  # Tracks (satellite_name, tle_timestamp) to avoid duplicates
    for entry in final_list_of_satellite_query_based:
        key = (entry["satellite_name"], entry["tle_timestamp"])  # Unique combination

        if key not in seen_entries:
            seen_entries.add(key)
            unique_entries.append(entry)

    # Sort the unique entries in descending order based on 'tle_timestamp'
    sorted_data = sorted(unique_entries, key=lambda x: x["tle_timestamp"], reverse=True)
    satellite_pass_timeline_data = creating_satellite_pass_full_timeline(sorted_data)

    return render_template(
        "satshisdata.html", satellite_timelines=satellite_pass_timeline_data
    )


if __name__ == "__main__":
    app.run()
