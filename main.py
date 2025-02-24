import os
import re
import json
import pdfplumber
import spacy
from google import genai  # Using the google-genai library
from flask import Flask, render_template, request, redirect, url_for, flash

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "secretkey"  # For flash messages

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Predefined lists for skills and education keywords
SKILLS_KEYWORDS = [
    "python", "java", "javascript", "sql", "machine learning", "data analysis",
    "project management", "communication", "teamwork", "problem solving"
]

EDUCATION_KEYWORDS = [
    "bachelor", "master", "phd", "university", "college", "institute", "diploma", "school"
]

def allowed_file(filename):
    """Check if the uploaded file is a PDF."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using pdfplumber."""
    with pdfplumber.open(pdf_path) as pdf:
        pages_text = [page.extract_text() for page in pdf.pages if page.extract_text()]
    text = "\n".join(pages_text)
    # Clean up excessive whitespace
    return re.sub(r'\s+', ' ', text).strip()

def parse_resume(text):
    """
    Use spaCy's NER and regex to extract key information.
    Extracts: name, email, phone, skills, education, and experience.
    """
    doc = nlp(text)

    # Extract name as the first PERSON entity found
    name = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), "Not Found")

    # Extract email using regex
    email_search = re.search(r"[\w\.-]+@[\w\.-]+", text)
    email = email_search.group(0) if email_search else "Not Found"

    # Extract phone number (covers many common formats)
    phone_search = re.search(r"\+?\d[\d\s\-\(\)]{8,15}\d", text)
    phone = phone_search.group(0) if phone_search else "Not Found"

    # Extract skills using fuzzy matching
    skills = []
    for skill in SKILLS_KEYWORDS:
        matches = re.findall(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE)
        if matches:
            skills.append(skill)
    skills = list(set(skills))  # Remove duplicates

    # Extract education details in a structured format
    education = []
    for keyword in EDUCATION_KEYWORDS:
        matches = re.findall(rf"\b{re.escape(keyword)}\b.*?(?:\n|$)", text, re.IGNORECASE)
        for match in matches:
            # Clean up irrelevant phrases
            cleaned_match = re.sub(r"(Hobbies|Accomplishments).*", "", match).strip()
            if len(cleaned_match.split()) > 3:  # Ensure meaningful entry
                education.append(cleaned_match)

    # Summarize education into structured format
    structured_education = []
    for entry in education:
        institution = re.search(r"[A-Z][a-zA-Z\s&]+(?:[,\s]*[A-Z][a-zA-Z\s&]*)*", entry)
        degree = re.search(r"(?i)(bachelor|master|phd|diploma|certificate).*?(?:\n|$)", entry)
        duration = re.search(r"(?i)(\d{4})\s*-\s*(?:present|\d{4})", entry)
        structured_entry = {
            "institution": institution.group(0) if institution else "Not Found",
            "degree": degree.group(0).strip() if degree else "Not Found",
            "duration": duration.group(0) if duration else "Not Found"
        }
        structured_education.append(structured_entry)

    # Extract work experience in a structured format
    experience = []
    job_title_patterns = [
        r"(?i)(software engineer|data analyst|project manager|developer|consultant)",
        r"(?i)(senior|junior|lead|manager|specialist)"
    ]
    company_patterns = [
        r"(?i)(at|for)\s+([A-Z][a-zA-Z\s&]+(?:[,\s]*[A-Z][a-zA-Z\s&]*)*)",
        r"(?i)(worked\s+at|employed\s+by)\s+([A-Z][a-zA-Z\s&]+)"
    ]
    duration_patterns = [
        r"(?i)(\d{4})\s*-\s*(?:present|\d{4})",
        r"(?i)(\d{1,2}\s+\w+\s+\d{4})\s*-\s*(?:present|\d{1,2}\s+\w+\s+\d{4})"
    ]

    # Combine all patterns to extract structured experience
    for pattern in job_title_patterns + company_patterns + duration_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):  # Handle regex groups
                experience.append(" ".join(match))
            else:
                experience.append(match)

    # Summarize experience into structured format
    structured_experience = []
    for entry in experience:
        title = re.search(r"(?i)(software engineer|data analyst|project manager|developer|consultant)", entry)
        company = re.search(r"(?i)(at|for)\s+([A-Z][a-zA-Z\s&]+(?:[,\s]*[A-Z][a-zA-Z\s&]*)*)", entry)
        duration = re.search(r"(?i)(\d{4})\s*-\s*(?:present|\d{4})", entry)
        structured_entry = {
            "title": title.group(0) if title else "Not Found",
            "company": company.group(2) if company else "Not Found",
            "duration": duration.group(0) if duration else "Not Found"
        }
        structured_experience.append(structured_entry)

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "skills": skills,
        "education": structured_education,
        "experience": structured_experience
    }

def rate_resume_with_gemini(resume_text):
    """
    Rate the resume using the google-genai library.
    We ask the model to return a JSON object with a field "rating" (an integer 1-10).
    """
    client = genai.Client(api_key="AIzaSyACCG9fWDn7aMQxxgT6HLzgGkePogEAzZo")
    prompt = (
        "Evaluate the following resume and provide a rating from 1 to 10 based on its quality and relevance. "
        "Return your response in the following JSON format without any extra text:\n"
        '{"rating": <number>}\n\n'
        f"{resume_text}"
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    raw_text = response.text.strip()
    try:
        data = json.loads(raw_text)
        rating = data.get("rating", "Not Available")
        return rating
    except Exception as e:
        rating_match = re.search(r'\b(\d+)\b', raw_text)
        return rating_match.group(1) if rating_match else "Not Available"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'resume' not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files['resume']
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            resume_text = extract_text_from_pdf(filepath)
            parsed_info = parse_resume(resume_text)
            rating = rate_resume_with_gemini(resume_text)
            parsed_info["rating"] = rating
            json_filename = filename.rsplit('.', 1)[0] + ".json"
            json_filepath = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)
            with open(json_filepath, "w") as f:
                json.dump(parsed_info, f, indent=4)
            return render_template("result.html", info=parsed_info)
        else:
            flash("Allowed file type is PDF")
            return redirect(request.url)
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)