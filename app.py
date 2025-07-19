import io
import os
import pickle
import re
import fitz
import time
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.pdfpage import PDFPage
from io import StringIO 
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import os

app = Flask(__name__)

model_dir = os.path.join(os.path.dirname(__file__), 'model')
clf = pickle.load(open(os.path.join(model_dir, 'clf.pkl'), 'rb'))
vectorizer = pickle.load(open(os.path.join(model_dir, 'vectorizer.pkl'), 'rb'))
label_encoder = pickle.load(open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb'))

def get_category_name(pred_id):
    try:
        return label_encoder.inverse_transform([pred_id])[0]
    except:
        return "Unknown"


def extract_skills(text):
    keywords = [
        "Python", "Java", "JavaScript", "C++", "C#", "Go", "Rust", "Ruby", "PHP", "Swift",
        "Kotlin", "TypeScript", "Scala", "R", "MATLAB", "SQL", "NoSQL", "MongoDB", "PostgreSQL", "MySQL",
        "React", "Angular", "Vue.js", "HTML", "CSS", "SASS", "Bootstrap", "Tailwind CSS", "jQuery",
        "Node.js", "Express.js", "Django", "Flask", "Spring Boot", "Laravel", "ASP.NET",
        "REST API", "GraphQL", "JSON", "AJAX", "FastAPI", "Git", "GitHub", "GitLab", "Docker",
        "Kubernetes", "Jenkins", "CI/CD", "Ansible", "Terraform", "Vagrant", "AWS", "Azure", "Google Cloud",
        "Linux", "Unix", "Windows Server", "Shell Scripting", "Bash", "PowerShell", "Machine Learning",
        "Deep Learning", "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy", "Matplotlib", "Seaborn",
        "Jupyter", "Keras", "XGBoost", "OpenCV", "Data Analysis", "Data Science", "Big Data", "Apache Spark",
        "Hadoop", "Tableau", "Power BI", "Excel", "Google Analytics", "ETL", "Airflow", "Kafka", "dbt",
        "Snowflake", "Looker", "Artificial Intelligence", "Natural Language Processing", "Computer Vision",
        "Neural Networks", "Statistics", "Probability", "Mathematics", "Algorithms", "Data Structures",
        "Problem Solving", "Project Management", "Agile", "Scrum", "Kanban", "JIRA", "Confluence", "Slack",
        "Microsoft Teams", "Zoom", "Communication", "Leadership", "Team Management", "Time Management",
        "Critical Thinking", "Analytical Thinking", "Creative Thinking", "Attention to Detail", "Multitasking",
        "Adaptability", "Collaboration", "Presentation Skills", "Decision Making", "Self-Motivation"
    ]
    return [kw for kw in keywords if kw.lower() in text.lower()]


def extract_text_from_pdf_file(file_stream):
    resource_manager = PDFResourceManager()
    string_io = StringIO()
    converter = TextConverter(resource_manager, string_io)
    interpreter = PDFPageInterpreter(resource_manager, converter)
    for page in PDFPage.get_pages(file_stream, caching=True, check_extractable=True):
        interpreter.process_page(page)
    text = string_io.getvalue()
    converter.close()
    string_io.close()
    return text


def clean_resume(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


@app.route("/")
def home():
    return render_template("home.html") 


@app.route("/analyze_description", methods=["GET", "POST"])
def analyze_description():
    if request.method == 'POST':
        if 'resume' not in request.files:
            return "❌ Please upload a resume PDF."

        resume_file = request.files['resume']
        job_description = request.form.get('job', '')

        if resume_file.filename == '':
            return "❌ No selected file."
        if not job_description.strip():
            return "❌ Please enter a job description."

        try:
            # ➤ Extract and clean
            file_stream = resume_file.stream
            raw_text = extract_text_from_pdf_file(file_stream)
            cleaned = clean_resume(raw_text)

            # ➤ Vectorize resume and job description
            resume_vec = vectorizer.transform([cleaned])
            job_vec = vectorizer.transform([clean_resume(job_description)])

            # ➤ Predict category
            pred_id = clf.predict(resume_vec)[0]
            category = get_category_name(pred_id)

            # ➤ Extract skills and match score
            skills = extract_skills(raw_text)
            similarity = cosine_similarity(job_vec, resume_vec).flatten()[0]

            return render_template("index1.html",
                                   category=category,
                                   score=similarity * 100,
                                   skill=skills)
        except Exception as e:
            return f"⚠️ Error processing the resume: {str(e)}"

    return render_template("index1.html")


all_skills = [
    "Python", "Java", "JavaScript", "C++", "C#", "Go", "Rust", "Ruby", "PHP", "Swift",
        "Kotlin", "TypeScript", "Scala", "R", "MATLAB", "SQL", "NoSQL", "MongoDB", "PostgreSQL", "MySQL",
        "React", "Angular", "Vue.js", "HTML", "CSS", "SASS", "Bootstrap", "Tailwind CSS", "jQuery",
        "Node.js", "Express.js", "Django", "Flask", "Spring Boot", "Laravel", "ASP.NET",
        "REST API", "GraphQL", "JSON", "AJAX", "FastAPI", "Git", "GitHub", "GitLab", "Docker",
        "Kubernetes", "Jenkins", "CI/CD", "Ansible", "Terraform", "Vagrant", "AWS", "Azure", "Google Cloud",
        "Linux", "Unix", "Windows Server", "Shell Scripting", "Bash", "PowerShell", "Machine Learning",
        "Deep Learning", "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy", "Matplotlib", "Seaborn",
        "Jupyter", "Keras", "XGBoost", "OpenCV", "Data Analysis", "Data Science", "Big Data", "Apache Spark",
        "Hadoop", "Tableau", "Power BI", "Excel", "Google Analytics", "ETL", "Airflow", "Kafka", "dbt",
        "Snowflake", "Looker", "Artificial Intelligence", "Natural Language Processing", "Computer Vision",
        "Neural Networks", "Statistics", "Probability", "Mathematics", "Algorithms", "Data Structures",
        "Problem Solving", "Project Management", "Agile", "Scrum", "Kanban", "JIRA", "Confluence", "Slack",
        "Microsoft Teams", "Zoom", "Communication", "Leadership", "Team Management", "Time Management",
        "Critical Thinking", "Analytical Thinking", "Creative Thinking", "Attention to Detail", "Multitasking",
        "Adaptability", "Collaboration", "Presentation Skills", "Decision Making", "Self-Motivation"
]
skill_variations = {"node.js": ["nodejs"], "data science": ["data-science"]}

def find_skills_in_text(text):
    text = text.lower()
    found = set()
    for skill in all_skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', text):
            found.add(skill)
    for main, variations in skill_variations.items():
        for var in variations:
            if re.search(r'\b' + re.escape(var) + r'\b', text):
                found.add(main)
    return found

def extract_github_skills(username):
    skills = set()
    try:
        repos = requests.get(f"https://api.github.com/users/{username}/repos").json()
        for repo in repos:
            lang = repo.get("language")
            if lang:
                skills.add(lang.lower())
            readme_url = f"https://raw.githubusercontent.com/{username}/{repo['name']}/main/README.md"
            try:
                readme = requests.get(readme_url).text.lower()
                skills.update(find_skills_in_text(readme))
            except:
                continue
    except Exception as e:
        print("GitHub Error:", e)
    return list(skills)

def extract_linkedin_skills(linkedin_url, email, password):
    skills = set()
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get("https://www.linkedin.com/login")
        time.sleep(2)
        driver.find_element(By.ID, "username").send_keys(email)
        driver.find_element(By.ID, "password").send_keys(password)
        driver.find_element(By.XPATH, "//button[@type='submit']").click()
        time.sleep(3)

        driver.get(linkedin_url)
        time.sleep(5)

        for _ in range(3):
            driver.execute_script("window.scrollBy(0, 1000);")
            time.sleep(2)

        page_text = driver.find_element(By.TAG_NAME, "body").text
        skills.update(find_skills_in_text(page_text))

    except Exception as e:
        print("LinkedIn Error:", e)
    finally:
        driver.quit()

    return list(skills)

def extract_resume_text(file_bytes):
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return " ".join([page.get_text().lower() for page in doc])
    except Exception as e:
        print("Resume PDF Error:", e)
        return ""

def calculate_match_score(resume_text, extracted_skills):
    if not resume_text or not extracted_skills:
        return 0.0
    documents = [resume_text, " ".join(extracted_skills)]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(documents)
    return round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100, 2)


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == "POST":
        github_username = request.form.get("github")
        linkedin_url = request.form.get("linkedin")
        email = request.form.get("email")
        password = request.form.get("password")
        file = request.files["resume"]

        resume_text = extract_resume_text(file.read())

        github_skills = extract_github_skills(github_username) if github_username else []
        linkedin_skills = extract_linkedin_skills(linkedin_url, email, password) if linkedin_url and email and password else []

        github_score = calculate_match_score(resume_text, github_skills)
        linkedin_score = calculate_match_score(resume_text, linkedin_skills)

        return render_template("index.html", 
            github_skills=github_skills, 
            linkedin_skills=linkedin_skills,
            github_score=github_score,
            linkedin_score=linkedin_score,
            resume_length=len(resume_text)
        )

    return render_template("index.html")

@app.route("/index")
def index():
    return render_template("index.html")




if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  
    app.run(debug=True, host='0.0.0.0', port=port)
