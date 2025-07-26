# AI Resume Ranking & Screening System  
*Automatically score resumes and validate profiles against GitHub/LinkedIn.*  

## Features  
1. **Resume-JD Matcher**: Predicts compatibility score between a resume and job description.  
2. **Profile Validator**:  
   - Authenticate via GitHub/LinkedIn OAuth.  
   - Compare skills (Resume vs. GitHub code, Resume vs. LinkedIn endorsements).  

## Tech Stack  
- Python, NLP (spaCy), Scikit-learn, Flask/FastAPI (for web).  
- APIs: GitHub API, LinkedIn API (OAuth 2.0).  

## How It Works  
1. Upload a resume (PDF/docx) and job description.  
2. Get a score (0-100) + skill gap analysis.  
3. Link GitHub/LinkedIn to validate consistency.  

## Setup  
```bash  
pip install -r requirements.txt  
