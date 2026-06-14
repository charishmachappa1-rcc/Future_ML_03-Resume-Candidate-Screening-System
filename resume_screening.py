import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
print(os.listdir())

# Step 1: Read resumes from file
with open("resumes.txt", "r") as file:
    resumes = file.readlines()

# Step 2: Job Description
job_description = "Looking for Python developer with machine learning and NLP skills"

# Step 3: Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

# Step 4: Clean Data
clean_resumes = [clean_text(resume) for resume in resumes]
clean_jd = clean_text(job_description)

# Step 5: Convert text to vectors
vectorizer = TfidfVectorizer()
all_text = clean_resumes + [clean_jd]
vectors = vectorizer.fit_transform(all_text)

# Step 6: Calculate similarity
scores = cosine_similarity(vectors[-1:], vectors[:-1])

# Step 7: Rank resumes
ranked = sorted(list(enumerate(scores[0])), key=lambda x: x[1], reverse=True)

print("\nCandidate Ranking:\n")
for i, score in ranked:
    print(f"Resume {i+1}: Score = {round(score*100, 2)}%")
    print(resumes[i])

# Step 8: Skill Extraction
skills = ["python", "machine learning", "nlp", "sql", "java"]

def extract_skills(text):
    return [skill for skill in skills if skill in text]

jd_skills = extract_skills(clean_jd)

print("\nSkill Gap Analysis:\n")

# Step 9: Skill Gap Analysis
for i, resume in enumerate(clean_resumes):
    res_skills = extract_skills(resume)
    missing = set(jd_skills) - set(res_skills)

    print(f"Resume {i+1}")
    print("Skills:", res_skills)
    print("Missing:", list(missing))
    print()
