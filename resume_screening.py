import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv(
    "datasets/Resume/Resume.csv",
    encoding="latin1"
)

# Display available resume categories
print("\nAvailable Resume Categories:")
print(df["Category"].value_counts())

# Sample Job Description
job_description = input("Enter the job description:\n")

# Filter resumes relevant to technology roles
filtered_df = df[
    df["Category"] == "INFORMATION-TECHNOLOGY"
].copy()

# Use the filtered resumes for screening
resumes = filtered_df["Resume_str"]

# Combine job description and resumes
documents = [job_description] + resumes.tolist()

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english")

tfidf_matrix = vectorizer.fit_transform(documents)

# Similarity scores
similarity_scores = cosine_similarity(
    tfidf_matrix[0:1],
    tfidf_matrix[1:]
)

ranking_df = filtered_df.copy()

ranking_df["Match Score"] = similarity_scores.flatten() * 100


# Skills to check in each resume
skills = [
    "Python",
    "Machine Learning",
    "Data Analysis",
    "SQL",
    "Pandas",
    "NumPy",
    "Scikit-Learn",
    "sklearn",
    "Excel",
    "Deep Learning",
    "NLP",
    "TensorFlow",
    "PyTorch",
    "Power BI",
    "Tableau",
    "Java",
    "C++",
    "AWS"
]

# Identify skills mentioned in the job description
required_skills = [
    skill for skill in skills
    if skill.lower() in job_description.lower()
]

# Find matched and missing skills for each candidate


def analyze_skills(resume):
    resume_text = str(resume).lower()

    matched = [
        skill for skill in required_skills
        if skill.lower() in resume_text
    ]

    missing = [
        skill for skill in required_skills
        if skill.lower() not in resume_text
    ]

    return pd.Series({
        "Matched Skills": ", ".join(matched) if matched else "None found",
        "Missing Skills": ", ".join(missing) if missing else "None detected"
    })


skill_results = ranking_df["Resume_str"].apply(analyze_skills)

ranking_df[["Matched Skills", "Missing Skills"]] = skill_results


# Calculate the percentage of required skills found
def calculate_skill_match(matched_skills):
    if not required_skills:
        return 0.0

    if matched_skills == "None found":
        matched_count = 0
    else:
        matched_count = len(matched_skills.split(", "))

    return round((matched_count / len(required_skills)) * 100, 2)


ranking_df["Skill Match Score (%)"] = (
    ranking_df["Matched Skills"].apply(calculate_skill_match)
)


# Normalize text similarity scores to a 0–100 scale
max_score = ranking_df["Match Score"].max()

if max_score > 0:
    ranking_df["Text Similarity Score (%)"] = (
        ranking_df["Match Score"] / max_score * 100
    )
else:
    ranking_df["Text Similarity Score (%)"] = 0.0

# Combine text similarity and skill matching
ranking_df["Overall Match Score (%)"] = (
    0.4 * ranking_df["Text Similarity Score (%)"]
    + 0.6 * ranking_df["Skill Match Score (%)"]
)

# Rank candidates using the combined score
ranking_df = ranking_df.sort_values(
    by="Overall Match Score (%)",
    ascending=False
)


top_candidates = ranking_df[
    [
        "ID",
        "Category",
        "Match Score",
        "Skill Match Score (%)",
        "Matched Skills",
        "Missing Skills",
        "Text Similarity Score (%)",
        "Overall Match Score (%)"
    ]
].head(10)


print("\nTop 10 Ranked Candidates:\n")
print(top_candidates.to_string(index=False))

# Visualization
plt.figure(figsize=(10, 5))

plt.bar(
    top_candidates["ID"].astype(str),
    top_candidates["Match Score"]
)

plt.title("Top 10 Candidate Match Scores")
plt.xlabel("Candidate ID")
plt.ylabel("Match Score (%)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("visuals/candidate_ranking.png")

plt.show()
