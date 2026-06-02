import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv(
    "datasets/Resume/Resume.csv",
    encoding="latin1"
)

# Sample Job Description
job_description = """
Looking for a Python Developer with skills in Python,
Machine Learning, Data Analysis, SQL, Pandas,
NumPy and Scikit-Learn.
"""

# Use first 100 resumes for faster processing
resumes = df["Resume_str"].head(100)

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

# Add scores to dataframe
ranking_df = df.head(100).copy()

ranking_df["Match Score"] = similarity_scores.flatten() * 100

# Sort by score
ranking_df = ranking_df.sort_values(
    by="Match Score",
    ascending=False
)

# Top 10 candidates
top_candidates = ranking_df[
    ["ID", "Category", "Match Score"]
].head(10)

print("\nTop 10 Ranked Candidates:\n")
print(top_candidates)

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
