# Resume Screening & Candidate Ranking System

## Overview

A beginner-friendly Machine Learning and NLP project developed as part of the **Future Interns Machine Learning Internship Program—Task 3**.

This system compares resume text against a user-provided job description, identifies relevant skill keywords, calculates text similarity and skill match scores, and ranks candidates to help with initial resume screening.

## Features

- **Custom Job Description:** Enter a job description to screen resumes for a specific role.
- **Category Filtering:** Filter resumes by a selected job category, such as Information Technology.
- **Text Similarity Scoring:** Use TF-IDF vectorization and cosine similarity to compare resume text with the job description.
- **Skill Matching:** Identify required skill keywords found in each resume.
- **Missing Skill Detection:** Display required skill keywords that were not detected.
- **Skill Match Score:** Calculate the percentage of required skill keywords detected in each resume.
- **Combined Ranking Score:** Combine normalized text similarity (40%) and skill matching (60%) to rank candidates.
- **Candidate Ranking:** Display the top 10 candidates based on the combined score.
- **Visualization:** Generate a bar chart of candidate match scores.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Matplotlib

## Dataset

This project uses the [Resume Dataset on Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset).

The dataset is not included in this repository. Download it from Kaggle and place `Resume.csv` at the path expected by the code:

`datasets/Resume/Resume.csv`

Ensure you comply with the dataset's license and terms of use.

## Project Workflow

1. Load the resume dataset.
2. Filter resumes by the selected job category.
3. Accept a job description from the user.
4. Convert the job description and resume text into TF-IDF vectors.
5. Calculate cosine similarity scores.
6. Detect matched and missing skill keywords.
7. Calculate skill match and combined ranking scores.
8. Sort candidates and display the top 10 results.
9. Generate a candidate-ranking visualization.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/samsritha552/resume-screening-ml.git
cd resume-screening-ml
```

### 2. Install dependencies

It is recommended to create and activate a Python virtual environment first.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

### 3. Add the dataset

Place the downloaded `Resume.csv` file at:

`datasets/Resume/Resume.csv`

### 4. Run the project

```bash
python resume_screening.py
```

Enter a job description when prompted. The program will display the ranked candidates and generate a visualization.

## Understanding the Scores

- **Match Score:** Raw cosine similarity multiplied by 100.
- **Text Similarity Score (%):** The raw similarity score normalized against the highest text similarity score in the filtered dataset.
- **Skill Match Score (%):** The percentage of required skill keywords detected in a resume.
- **Overall Match Score (%):** A weighted experimental ranking score using 40% normalized text similarity and 60% skill matching.

These scores are ranking aids, not probabilities of being hired or validated measures of candidate quality.

## Limitations

- Skill matching uses basic keyword detection and may miss synonyms, alternative spellings, or skills described in different words.
- A missing skill means the keyword was not detected; it does not prove that the candidate lacks the skill.
- TF-IDF measures textual similarity, not a candidate's actual ability or suitability.
- The combined ranking weights are experimental and have not been validated against hiring outcomes.
- The system is intended to assist human review, not make hiring decisions automatically.

## Future Improvements

- Build an interactive dashboard using Streamlit.
- Improve skill extraction and synonym handling.
- Evaluate the ranking method using a carefully designed test set.

## Author

**Samsritha**  
B.Tech Student — Computer Science and Engineering (AI/ML)

Developed as part of the Future Interns Machine Learning Internship Program (2026).
