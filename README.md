# AI Resume Ranker

An AI-powered resume screening and ranking application built using Python and Streamlit.

## Project Overview

AI Resume Ranker helps recruiters evaluate multiple resumes against a given job description using Natural Language Processing and Machine Learning techniques.

## Features

- Upload multiple resumes in PDF and DOCX formats.
- Extract information from resumes.
- Identify technical skills.
- Analyze professional experience.
- Evaluate education qualifications.
- Compare resumes with job descriptions.
- Calculate semantic similarity using Sentence Transformers.
- Rank candidates based on matching criteria.
- Visualize candidate comparison results.

## Technologies Used

- Python
- Streamlit
- Pandas
- Sentence Transformers
- Scikit-learn
- PyMuPDF
- Python-docx
- Plotly

## Installation

1. Clone the repository:

   `git clone YOUR_REPOSITORY_URL`

2. Navigate to the project directory:

   `cd AI-Resume-Ranker`

3. Install dependencies:

   `pip install -r requirements.txt`

4. Run the application:

   `streamlit run app.py`

## AI Model

The project uses the `all-MiniLM-L6-v2` Sentence Transformer model to calculate semantic similarity between resumes and job descriptions.

## Future Improvements

- Add candidate database integration.
- Improve resume information extraction.
- Add downloadable candidate ranking reports.
- Develop a recruiter dashboard.

## Author

Samriddhi Khankale
Suraj Kudale
Ishan Kulkarni
Omraj Kulkarni