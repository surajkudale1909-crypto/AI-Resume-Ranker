import streamlit as st
import pandas as pd
import re
import fitz

from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Ranker",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Ranker")

st.markdown(
    """
    Upload multiple resumes and compare them against a Job Description.

    The system evaluates:

    • Required skills  
    • Semantic similarity with the JD  
    • Professional experience  
    • Education level  
    • Education hierarchy  
    • Field / specialization  
    • Certifications
    """
)


# ============================================================
# EDUCATION HIERARCHY
# ============================================================

EDUCATION_HIERARCHY = {
    "diploma": 1,
    "bachelor": 2,
    "master": 3,
    "phd": 4
}


# ============================================================
# EDUCATION DEGREE MAPPING
# ============================================================

EDUCATION_DEGREES = {

    # --------------------------------------------------------
    # DIPLOMA
    # --------------------------------------------------------

    "diploma": {
        "level": "diploma",
        "aliases": [
            "diploma"
        ]
    },

    # --------------------------------------------------------
    # BACHELOR
    # --------------------------------------------------------

    "B.Tech": {
        "level": "bachelor",
        "aliases": [
            "b.tech",
            "btech",
            "b tech",
            "bachelor of technology"
        ]
    },

    "B.E.": {
        "level": "bachelor",
        "aliases": [
            "b.e.",
            "b.e",
            "be",
            "bachelor of engineering"
        ]
    },

    "B.Sc": {
        "level": "bachelor",
        "aliases": [
            "b.sc",
            "bsc",
            "bachelor of science"
        ]
    },

    "BCA": {
        "level": "bachelor",
        "aliases": [
            "bca",
            "bachelor of computer applications"
        ]
    },

    "B.Com": {
        "level": "bachelor",
        "aliases": [
            "b.com",
            "bcom",
            "bachelor of commerce"
        ]
    },

    # --------------------------------------------------------
    # MASTER
    # --------------------------------------------------------

    "M.Tech": {
        "level": "master",
        "aliases": [
            "m.tech",
            "mtech",
            "m tech",
            "master of technology"
        ]
    },

    "M.E.": {
        "level": "master",
        "aliases": [
            "m.e.",
            "m.e",
            "master of engineering"
        ]
    },

    "M.Sc": {
        "level": "master",
        "aliases": [
            "m.sc",
            "msc",
            "master of science"
        ]
    },

    "MCA": {
        "level": "master",
        "aliases": [
            "mca",
            "master of computer applications"
        ]
    },

    "MBA": {
        "level": "master",
        "aliases": [
            "mba",
            "master of business administration"
        ]
    },

    # --------------------------------------------------------
    # PHD
    # --------------------------------------------------------

    "Ph.D.": {
        "level": "phd",
        "aliases": [
            "ph.d.",
            "ph.d",
            "phd",
            "doctor of philosophy"
        ]
    }
}


# ============================================================
# SKILL DATABASE
# ============================================================

SKILL_DATABASE = {

    # Programming
    "python": ["python"],
    "java": ["java"],
    "c": ["c programming", "c language"],
    "cpp": ["c++", "cpp"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "go": ["golang", "go language"],
    "rust": ["rust"],

    # Backend
    "fastapi": ["fastapi"],
    "flask": ["flask"],
    "django": ["django"],
    "spring boot": ["spring boot"],
    "node.js": ["node.js", "nodejs", "node js"],
    "express.js": ["express.js", "expressjs", "express js"],
    "rest api": [
        "rest api",
        "rest apis",
        "restful api",
        "restful apis"
    ],
    "graphql": ["graphql"],

    # Frontend
    "html": ["html", "html5"],
    "css": ["css", "css3"],
    "react": ["react", "react.js", "reactjs"],
    "angular": ["angular"],
    "vue": ["vue", "vue.js"],

    # Databases
    "sql": ["sql"],
    "mysql": ["mysql"],
    "postgresql": ["postgresql", "postgres"],
    "mongodb": ["mongodb", "mongo db"],
    "sqlite": ["sqlite"],
    "oracle": ["oracle"],

    # DevOps / Cloud
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "git": ["git"],
    "github": ["github"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud"],
    "jenkins": ["jenkins"],
    "terraform": ["terraform"],

    # AI / ML
    "machine learning": [
        "machine learning",
        "machine-learning"
    ],

    "deep learning": [
        "deep learning",
        "deep-learning"
    ],

    "artificial intelligence": [
        "artificial intelligence"
    ],

    "scikit-learn": [
        "scikit-learn",
        "sklearn"
    ],

    "tensorflow": [
        "tensorflow"
    ],

    "pytorch": [
        "pytorch"
    ],

    "keras": [
        "keras"
    ],

    "opencv": [
        "opencv"
    ],

    "nlp": [
        "natural language processing",
        "nlp"
    ],

    "computer vision": [
        "computer vision"
    ],

    # Data
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "matplotlib": ["matplotlib"],
    "power bi": ["power bi", "powerbi"],
    "tableau": ["tableau"],
    "excel": ["microsoft excel", "excel"],

    # Other
    "linux": ["linux"],
    "solidity": ["solidity"],
    "blockchain": ["blockchain"]
}


# ============================================================
# FIELD / SPECIALIZATION DATABASE
# ============================================================

FIELD_ALIASES = {

    "computer science": [
        "computer science",
        "computer science engineering",
        "computer science and engineering",
        "cse"
    ],

    "computer engineering": [
        "computer engineering"
    ],

    "information technology": [
        "information technology",
        "it engineering",
        "information technology engineering"
    ],

    "artificial intelligence": [
        "artificial intelligence",
        "artificial intelligence and machine learning",
        "ai"
    ],

    "data science": [
        "data science"
    ],

    "software engineering": [
        "software engineering"
    ],

    "electronics": [
        "electronics",
        "electronics engineering",
        "ece"
    ],

    "mechanical engineering": [
        "mechanical engineering"
    ],

    "civil engineering": [
        "civil engineering"
    ],

    "statistics": [
        "statistics"
    ]
}


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


model = load_model()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.lower()

    text = text.replace("\n", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(file):

    text = ""

    pdf = fitz.open(
        stream=file.read(),
        filetype="pdf"
    )

    for page in pdf:

        text += page.get_text()

    pdf.close()

    return text


# ============================================================
# DOCX TEXT EXTRACTION
# ============================================================

def extract_docx_text(file):

    doc = Document(file)

    text = []

    for paragraph in doc.paragraphs:

        text.append(
            paragraph.text
        )

    return "\n".join(text)


# ============================================================
# RESUME TEXT EXTRACTION
# ============================================================

def extract_resume_text(file):

    if file.name.lower().endswith(".pdf"):

        return extract_pdf_text(file)

    elif file.name.lower().endswith(".docx"):

        return extract_docx_text(file)

    return ""


# ============================================================
# SKILL EXTRACTION
# ============================================================

def extract_skills(text):

    text = clean_text(text)

    detected_skills = []

    for skill, aliases in SKILL_DATABASE.items():

        for alias in aliases:

            pattern = (
                r"(?<![a-z])"
                + re.escape(alias.lower())
                + r"(?![a-z])"
            )

            if re.search(
                pattern,
                text
            ):

                detected_skills.append(
                    skill
                )

                break

    return sorted(
        set(detected_skills)
    )


# ============================================================
# EXPERIENCE EXTRACTION
# ============================================================

def extract_experience(text):

    text = clean_text(text)

    patterns = [

        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s*(?:of)?\s*experience",

        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s*exp",

        r"experience\s*(?:of|:|-)?\s*(\d+(?:\.\d+)?)\s*\+?\s*years?",

        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s*in"
    ]

    values = []

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text
        )

        for match in matches:

            try:

                values.append(
                    float(match)
                )

            except:

                pass

    if values:

        return max(values)

    return 0


# ============================================================
# REQUIRED EXPERIENCE FROM JD
# ============================================================

def extract_required_experience(jd):

    jd = clean_text(jd)

    patterns = [

        r"at least\s+(\d+(?:\.\d+)?)\s*\+?\s*years?",

        r"minimum\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*\+?\s*years?",

        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s*(?:of)?\s*experience",

        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s*experience"
    ]

    values = []

    for pattern in patterns:

        matches = re.findall(
            pattern,
            jd
        )

        for match in matches:

            try:

                values.append(
                    float(match)
                )

            except:

                pass

    if values:

        return max(values)

    return 0


# ============================================================
# DETECT EDUCATION IN RESUME
# ============================================================

def detect_education(text):

    text = clean_text(text)

    detected = []

    for degree, data in EDUCATION_DEGREES.items():

        for alias in data["aliases"]:

            pattern = (
                r"(?<![a-z])"
                + re.escape(alias.lower())
                + r"(?![a-z])"
            )

            if re.search(
                pattern,
                text
            ):

                detected.append({
                    "degree": degree,
                    "level": data["level"],
                    "level_value":
                        EDUCATION_HIERARCHY[
                            data["level"]
                        ]
                })

                break

    # Keep only the highest detected qualification
    if detected:

        detected.sort(
            key=lambda x: x["level_value"],
            reverse=True
        )

        return detected[0]

    return None


# ============================================================
# EDUCATION REQUIREMENT FROM JD
# ============================================================

def extract_required_education(jd):

    jd = clean_text(jd)

    # --------------------------------------------------------
    # IMPORTANT:
    # We deliberately detect ONE required education level.
    # We do not create "B.Tech or M.Tech" requirements.
    # --------------------------------------------------------

    phd_patterns = [
        "ph.d.",
        "ph.d",
        "phd",
        "doctor of philosophy"
    ]

    master_patterns = [
        "m.tech",
        "mtech",
        "m tech",
        "master of technology",
        "m.e.",
        "master of engineering",
        "m.sc",
        "msc",
        "master of science",
        "mca",
        "master of computer applications",
        "mba",
        "master of business administration",
        "master's degree",
        "masters degree",
        "master degree"
    ]

    bachelor_patterns = [
        "b.tech",
        "btech",
        "b tech",
        "bachelor of technology",
        "b.e.",
        "bachelor of engineering",
        "b.sc",
        "bsc",
        "bachelor of science",
        "bca",
        "bachelor of computer applications",
        "b.com",
        "bcom",
        "bachelor of commerce",
        "bachelor's degree",
        "bachelors degree",
        "bachelor degree"
    ]

    diploma_patterns = [
        "diploma"
    ]

    # Highest explicit requirement takes priority
    # if multiple education words happen to appear
    # in explanatory JD text.

    for pattern in phd_patterns:

        if re.search(
            r"(?<![a-z])"
            + re.escape(pattern)
            + r"(?![a-z])",
            jd
        ):

            return {
                "degree": "Ph.D.",
                "level": "phd",
                "level_value": 4
            }

    for pattern in master_patterns:

        if re.search(
            r"(?<![a-z])"
            + re.escape(pattern)
            + r"(?![a-z])",
            jd
        ):

            return {
                "degree": "Master's degree",
                "level": "master",
                "level_value": 3
            }

    for pattern in bachelor_patterns:

        if re.search(
            r"(?<![a-z])"
            + re.escape(pattern)
            + r"(?![a-z])",
            jd
        ):

            return {
                "degree": "Bachelor's degree",
                "level": "bachelor",
                "level_value": 2
            }

    for pattern in diploma_patterns:

        if re.search(
            r"(?<![a-z])"
            + re.escape(pattern)
            + r"(?![a-z])",
            jd
        ):

            return {
                "degree": "Diploma",
                "level": "diploma",
                "level_value": 1
            }

    return None


# ============================================================
# DETECT FIELDS
# ============================================================

def detect_fields(text):

    text = clean_text(text)

    detected = []

    for field, aliases in FIELD_ALIASES.items():

        for alias in aliases:

            pattern = (
                r"(?<![a-z])"
                + re.escape(alias.lower())
                + r"(?![a-z])"
            )

            if re.search(
                pattern,
                text
            ):

                detected.append(
                    field
                )

                break

    return sorted(
        set(detected)
    )


# ============================================================
# EXTRACT REQUIRED FIELDS FROM JD
# ============================================================

def extract_field_requirements(jd):

    jd = clean_text(jd)

    required_fields = []

    for field, aliases in FIELD_ALIASES.items():

        for alias in aliases:

            pattern = (
                r"(?<![a-z])"
                + re.escape(alias.lower())
                + r"(?![a-z])"
            )

            if re.search(
                pattern,
                jd
            ):

                required_fields.append(
                    field
                )

                break

    return sorted(
        set(required_fields)
    )


# ============================================================
# CERTIFICATIONS
# ============================================================

def extract_certification_requirements(jd):

    jd = clean_text(jd)

    certifications = []

    known_certifications = [

        "aws certified",

        "azure certified",

        "google cloud certified",

        "pmp",

        "cissp",

        "comptia",

        "cka",

        "ckad"
    ]

    for certification in known_certifications:

        if certification in jd:

            certifications.append(
                certification
            )

    return certifications


# ============================================================
# JD CONDITIONS
# ============================================================

def extract_jd_conditions(jd):

    conditions = []

    # --------------------------------------------------------
    # EXPERIENCE
    # --------------------------------------------------------

    required_experience = (
        extract_required_experience(jd)
    )

    if required_experience > 0:

        conditions.append({

            "type": "Experience",

            "required":
                f"{required_experience}+ years",

            "value":
                required_experience
        })

    # --------------------------------------------------------
    # EDUCATION
    # --------------------------------------------------------

    required_education = (
        extract_required_education(jd)
    )

    if required_education:

        conditions.append({

            "type": "Education",

            "required":
                required_education["degree"],

            "value":
                required_education
        })

    # --------------------------------------------------------
    # FIELD
    # --------------------------------------------------------

    fields = extract_field_requirements(
        jd
    )

    if fields:

        conditions.append({

            "type":
                "Field / Specialization",

            "required":
                ", ".join(fields),

            "value":
                fields
        })

    # --------------------------------------------------------
    # CERTIFICATIONS
    # --------------------------------------------------------

    certifications = (
        extract_certification_requirements(
            jd
        )
    )

    for certification in certifications:

        conditions.append({

            "type":
                "Certification",

            "required":
                certification,

            "value":
                certification
        })

    return conditions


# ============================================================
# CHECK EDUCATION HIERARCHY
# ============================================================

def check_education_condition(
    required_education,
    candidate_education
):

    if not candidate_education:

        return {

            "passed": False,

            "status": "FAIL",

            "candidate":
                "Education not detected"
        }

    required_level = (
        required_education["level_value"]
    )

    candidate_level = (
        candidate_education["level_value"]
    )

    # --------------------------------------------------------
    # HIERARCHY CHECK
    # --------------------------------------------------------
    #
    # Candidate must be at least the required level.
    #
    # B.Tech requirement:
    # B.Tech -> PASS
    # M.Tech -> PASS
    # Ph.D.  -> PASS
    #
    # M.Tech requirement:
    # B.Tech -> FAIL
    # M.Tech -> PASS
    # Ph.D.  -> PASS
    #
    # Ph.D. requirement:
    # B.Tech -> FAIL
    # M.Tech -> FAIL
    # Ph.D.  -> PASS
    # --------------------------------------------------------

    passed = (
        candidate_level >= required_level
    )

    return {

        "passed": passed,

        "status":
            "PASS"
            if passed
            else "FAIL",

        "candidate":
            candidate_education["degree"]
    }


# ============================================================
# CHECK ALL CONDITIONS
# ============================================================

def check_conditions(

    resume_text,

    jd_conditions,

    resume_experience,

    resume_education,

    resume_fields
):

    results = []

    resume_text_clean = (
        clean_text(resume_text)
    )

    for condition in jd_conditions:

        condition_type = (
            condition["type"]
        )

        # ====================================================
        # EXPERIENCE
        # ====================================================

        if condition_type == "Experience":

            required = condition["value"]

            passed = (
                resume_experience >= required
            )

            results.append({

                "Condition":
                    "Experience",

                "Required":
                    f"{required}+ years",

                "Candidate":
                    (
                        f"{resume_experience} years"
                        if resume_experience > 0
                        else "Not specified"
                    ),

                "Status":
                    "PASS"
                    if passed
                    else "FAIL",

                "Passed":
                    passed
            })

        # ====================================================
        # EDUCATION
        # ====================================================

        elif condition_type == "Education":

            required_education = (
                condition["value"]
            )

            education_result = (
                check_education_condition(
                    required_education,
                    resume_education
                )
            )

            results.append({

                "Condition":
                    "Education",

                "Required":
                    required_education[
                        "degree"
                    ],

                "Candidate":
                    education_result[
                        "candidate"
                    ],

                "Status":
                    education_result[
                        "status"
                    ],

                "Passed":
                    education_result[
                        "passed"
                    ]
            })

        # ====================================================
        # FIELD
        # ====================================================

        elif condition_type == (
            "Field / Specialization"
        ):

            required_fields = (
                condition["value"]
            )

            matched_fields = []

            for required_field in (
                required_fields
            ):

                if required_field in (
                    resume_fields
                ):

                    matched_fields.append(
                        required_field
                    )

            passed = (
                len(matched_fields) > 0
            )

            results.append({

                "Condition":
                    "Field / Specialization",

                "Required":
                    ", ".join(
                        required_fields
                    ),

                "Candidate":
                    (
                        ", ".join(
                            resume_fields
                        )
                        if resume_fields
                        else "Not specified"
                    ),

                "Status":
                    "PASS"
                    if passed
                    else "FAIL",

                "Passed":
                    passed
            })

        # ====================================================
        # CERTIFICATION
        # ====================================================

        elif condition_type == (
            "Certification"
        ):

            required_certification = (
                condition["value"]
            )

            passed = (
                required_certification
                in resume_text_clean
            )

            results.append({

                "Condition":
                    "Certification",

                "Required":
                    required_certification,

                "Candidate":
                    (
                        "Mentioned"
                        if passed
                        else "Not mentioned"
                    ),

                "Status":
                    "PASS"
                    if passed
                    else "FAIL",

                "Passed":
                    passed
            })

    return results


# ============================================================
# DETERMINE ELIGIBILITY
# ============================================================

def determine_eligibility(
    condition_results
):

    failed_conditions = [

        result

        for result in condition_results

        if not result["Passed"]
    ]

    if failed_conditions:

        return False, failed_conditions

    return True, []


# ============================================================
# SEMANTIC SIMILARITY
# ============================================================

def semantic_similarity(
    jd,
    resume
):

    jd_embedding = model.encode(
        [jd]
    )

    resume_embedding = model.encode(
        [resume]
    )

    similarity = cosine_similarity(

        jd_embedding,

        resume_embedding

    )[0][0]

    return float(
        similarity * 100
    )


# ============================================================
# SKILL SCORE
# ============================================================

def calculate_skill_score(

    required_skills,

    resume_skills
):

    if not required_skills:

        return 0, [], []

    normalized_required = [

        skill.strip().lower()

        for skill in required_skills

        if skill.strip()
    ]

    normalized_resume = [

        skill.lower()

        for skill in resume_skills
    ]

    matched = []

    missing = []

    for skill in normalized_required:

        found = False

        # Direct match
        if skill in normalized_resume:

            found = True

        # Alias match
        if (
            not found
            and skill in SKILL_DATABASE
        ):

            aliases = (
                SKILL_DATABASE[skill]
            )

            for alias in aliases:

                if (
                    alias.lower()
                    in normalized_resume
                ):

                    found = True
                    break

        if found:

            matched.append(
                skill
            )

        else:

            missing.append(
                skill
            )

    score = (

        len(matched)
        /
        len(normalized_required)

    ) * 100

    return (
        score,
        matched,
        missing
    )


# ============================================================
# KEYWORD SCORE
# ============================================================

def calculate_keyword_score(
    jd,
    resume
):

    jd_words = set(

        re.findall(

            r"\b[a-zA-Z]{3,}\b",

            clean_text(jd)
        )
    )

    resume_words = set(

        re.findall(

            r"\b[a-zA-Z]{3,}\b",

            clean_text(resume)
        )
    )

    if not jd_words:

        return 0

    common_words = (
        jd_words.intersection(
            resume_words
        )
    )

    score = (

        len(common_words)
        /
        len(jd_words)

    ) * 100

    return min(
        score,
        100
    )


# ============================================================
# EXPERIENCE SCORE
# ============================================================

def calculate_experience_score(

    candidate_experience,

    required_experience
):

    if required_experience <= 0:

        return 100

    if candidate_experience <= 0:

        return 0

    ratio = (

        candidate_experience
        /
        required_experience
    )

    return min(
        ratio * 100,
        100
    )


# ============================================================
# OVERALL SCORE
# ============================================================

def calculate_overall_score(

    skill_score,

    semantic_score,

    experience_score,

    keyword_score
):

    score = (

        0.40 * skill_score

        +

        0.35 * semantic_score

        +

        0.15 * experience_score

        +

        0.10 * keyword_score
    )

    return round(
        score,
        2
    )


# ============================================================
# COMPLETE CANDIDATE ANALYSIS
# ============================================================

def analyze_candidate(

    candidate_name,

    resume_text,

    jd,

    required_skills
):

    # --------------------------------------------------------
    # Resume extraction
    # --------------------------------------------------------

    resume_skills = (
        extract_skills(
            resume_text
        )
    )

    experience = (
        extract_experience(
            resume_text
        )
    )

    education = (
        detect_education(
            resume_text
        )
    )

    fields = (
        detect_fields(
            resume_text
        )
    )

    # --------------------------------------------------------
    # JD conditions
    # --------------------------------------------------------

    jd_conditions = (
        extract_jd_conditions(
            jd
        )
    )

    # --------------------------------------------------------
    # Condition checking
    # --------------------------------------------------------

    condition_results = (
        check_conditions(

            resume_text,

            jd_conditions,

            experience,

            education,

            fields
        )
    )

    eligible, failed_conditions = (
        determine_eligibility(
            condition_results
        )
    )

    # --------------------------------------------------------
    # Skills
    # --------------------------------------------------------

    (
        skill_score,
        matched_skills,
        missing_skills
    ) = calculate_skill_score(

        required_skills,

        resume_skills
    )

    # --------------------------------------------------------
    # Semantic match
    # --------------------------------------------------------

    semantic_score = (
        semantic_similarity(
            jd,
            resume_text
        )
    )

    # --------------------------------------------------------
    # Experience
    # --------------------------------------------------------

    required_experience = (
        extract_required_experience(
            jd
        )
    )

    experience_score = (
        calculate_experience_score(

            experience,

            required_experience
        )
    )

    # --------------------------------------------------------
    # Keyword
    # --------------------------------------------------------

    keyword_score = (
        calculate_keyword_score(

            jd,

            resume_text
        )
    )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    overall_score = (
        calculate_overall_score(

            skill_score,

            semantic_score,

            experience_score,

            keyword_score
        )
    )

    return {

        "Candidate":
            candidate_name,

        "Overall Score":
            overall_score,

        "Skill Score":
            round(
                skill_score,
                2
            ),

        "Semantic Match":
            round(
                semantic_score,
                2
            ),

        "Experience Score":
            round(
                experience_score,
                2
            ),

        "Keyword Score":
            round(
                keyword_score,
                2
            ),

        "Experience":
            experience,

        "Education":
            (
                education["degree"]
                if education
                else "Not specified"
            ),

        "Education Level":
            (
                education["level"]
                if education
                else "Not specified"
            ),

        "Fields":
            (
                ", ".join(fields)
                if fields
                else "Not specified"
            ),

        "Matched Skills":
            ", ".join(
                matched_skills
            ),

        "Missing Skills":
            ", ".join(
                missing_skills
            ),

        "Resume Skills":
            ", ".join(
                resume_skills
            ),

        "Condition Results":
            condition_results,

        "Eligible":
            eligible,

        "Failed Conditions":
            failed_conditions,

        "Resume Text":
            resume_text
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Scoring Settings"
)

st.sidebar.info(

    """
    ### Ranking Weights

    **Required Skills**
    → 40%

    **Semantic JD Match**
    → 35%

    **Experience**
    → 15%

    **Keyword Match**
    → 10%

    ### Education Hierarchy

    Ph.D.
    ↓
    Master's
    ↓
    Bachelor's
    ↓
    Diploma

    A higher qualification satisfies
    a lower-level education requirement.
    """
)


# ============================================================
# JOB DESCRIPTION
# ============================================================

st.header(
    "1️⃣ Job Description"
)

jd = st.text_area(

    "Paste the complete Job Description",

    height=280,

    placeholder="""
Example:

We are looking for a Machine Learning Engineer.

The candidate must have at least 2 years
of professional experience in machine learning.

The candidate must have a Bachelor's degree
in Computer Science.

The candidate must have hands-on experience
with Python and machine learning algorithms.

The candidate must have practical experience
using scikit-learn and at least one deep
learning framework.

Experience with SQL is required.

Experience with AWS is preferred but not mandatory.
"""
)


# ============================================================
# REQUIRED SKILLS
# ============================================================

st.header(
    "2️⃣ Required Skills"
)

required_skills_input = st.text_input(

    "Enter required skills separated by commas",

    placeholder=(
        "Python, Machine Learning, "
        "Scikit-learn, TensorFlow, SQL"
    )
)

required_skills = [

    skill.strip()

    for skill
    in required_skills_input.split(",")

    if skill.strip()
]


# ============================================================
# SHOW DETECTED CONDITIONS
# ============================================================

if jd:

    st.header(
        "🔍 Detected JD Conditions"
    )

    detected_conditions = (
        extract_jd_conditions(
            jd
        )
    )

    if detected_conditions:

        condition_display = []

        for condition in (
            detected_conditions
        ):

            condition_display.append({

                "Condition":
                    condition["type"],

                "Requirement":
                    condition["required"]
            })

        st.dataframe(

            pd.DataFrame(
                condition_display
            ),

            use_container_width=True,

            hide_index=True
        )

    else:

        st.info(
            "No explicit eligibility "
            "conditions detected."
        )


# ============================================================
# SHOW EDUCATION HIERARCHY
# ============================================================

if jd:

    required_education = (
        extract_required_education(
            jd
        )
    )

    if required_education:

        st.info(

            f"""
            **Education requirement detected:**
            {required_education['degree']}

            **Education rule:**
            A candidate with an education level
            equal to or higher than this level
            will satisfy the education requirement.
            """
        )


# ============================================================
# RESUME UPLOAD
# ============================================================

st.header(
    "3️⃣ Upload Resumes"
)

uploaded_files = st.file_uploader(

    "Upload PDF or DOCX resumes",

    type=[
        "pdf",
        "docx"
    ],

    accept_multiple_files=True
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(

    "🚀 Analyze & Rank Resumes",

    type="primary"
):

    if not jd.strip():

        st.error(
            "Please enter a Job Description."
        )

        st.stop()

    if not uploaded_files:

        st.error(
            "Please upload at least one resume."
        )

        st.stop()

    # --------------------------------------------------------
    # PROCESS RESUMES
    # --------------------------------------------------------

    results = []

    progress = st.progress(0)

    total = len(
        uploaded_files
    )

    for index, file in enumerate(
        uploaded_files
    ):

        resume_text = (
            extract_resume_text(
                file
            )
        )

        if not resume_text.strip():

            st.warning(

                f"Could not extract text "
                f"from {file.name}"
            )

            continue

        candidate_name = file.name

        candidate_name = re.sub(

            r"\.(pdf|docx)$",

            "",

            candidate_name,

            flags=re.IGNORECASE
        )

        result = (
            analyze_candidate(

                candidate_name,

                resume_text,

                jd,

                required_skills
            )
        )

        results.append(
            result
        )

        progress.progress(
            (index + 1) / total
        )

    if not results:

        st.error(
            "No readable resumes were found."
        )

        st.stop()

    # ========================================================
    # CREATE RANKING DATA
    # ========================================================

    ranking_data = []

    for result in results:

        ranking_data.append({

            "Candidate":
                result["Candidate"],

            "Eligibility":
                (
                    "✅ Eligible"
                    if result["Eligible"]
                    else "❌ Not Eligible"
                ),

            "Overall Score":
                result["Overall Score"],

            "Skill Score":
                result["Skill Score"],

            "Semantic Match":
                result["Semantic Match"],

            "Experience Score":
                result["Experience Score"],

            "Keyword Score":
                result["Keyword Score"],

            "Experience":
                result["Experience"],

            "Education":
                result["Education"],

            "Education Level":
                result["Education Level"],

            "Field":
                result["Fields"],

            "Matched Skills":
                result["Matched Skills"],

            "Missing Skills":
                result["Missing Skills"]
        })

    ranking_df = pd.DataFrame(
        ranking_data
    )

    # --------------------------------------------------------
    # ELIGIBILITY FIRST
    # --------------------------------------------------------

    ranking_df["Eligibility Rank"] = (

        ranking_df["Eligibility"]
        .str.contains(
            "Eligible"
        )
        .astype(int)
    )

    ranking_df = ranking_df.sort_values(

        by=[
            "Eligibility Rank",
            "Overall Score"
        ],

        ascending=[
            False,
            False
        ]
    )

    ranking_df = ranking_df.drop(
        columns=[
            "Eligibility Rank"
        ]
    )

    ranking_df = ranking_df.reset_index(
        drop=True
    )

    # ========================================================
    # RANKING TABLE
    # ========================================================

    st.header(
        "🏆 Resume Ranking"
    )

    st.dataframe(

        ranking_df,

        use_container_width=True,

        hide_index=True
    )


    # ========================================================
    # HIGHEST-RANKED ELIGIBLE CANDIDATE
    # ========================================================

    eligible_results = [

        result

        for result in results

        if result["Eligible"]
    ]

    st.header(
        "🎯 Highest-Ranked Eligible Match"
    )

    if eligible_results:

        highest_ranked = max(

            eligible_results,

            key=lambda x:
                x["Overall Score"]
        )

        st.success(

            f"""
            **{highest_ranked['Candidate']}**

            Overall Match Score:
            **{highest_ranked['Overall Score']}%**

            Skill Match:
            **{highest_ranked['Skill Score']}%**

            Semantic Match:
            **{highest_ranked['Semantic Match']}%**

            Experience:
            **{highest_ranked['Experience']} years**

            Education:
            **{highest_ranked['Education']}**
            """
        )

        st.caption(

            "This is the highest-ranked eligible "
            "match according to the configured "
            "criteria."
        )

    else:

        st.warning(

            "No candidate satisfies all detected "
            "mandatory conditions."
        )


    # ========================================================
    # SCORE COMPARISON
    # ========================================================

    st.header(
        "📊 Overall Score Comparison"
    )

    chart = px.bar(

        ranking_df,

        x="Candidate",

        y="Overall Score",

        text="Overall Score",

        title="Resume Match Scores"
    )

    chart.update_traces(

        texttemplate="%{text:.1f}%"
    )

    chart.update_layout(

        yaxis_title="Score (%)",

        xaxis_title="Candidate"
    )

    st.plotly_chart(

        chart,

        use_container_width=True
    )


    # ========================================================
    # DETAILED ANALYSIS
    # ========================================================

    st.header(
        "🔎 Detailed Candidate Analysis"
    )

    for result in results:

        with st.expander(

            f"{result['Candidate']} — "
            f"{result['Overall Score']}%"
        ):

            # ------------------------------------------------
            # BASIC METRICS
            # ------------------------------------------------

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            col1.metric(

                "Overall Score",

                f"{result['Overall Score']}%"
            )

            col2.metric(

                "Skill Score",

                f"{result['Skill Score']}%"
            )

            col3.metric(

                "Semantic Match",

                f"{result['Semantic Match']}%"
            )

            col4.metric(

                "Experience",

                f"{result['Experience']} years"
            )


            # ------------------------------------------------
            # ELIGIBILITY
            # ------------------------------------------------

            if result["Eligible"]:

                st.success(

                    "✅ Candidate satisfies all "
                    "detected mandatory conditions."
                )

            else:

                st.error(

                    "❌ Candidate does not satisfy "
                    "all detected mandatory conditions."
                )

                if result[
                    "Failed Conditions"
                ]:

                    st.write(
                        "**Failed Conditions:**"
                    )

                    for failure in result[
                        "Failed Conditions"
                    ]:

                        st.write(

                            f"- **{failure['Condition']}**: "
                            f"Required: "
                            f"{failure['Required']} | "
                            f"Candidate: "
                            f"{failure['Candidate']}"
                        )


            # ------------------------------------------------
            # EDUCATION
            # ------------------------------------------------

            st.subheader(
                "🎓 Education"
            )

            st.write(

                f"**Highest detected qualification:** "
                f"{result['Education']}"
            )

            st.write(

                f"**Education level:** "
                f"{result['Education Level']}"
            )

            st.write(

                f"**Specialization:** "
                f"{result['Fields']}"
            )


            # ------------------------------------------------
            # CONDITION TABLE
            # ------------------------------------------------

            if result[
                "Condition Results"
            ]:

                st.subheader(
                    "📋 Condition Analysis"
                )

                condition_df = (
                    pd.DataFrame(
                        result[
                            "Condition Results"
                        ]
                    )
                )

                condition_df = (
                    condition_df[
                        [
                            "Condition",
                            "Required",
                            "Candidate",
                            "Status"
                        ]
                    ]
                )

                st.dataframe(

                    condition_df,

                    use_container_width=True,

                    hide_index=True
                )


            # ------------------------------------------------
            # SKILLS
            # ------------------------------------------------

            st.subheader(
                "🛠️ Skills Analysis"
            )

            col1, col2 = (
                st.columns(2)
            )

            with col1:

                st.write(
                    "### ✅ Matched Skills"
                )

                if result[
                    "Matched Skills"
                ]:

                    for skill in result[
                        "Matched Skills"
                    ].split(", "):

                        st.write(
                            f"✓ {skill}"
                        )

                else:

                    st.write(
                        "No required skills matched."
                    )

            with col2:

                st.write(
                    "### ❌ Missing Skills"
                )

                if result[
                    "Missing Skills"
                ]:

                    for skill in result[
                        "Missing Skills"
                    ].split(", "):

                        st.write(
                            f"✗ {skill}"
                        )

                else:

                    st.write(
                        "No required skills missing."
                    )


            # ------------------------------------------------
            # ALL RESUME SKILLS
            # ------------------------------------------------

            st.subheader(
                "📚 Detected Resume Skills"
            )

            st.write(

                result["Resume Skills"]

                if result["Resume Skills"]

                else "No skills detected."
            )


            # ------------------------------------------------
            # SCORE BREAKDOWN
            # ------------------------------------------------

            st.subheader(
                "📈 Score Breakdown"
            )

            score_data = pd.DataFrame({

                "Metric": [

                    "Required Skills",

                    "Semantic JD Match",

                    "Experience",

                    "Keyword Match"
                ],

                "Score": [

                    result[
                        "Skill Score"
                    ],

                    result[
                        "Semantic Match"
                    ],

                    result[
                        "Experience Score"
                    ],

                    result[
                        "Keyword Score"
                    ]
                ]
            })

            score_chart = px.bar(

                score_data,

                x="Metric",

                y="Score",

                text="Score",

                title="Candidate Score Breakdown"
            )

            score_chart.update_traces(

                texttemplate="%{text:.1f}%"
            )

            score_chart.update_layout(

                yaxis_title="Score (%)",

                xaxis_title=""
            )

            st.plotly_chart(

                score_chart,

                use_container_width=True
            )


            # ------------------------------------------------
            # RESUME TEXT
            # ------------------------------------------------

            with st.expander(
                "📄 Extracted Resume Text"
            ):

                st.text(
                    result["Resume Text"]
                )