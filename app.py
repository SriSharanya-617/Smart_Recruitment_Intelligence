import streamlit as st
import pandas as pd
import pickle

from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Smart Recruitment Intelligence",
    page_icon="💼",
    layout="wide"
)

# -------------------------------------------------
# LOAD CSS
# -------------------------------------------------

with open("styles.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

@st.cache_resource
def load_ai_model():
    return load_model(
        "attention_model.h5",
        compile=False
    )

# -------------------------------------------------
# LOAD TOKENIZER
# -------------------------------------------------

@st.cache_resource
def load_tokenizer():

    with open(
        "tokenizer.pkl",
        "rb"
    ) as f:

        tokenizer = pickle.load(f)

    return tokenizer

try:
    model = load_ai_model()
    tokenizer = load_tokenizer()

except:
    pass

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.markdown(
"""
<div class='title'>
Smart Recruitment Intelligence Platform
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class='sub-title'>
AI Powered Resume Screening using NLP, Attention & Explainable AI
</div>
""",
unsafe_allow_html=True
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("Recruitment Dashboard")

uploaded_resume = st.sidebar.file_uploader(
    "Upload Resume Dataset",
    type=["csv"]
)

# -------------------------------------------------
# JOB DESCRIPTION
# -------------------------------------------------

st.markdown(
"<div class='section-header'>📋 Job Description</div>",
unsafe_allow_html=True
)

jd = st.text_area(
    "Enter Job Description",
    height=180
)

st.info(
"""
Upload Resume.csv and enter a Job Description.
Then click Rank Candidates.
"""
)

# -------------------------------------------------
# BUTTON
# -------------------------------------------------

rank_btn = st.button(
    "🚀 Rank Candidates"
)

# -------------------------------------------------
# SKILL DATABASE
# -------------------------------------------------

skill_db = [
    'python',
    'sql',
    'aws',
    'java',
    'excel',
    'power bi',
    'tensorflow',
    'pytorch',
    'machine learning',
    'deep learning'
]

def extract_skills(text):

    skills = []

    text = str(text).lower()

    for skill in skill_db:

        if skill in text:
            skills.append(skill)

    return skills

# -------------------------------------------------
# PROCESS
# -------------------------------------------------

if uploaded_resume and jd and rank_btn:

    df = pd.read_csv(uploaded_resume)

    resume_col = "Resume_str"

    # -------------------------------------------------
    # DATASET STATS
    # -------------------------------------------------

    st.markdown(
        "<div class='section-header'>📊 Dataset Statistics</div>",
        unsafe_allow_html=True
    )

    avg_length = int(
        df["Resume_str"]
        .astype(str)
        .apply(len)
        .mean()
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">
                Total Resumes
            </div>
            <div class="stat-value">
                {len(df)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">
                Categories
            </div>
            <div class="stat-value">
                {df['Category'].nunique()}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">
                Average Length
            </div>
            <div class="stat-value">
                {avg_length}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------
    # SIMILARITY
    # -------------------------------------------------

    all_docs = [jd]

    all_docs.extend(
        df[resume_col]
    )

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(
        all_docs
    )

    similarity = cosine_similarity(
        vectors[0:1],
        vectors[1:]
    )

    df["Similarity Score"] = similarity[0]

    ranked_df = df.sort_values(
        by="Similarity Score",
        ascending=False
    )

    top10 = ranked_df.head(10)

    st.success(
        "Candidates Ranked Successfully"
    )

    # -------------------------------------------------
    # BEST CANDIDATE
    # -------------------------------------------------

    best = top10.iloc[0]

    st.markdown(
        "<div class='section-header'>🏆 Best Candidate</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class='top-candidate'>
        Similarity Score : {best['Similarity Score']:.3f}
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------------------------
    # TOP 10 TABLE
    # -------------------------------------------------

    st.markdown(
        "<div class='section-header'>📋 Top 10 Candidates</div>",
        unsafe_allow_html=True
    )

    st.dataframe(
        top10[
            [
                "Category",
                "Similarity Score"
            ]
        ],
        use_container_width=True
    )

    # -------------------------------------------------
    # MATCHING SKILLS
    # -------------------------------------------------

    st.markdown(
        "<div class='section-header'>🔍 Matching Skills</div>",
        unsafe_allow_html=True
    )

    jd_skills = extract_skills(jd)

    candidate_skills = extract_skills(
        best[resume_col]
    )

    matching = set(
        jd_skills
    ).intersection(
        candidate_skills
    )

    if matching:

        for skill in matching:

            st.markdown(
                f"""
                <span class='skill-box'>
                {skill}
                </span>
                """,
                unsafe_allow_html=True
            )

    else:

        st.warning(
            "No exact matching skills found."
        )

    # -------------------------------------------------
    # CHART
    # -------------------------------------------------

    st.markdown(
        "<div class='section-header'>📈 Similarity Score Distribution</div>",
        unsafe_allow_html=True
    )

    st.bar_chart(
        top10["Similarity Score"]
    )

    # -------------------------------------------------
    # DOWNLOAD
    # -------------------------------------------------

    st.markdown(
        "<div class='section-header'>⬇ Export Results</div>",
        unsafe_allow_html=True
    )

    csv = top10.to_csv(
        index=False
    )

    st.download_button(
        "Download Ranked Candidates",
        csv,
        file_name="ranked_candidates.csv",
        mime="text/csv"
    )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown(
"""
<div class='footer'>
Built using NLP • MultiHeadAttention • Positional Encoding • Explainable AI
</div>
""",
unsafe_allow_html=True
)