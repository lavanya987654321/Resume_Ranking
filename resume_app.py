import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    
    
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return cosine_similarities

# Function to set background color and text color
def set_styles():
    st.markdown(
        """
        <style>
            /* White background and black text */
            .stApp {
                background-color: white !important;
                color: black !important;
            }

            /* Force left alignment for all content */
            .block-container {
                text-align: left !important;
                max-width: 80%;
                margin: auto;
            }

            /* Set all text elements to black */
            .stMarkdown, .stTitle, .stHeader, .stSubheader, .stText, .stCaption, 
            .stCode, .stCheckbox, .stRadio, .stSelectbox, .stMultiselect, .stButton, 
            .stFileUploader, .stAlert div { 
                text-align: left !important; 
                color: black !important;
            }

            /* Set sidebar background to white */
            .css-1aumxhk {
                background-color: white !important;
            }

            .stTextArea label{
                color:black !important; }
           
            /* Ensure file uploader label text is black */
            .stFileUploader label {
                color: black !important;
            }

            .st.Title lable{
            text-align: center;
            }

        </style>
        """,
        unsafe_allow_html=True
    )

# Apply styles globally
set_styles()

# Streamlit Multi-Page Navigation
st.sidebar.markdown("<h3 style='color: white;'>Navigation</h3>", unsafe_allow_html=True)
page = st.sidebar.radio("Go to", ["Home", "Job Description"])


if page == "Home":
    st.title("AI Resume Screening & Candidate Ranking System")
    
    # Left-aligned markdown using simple text formatting
    st.markdown("""
        **Welcome to the AI-powered Resume Screening & Ranking System!** 
        
        This system helps recruiters and HR professionals quickly analyze and rank resumes based on job descriptions.  
        Automate candidate shortlisting with AI-driven relevance scoring for efficient hiring.  

        ### How It Works
        - Enter the job description on the 'Job Description' page.  
        - Upload resumes in PDF format on the 'Resume Screening' page.  
        - The system will analyze and rank resumes based on relevance.  
        - View ranked resumes to find the best candidates quickly.  

        ### Why Use This App?
        ✅ Automates resume screening using AI  
        ✅ Saves time for recruiters and HR professionals  
        ✅ Provides objective and data-driven ranking of candidates  
        ✅ Ensures fair evaluation of all applicants  
        ✅ Helps organizations find the best talent efficiently  

        ### Key Features
        🔹 AI-powered ranking based on job relevance  
        🔹 Supports multiple resume uploads in PDF format  
        🔹 User-friendly interface with easy navigation  
        🔹 Secure and efficient processing  
        🔹 Works with diverse job descriptions and resume formats  
    """, unsafe_allow_html=True)

elif page == "Job Description":
    st.title("Enter Job Description")
    job_description = st.text_area("Enter Job Description")
    st.session_state["job_description"] = job_description

    st.title("Upload and Rank Resumes")
    uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

    job_description = st.session_state.get("job_description", "")
    if uploaded_files and job_description:
        resumes = [extract_text(file) for file in uploaded_files]
        scores = rank_resumes(job_description, resumes)

        results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
        results = results.sort_values(by="Score", ascending=False)

        st.header("Resume Ranking Results")
        st.write(results)
    elif not job_description:
        st.warning("Please enter a job description first in the 'Job Description' page.")
