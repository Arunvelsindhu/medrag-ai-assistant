import streamlit as st
from pypdf import PdfReader
from fpdf import FPDF
 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
 
# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
 
st.set_page_config(
    page_title="MedRAG AI",
    page_icon="🩺",
    layout="wide"
)
 
# --------------------------------------------------
# API KEY CHECK
# --------------------------------------------------
# On Streamlit Cloud this must be set under App settings -> Secrets as:
#   GOOGLE_API_KEY = "your-key-here"
# Locally, put the same line in .streamlit/secrets.toml
 
if "GOOGLE_API_KEY" not in st.secrets:
    st.error(
        "GOOGLE_API_KEY is not set. Add it in Streamlit Cloud under "
        "App settings -> Secrets (or in .streamlit/secrets.toml locally)."
    )
    st.stop()
 
# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
 
with st.sidebar:
 
    st.title("🩺 MedRAG AI")
 
    st.markdown("""
    ### Features
 
    - Multi PDF Upload
    - Medical Report Analysis
    - AI Summary
    - RAG Chatbot
    - PDF Export
    - Streamlit Cloud Ready
    """)
 
# --------------------------------------------------
# TITLE
# --------------------------------------------------
 
st.title("🩺 MedRAG AI Assistant")
 
st.caption("AI Powered Medical Report Analyzer")
 
# --------------------------------------------------
# LOAD GEMINI (chat model + embeddings)
# --------------------------------------------------
# NOTE: gemini-2.0-flash has been shut down by Google. Using
# gemini-flash-latest keeps this pointed at the current stable
# flash model without needing manual updates every few months.
 
 
@st.cache_resource
def load_llm():
 
    return ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0.2
    )
 
 
@st.cache_resource
def load_embeddings():
    # Using Google's hosted embedding model instead of a local
    # HuggingFace/sentence-transformers model. This removes the
    # torch/transformers/sentence-transformers dependencies (several GB),
    # which were almost certainly why deployment to Streamlit Cloud
    # failed even though it ran fine locally.
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=st.secrets["GOOGLE_API_KEY"],
    )
 
 
try:
    llm = load_llm()
    embeddings = load_embeddings()
except Exception as e:
    st.error(f"Failed to initialize Gemini: {e}")
    st.stop()
 
 
def extract_text(content):
    """
    Newer Gemini models can return response.content as either a plain
    string, or a list of content blocks like:
        [{"type": "text", "text": "...", "extras": {...}}]
    This pulls the actual text out regardless of which shape comes back.
    """
    if isinstance(content, str):
        return content
 
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
 
    return str(content)
 
# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
 
uploaded_files = st.file_uploader(
    "Upload Medical Reports",
    type=["pdf"],
    accept_multiple_files=True
)
 
# --------------------------------------------------
# PROCESS PDFS
# --------------------------------------------------
 
if uploaded_files:
 
    # Build a simple signature for the current set of uploaded files so we
    # only re-process (re-read PDFs, re-chunk, re-embed) when the files
    # actually change — not on every rerun. Streamlit reruns the whole
    # script on every interaction, including sending a chat message, so
    # without this the app was silently re-reading every PDF and
    # re-calling the embeddings API for every chunk on every single chat
    # message, which is slow and can hit API rate limits — that's why chat
    # looked like it wasn't responding.
    file_signature = tuple((f.name, f.size) for f in uploaded_files)
 
    if st.session_state.get("file_signature") != file_signature:
 
        all_text = ""
 
        with st.spinner("Reading PDFs..."):
 
            for file in uploaded_files:
 
                try:
 
                    pdf_reader = PdfReader(file)
 
                    for page in pdf_reader.pages:
 
                        page_text = page.extract_text()
 
                        if page_text:
                            all_text += page_text + "\n"
 
                except Exception as e:
 
                    st.error(f"Error reading {file.name}: {e}")
 
        if len(all_text.strip()) == 0:
 
            st.error("No readable text found.")
            st.stop()
 
        # --------------------------------------------------
        # CHUNKING
        # --------------------------------------------------
 
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
 
        chunks = splitter.split_text(all_text)
 
        # --------------------------------------------------
        # VECTOR DATABASE
        # --------------------------------------------------
 
        with st.spinner("Creating Vector Database..."):
 
            try:
                vectorstore = FAISS.from_texts(
                    chunks,
                    embedding=embeddings
                )
            except Exception as e:
                st.error(f"Failed to build vector database: {e}")
                st.stop()
 
        # Cache everything for this set of files so later reruns (like
        # sending a chat message) reuse it instead of rebuilding it.
        st.session_state.file_signature = file_signature
        st.session_state.all_text = all_text
        st.session_state.num_chunks = len(chunks)
        st.session_state.vectorstore = vectorstore
        st.session_state.messages = []  # new files -> fresh chat
 
    all_text = st.session_state.all_text
    vectorstore = st.session_state.vectorstore
 
    # --------------------------------------------------
    # VIEW TEXT
    # --------------------------------------------------
 
    with st.expander("Extracted Text"):
 
        st.write(all_text[:5000])
 
    st.success(f"Created {st.session_state.num_chunks} chunks")
 
    st.success("Knowledge Base Ready")
 
    # --------------------------------------------------
    # AI SUMMARY
    # --------------------------------------------------
 
    st.subheader("📄 AI Medical Summary")
 
    if st.button("Generate Summary"):
 
        with st.spinner("Generating Summary..."):
 
            summary_prompt = f"""
            You are a medical report assistant.
 
            Analyze the following reports and provide:
 
            1. Key Findings
            2. Important Medical Values
            3. Abnormal Results
            4. Recommendations
 
            Report:
 
            {all_text[:12000]}
            """
 
            try:
                summary = llm.invoke(summary_prompt)
                summary_text = extract_text(summary.content)
            except Exception as e:
                st.error(f"Failed to generate summary: {e}")
                st.stop()
 
            st.success("Summary Generated")
 
            st.write(summary_text)
 
            # ---------------- PDF Export ----------------
 
            pdf = FPDF()
 
            pdf.add_page()
 
            pdf.set_font("Arial", size=12)
 
            pdf.multi_cell(
                0,
                10,
                summary_text.encode(
                    "latin-1",
                    "replace"
                ).decode("latin-1")
            )
 
            pdf.output("medical_summary.pdf")
 
            with open(
                "medical_summary.pdf",
                "rb"
            ) as f:
 
                st.download_button(
                    "⬇ Download Summary PDF",
                    data=f,
                    file_name="medical_summary.pdf",
                    mime="application/pdf"
                )
 
    # --------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------
 
    st.subheader("💬 Medical AI Chat")
 
    if "messages" not in st.session_state:
 
        st.session_state.messages = []
 
    for msg in st.session_state.messages:
 
        with st.chat_message(msg["role"]):
 
            st.markdown(msg["content"])
 
    # --------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------
 
    question = st.chat_input(
        "Ask anything about your reports..."
    )
 
    if question:
 
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )
 
        with st.chat_message("user"):
 
            st.markdown(question)
 
        # ----------------------------------------------
        # RAG RETRIEVAL
        # ----------------------------------------------
 
        docs = vectorstore.similarity_search(
            question,
            k=4
        )
 
        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )
 
        prompt = f"""
        You are a medical AI assistant.
 
        Use ONLY the context below.
 
        Context:
        {context}
 
        Question:
        {question}
 
        Answer clearly.
        """
 
        with st.spinner("Generating Answer..."):
 
            try:
                answer = llm.invoke(prompt)
                response = extract_text(answer.content)
            except Exception as e:
                response = f"Failed to generate an answer: {e}"
 
        with st.chat_message("assistant"):
 
            st.markdown(response)
 
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )
 
        with st.expander("Retrieved Context"):
 
            st.write(context)
 
# --------------------------------------------------
# FOOTER
# --------------------------------------------------
 
st.divider()
 
st.markdown(
    """
    <center>
    MedRAG AI Assistant <br>
    Built with Streamlit + Gemini + FAISS
    </center>
    """,
    unsafe_allow_html=True
)