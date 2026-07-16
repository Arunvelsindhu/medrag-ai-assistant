# 🩺 MedRAG AI Assistant

AI-powered medical report analyzer built using:

* Streamlit
* LangChain
* FAISS
* Google Gemini (gemini-flash-latest + gemini-embedding-001)

---

## 🚀 Features

* Multi PDF Upload
* AI Medical Summary
* AI Medical Chatbot
* Download Summary PDF
* Vector Search using FAISS

---

## 🛠 Technologies Used

* Python
* Streamlit
* LangChain
* FAISS
* Google Generative AI (Gemini) — chat + embeddings

---

## 📂 Project Workflow

1. Upload medical reports (PDFs)
2. Extract text from PDFs
3. Split text into chunks
4. Create vector embeddings (via Gemini's embedding API)
5. Store vectors using FAISS
6. Ask medical questions
7. Generate AI-powered answers

---

## 🔑 Setup

You need a Google AI Studio API key: https://aistudio.google.com/apikey

### Run locally

1. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

2. Create `.streamlit/secrets.toml` in the project root (this file should
   **not** be committed — it's already covered by `.gitignore`):

   ```toml
   GOOGLE_API_KEY = "your-api-key-here"
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

### Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (make sure `.streamlit/secrets.toml` is **not**
   committed — never commit real API keys).
2. On https://share.streamlit.io, create a new app pointing at this repo
   and `app.py`.
3. In the app's **Settings → Secrets**, add:

   ```toml
   GOOGLE_API_KEY = "your-api-key-here"
   ```

4. Deploy. No local-model downloads are required — embeddings and chat
   both run through the Gemini API, so the app stays within the free
   tier's memory limits.

---

## 👨‍💻 Author

Arunvel
