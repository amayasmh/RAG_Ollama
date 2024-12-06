import streamlit as st
import pandas as pd
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configurer la page Streamlit
st.set_page_config(page_title="Chatbot RAG", layout="wide")

# Charger et préparer les données en arrière-plan
@st.cache_resource
def setup_rag_system(file_path):
    df = pd.read_json(file_path, lines=True)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    descriptions = df['description'].dropna().tolist()
    chunks = []
    for description in descriptions:
        if isinstance(description, list):
            description = " ".join(description)
        elif not isinstance(description, str):
            continue
        chunks.extend(text_splitter.split_text(description))

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_numpy=False)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    text_embeddings = [(chunks[i], embeddings[i].tolist()) for i in range(len(chunks))]
    faiss_index = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embedding_model
    )
    retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    return retriever

file_path = '/home/amayas/Bureau/M2 Data/RAG_Ollama/data/meta.jsonl'
retriever = setup_rag_system(file_path)

# Configurer le modèle LLM (Ollama)
llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434", temperature=0.8, top_p=0.98)

# Définir un prompt structuré
prompt_template = """
Vous êtes un assistant intelligent destiné à répondre aux questions des utilisateurs en utilisant uniquement les informations provenant des documents pertinents fournis. Voici vos instructions :
- Si la question est une salutation (par exemple, "Bonjour", "Salut", "Hello"), répondez simplement : "Bonjour ! Comment puis-je vous aider ?"
- Limitez vos réponses exclusivement aux informations issues des documents fournis ci-dessous.
- Si vous ne trouvez pas d'information pertinente dans les documents, répondez explicitement : "Je ne sais pas."
- Soyez précis et concis dans vos réponses.

Voici les documents pertinents :
{context}

Question de l'utilisateur :
{question}
"""
PROMPT = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": PROMPT}
)

# Interface utilisateur Streamlit
st.title("Chatbot RAG")

# Initialiser l'état de la session pour stocker l'historique de la conversation
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Afficher l'historique des messages
chat_history = st.empty()  # Conteneur pour rafraîchir dynamiquement l'historique
with chat_history.container():
    for message in st.session_state.conversation:
        # Style des messages avec l'heure à droite
        st.markdown(
            f"""
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;'>
                <div>
                    <b>Vous :</b> <b>{message['user']}</b>
                </div>
                <div style='font-size: 0.8em; color: gray;'>{message['time']}</div>
            </div>
            <div style='margin-left: 20px; margin-bottom: 10px;'><b>Assistant :</b> {message['bot']}</div>
            """,
            unsafe_allow_html=True
        )
    st.write("<hr style='margin-top: 10px;'>", unsafe_allow_html=True)

# Formulaire pour soumettre une question avec "Entrée"
with st.form("question_form", clear_on_submit=True):
    query = st.text_input("Posez votre question ici :", "", key="input_query", label_visibility="collapsed")
    submitted = st.form_submit_button("Envoyer")

# Gestion de la soumission de la question
if submitted:
    if query.strip() == "":
        st.error("Veuillez entrer une question.")
    else:
        with st.spinner("Recherche en cours..."):
            response = qa_chain({"query": query})
            bot_response = response["result"]

            # Ajouter la question et la réponse à l'historique
            st.session_state.conversation.append({
                "time": datetime.now().strftime("%H:%M"),
                "user": query,
                "bot": bot_response
            })

        # Rafraîchir dynamiquement l'historique des messages
        chat_history.empty()
        with chat_history.container():
            for message in st.session_state.conversation:
                # Style des messages avec l'heure à droite
                st.markdown(
                    f"""
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;'>
                        <div>
                            <b>Vous :</b> <b>{message['user']}</b>
                        </div>
                        <div style='font-size: 0.8em; color: gray;'>{message['time']}</div>
                    </div>
                    <div style='margin-left: 20px; margin-bottom: 10px;'><b>Assistant :</b> {message['bot']}</div>
                    """,
                    unsafe_allow_html=True
                )
            st.write("<hr style='margin-top: 10px;'>", unsafe_allow_html=True)
