import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaLLM  # Import corrigé
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# Charger le fichier meta.jsonl
file_path = '/home/amayas/Bureau/M2 Data/RAG_Ollama/data/meta.jsonl'
df = pd.read_json(file_path, lines=True)

# Afficher les 5 premières lignes du dataframe
#print(df.head())

# Instancier le text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,  # Taille des morceaux
    chunk_overlap=128  # Chevauchement
)

# Sélectionner uniquement la colonne contenant les descriptions de produits
descriptions = df['description'].dropna().tolist()

# Diviser chaque description en morceaux
chunks = []
for description in descriptions:
    if isinstance(description, list):  # Si description est une liste, la convertir en chaîne
        description = " ".join(description)
    elif not isinstance(description, str):  # Si ce n'est pas une chaîne, ignorer
        continue

    # Diviser les descriptions valides en morceaux
    chunks.extend(text_splitter.split_text(description))

# Afficher quelques morceaux pour vérifier
#print(f"Nombre total de morceaux : {len(chunks)}")
#print(f"Exemple de morceau : {chunks[:5]}")

# Charger un modèle d’embedding léger
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Générer des embeddings
embeddings = model.encode(chunks, convert_to_numpy=False)

# Vérifier un exemple
#print(f"Exemple d'embedding : {embeddings[0]}")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Créer une liste combinée de textes et d'embeddings
text_embeddings = [(chunks[i], embeddings[i].tolist()) for i in range(len(chunks))]


# Créer l'index FAISS
faiss_index = FAISS.from_embeddings(
    text_embeddings=text_embeddings,
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)

# Sauvegarder l'index FAISS dans un fichier
faiss_index.save_local('/home/amayas/Bureau/M2 Data/RAG_Ollama/data/faiss_index')

# Charger l'index FAISS depuis le fichier (pour vérification ou utilisation future)
#faiss_index = FAISS.load_local(
#    folder_path='/home/amayas/Bureau/M2 Data/RAG_Ollama/data/faiss_index',
#    embeddings=text_embeddings[embeddings[0].tolist()],
#    allow_dangerous_deserialization=True  # Activer la désérialisation
#)



# Configurer le système de récupération
retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Définir le modèle LLM (Ollama)
llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434",temperature = 0.8,top_p = 0.98)
# Définir un prompt structuré avec PromptTemplate
prompt_template = """
Vous êtes un assistant intelligent destiné à répondre aux questions des utilisateurs en utilisant uniquement les informations provenant des documents pertinents fournis. Voici vos instructions :
- Limitez vos réponses exclusivement aux informations issues des documents fournis ci-dessous.
- Si vous ne trouvez pas d'information pertinente, répondez explicitement : "Je ne sais pas."
- Fournissez les passages ou documents spécifiques d'où vous extrayez les informations.
- Ne générez pas d'informations sensibles, inappropriées ou incorrectes.
- Soyez précis et concis dans vos réponses.

Voici les documents pertinents :
{context}

Question de l'utilisateur :
{question}
"""
PROMPT = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
print("Prompt généré :", PROMPT)

# Configurer une chaîne RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Combinaison des documents et question
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# Exemple de requête utilisateur
query = "tell me about OnePlus 6T"  # Exemple de question
response = qa_chain({"query": query})

# Afficher la réponse
print("\nRéponse générée :")
print(response["result"])
