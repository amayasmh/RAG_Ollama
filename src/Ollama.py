import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

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
print(f"Exemple d'embedding : {embeddings[0]}")

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
retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Exemple de requête utilisateur
query = "extratereste" # Une requête d'exemple
results = retriever.get_relevant_documents(query)  # Récupérer les documents pertinents

# Afficher les résultats de récupération
print("Résultats de recherche :")
for idx, doc in enumerate(results):
    print(f"\nRésultat {idx + 1} :")
    print(f"Texte : {doc.page_content}")
    print(f"Métadonnées : {doc.metadata}")