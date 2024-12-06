import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Charger le fichier meta.jsonl
file_path = '/home/amayas/Bureau/M2 Data/RAG_Ollama/data/meta.jsonl'
df = pd.read_json(file_path, lines=True)

# Afficher les 5 premières lignes du dataframe
print(df.head())

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
print(f"Nombre total de morceaux : {len(chunks)}")
print(f"Exemple de morceau : {chunks[:5]}")
