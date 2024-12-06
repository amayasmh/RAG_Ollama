import pandas as pd

# Charger le fichier meta.jsonl
file_path = '/home/amayas/Bureau/M2 Data/RAG_Ollama/data/meta.jsonl'
df = pd.read_json(file_path, lines=True)

# Afficher les 5 premières lignes du dataframe
print(df.head())