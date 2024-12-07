# Rapport du projet : RAG_Ollama

## Introduction

Le projet **RAG_Ollama** a pour objectif de développer un chatbot interactif basé sur la génération augmentée par récupération (**Retrieval Augmented Generation**, RAG). Ce système combine une base vectorielle FAISS pour la recherche documentaire et un modèle de langage pré-entraîné (**LLM**) via **Ollama** pour générer des réponses. Le chatbot répond aux questions des utilisateurs en s'appuyant uniquement sur une base de connaissances locale.

---

## Objectifs

- Mettre en place un pipeline complet de génération augmentée par récupération (**RAG**).
- Répondre aux requêtes des utilisateurs uniquement à partir de documents fournis.
- Créer une interface utilisateur interactive en utilisant **Streamlit**.
- Fournir des réponses rapides et précises grâce à un modèle LLM (`llama3.2`).

---

## Fonctionnalités principales

1. **Recherche documentaire** :
   - Recherche des passages pertinents dans une base vectorielle construite avec **FAISS**.
2. **Modèle LLM (Ollama)** :
   - Utilisation de `llama3.2` pour générer des réponses naturelles basées sur les données récupérées.
3. **Interface utilisateur** :
   - Un chatbot interactif avec un historique des conversations et un champ d'entrée pour les requêtes.
   - Envoi des questions par clic ou via la touche `Entrée`.
4. **Prompt structuré** :
   - Guide le modèle LLM pour fournir des réponses précises et éviter les généralisations hors contexte.

---

## Prérequis

- **Python** : version 3.12.8
- **Ollama CLI** : pour exécuter le serveur et interagir avec le modèle `llama3.2`.
- **Fichiers nécessaires** :
  - `meta.jsonl` : fichier contenant les descriptions pour la base vectorielle.
  - Dossier `faiss_index` : contient les fichiers `index.faiss` et `index.pkl` pour la base vectorielle FAISS.

---

## Installation et configuration

### Étape 1 : Cloner le projet

Clonez le projet à partir du dépôt Git :

```bash
git clone https://github.com/votre-utilisateur/RAG_Ollama.git
cd RAG_Ollama
```

Étape 2 : Créer un environnement virtuel

Créez un environnement Python virtuel pour isoler les dépendances :

```bash
python3 -m venv env
source env/bin/activate  # Linux/macOS
# .\\env\\Scripts\\activate  # Windows
```

### Étape 3 : Installer les dépendances

Installez toutes les bibliothèques nécessaires à partir du fichier requirements.txt :

```bash
pip install -r requirements.txt
```

### Étape 4 : Configurer Ollama

Installer Ollama CLI : Suivez les instructions disponibles sur Ollama.
Télécharger le modèle llama3.2 :

```bash
ollama pull llama3.2
```

Lancer le serveur Ollama :

```bash
ollama serve
```

Vérifiez que le serveur est actif sur http://localhost:11434.

### Étape 5 : Préparer la base vectorielle FAISS

Assurez-vous que le dossier data/faiss_index contient les fichiers suivants :
index.faiss
index.pkl

Si ces fichiers ne sont pas présents, ils seront générés automatiquement lors de la première exécution si le fichier meta.jsonl est correctement configuré.

### Lancement du projet

Une fois toutes les étapes de configuration terminées, exécutez le projet avec Streamlit :

```bash
streamlit run src/Ollama.py
```

Cela ouvrira une interface web accessible à l'adresse http://localhost:8501.

Fonctionnement de l'interface

Historique des messages :
    L'historique des messages est affiché dans un format clair avec l'heure d'envoi visible à droite.
Saisie utilisateur :
    Posez des questions dans le champ de saisie.
    Vous pouvez envoyer une question en appuyant sur Entrée ou sur le bouton Envoyer.
Réponses intelligentes :
    Le chatbot répond immédiatement avec des informations pertinentes tirées des documents.
    Pour des salutations comme "Bonjour", le bot répondra avec "Bonjour ! Comment puis-je vous aider ?".

### Structure du projet

RAG_Ollama/
├── data/
│   ├── meta.jsonl          # Données brutes pour créer la base vectorielle
│   ├── faiss_index/
│       ├── index.faiss     # Index FAISS pour la recherche
│       ├── index.pkl       # Métadonnées pour FAISS
├── src/
│   ├── Ollama.py           # Script principal pour exécuter le chatbot
├── requirements.txt        # Fichier des dépendances
├── README.md               # Documentation du projet

### Auteurs:

Aghiles SAGHIR
Amayas MAHMOUDI
