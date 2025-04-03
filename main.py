import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def load_json(json_path):
    """Charge les informations extraites des images."""
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f" Erreur : Le fichier {json_path} est introuvable.")
        return None
    except Exception as e:
        print(f" Erreur lors de la lecture du fichier JSON : {e}")
        return None

def load_csv(csv_path):
    """Charge le relevé bancaire au format CSV."""
    try:
        return pd.read_csv(csv_path, encoding="utf-8")
    except FileNotFoundError:
        print(f" Erreur : Le fichier {csv_path} est introuvable.")
        return None
    except Exception as e:
        print(f" Erreur lors de la lecture du fichier CSV : {e}")
        return None

def preprocess_data(df):
    """Nettoie et normalise les données du relevé bancaire."""
    required_columns = ['vendor', 'amount']
    for col in required_columns:
        if col not in df.columns:
            print(f" Avertissement : La colonne '{col}' est manquante dans le CSV.")
            return None

    df['vendor'] = df['vendor'].astype(str).str.lower().str.strip()
    df = df.dropna(subset=['amount', 'vendor'])
    return df

def match_transaction_with_receipts(bank_row, invoices_data, model, tolerance=0.05):
    """Trouve l'image qui correspond le mieux à une transaction bancaire."""
    transaction_amount = float(bank_row["amount"])
    transaction_vendor = bank_row["vendor"].lower().strip()
    
    # Filtrer les factures par montant
    potential_matches = [
        invoice for invoice in invoices_data
        if (float(invoice["Informations"]["montant"]) >= transaction_amount - tolerance) and
           (float(invoice["Informations"]["montant"]) <= transaction_amount + tolerance)
    ]
    
    print(f" Transaction : {transaction_vendor} | {transaction_amount} €")
    print(f" Factures possibles : {[p['Fichier'] for p in potential_matches]}")
    
    if potential_matches:
        # Embeddings des noms des factures (sans adresse)
        invoice_texts = [p["Informations"]["nom_commerce"] for p in potential_matches]
        invoice_embeddings = model.encode(invoice_texts)
        
        # Embedding de la transaction bancaire
        transaction_embedding = model.encode([transaction_vendor])
        
        # Calculer la similarité cosinus
        similarities = cosine_similarity(transaction_embedding, invoice_embeddings).flatten()
        
        print(f" Scores de similarité : {similarities}")
        
        if np.max(similarities) > 0.3:  # Seuil ajusté pour plus de flexibilité
            best_match_index = np.argmax(similarities)
            best_match = potential_matches[best_match_index]
            
            print(f"Meilleure correspondance trouvée : {best_match['Fichier']} (Score {np.max(similarities):.2f})")
            
            return {
                "Transaction": bank_row.to_dict(),
                "Image": best_match["Fichier"],
                "Score": np.max(similarities)
            }
    
    print(f" Aucune correspondance trouvée pour cette transaction.")
    return None

def match_bank_transactions(bank_df, invoices_data, model):
    """Associe chaque ligne du relevé bancaire avec une image."""
    matches = []
    
    for _, row in bank_df.iterrows():
        match = match_transaction_with_receipts(row, invoices_data, model)
        if match is not None:
            matches.append(match)
    
    return matches

def main():
    # Chemins des fichiers
    json_path = "extracted_invoices.json"
    csv_path = "bank_statements-20250331T125655Z-001/bank_statements/releve_01.csv"

    # Charger les fichiers
    invoices_data = load_json(json_path)
    bank_df = load_csv(csv_path)

    if invoices_data is not None and bank_df is not None:
        # Charger le modèle d'embedding
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        # Prétraiter les données bancaires
        bank_df = preprocess_data(bank_df)

        if bank_df is not None:  # Vérifier si les données ont été nettoyées correctement
            # Associer les transactions aux factures
            matches = match_bank_transactions(bank_df, invoices_data, model)

            # Afficher les correspondances
            if matches:
                for match in matches:
                    print(f" Transaction: {match['Transaction']}")
                    print(f" Image: {match['Image']} (Score {match['Score']:.2f})")
                    print("-" * 50)
            else:
                print("❌ Aucune correspondance trouvée.")

if __name__ == "__main__":
    main()
