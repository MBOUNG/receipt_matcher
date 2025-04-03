import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import logging

# Configuration de la journalisation
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_json(json_path):
    """Charge les données JSON à partir d'un fichier."""
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        logging.error(f"Fichier introuvable: {json_path}")
    except Exception as e:
        logging.error(f"Erreur lors de la lecture du fichier JSON: {e}")
    return None

def load_csv(csv_paths):
    """Charge les données CSV à partir de plusieurs fichiers."""
    dataframes = []
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
            dataframes.append(df)
        except FileNotFoundError:
            logging.error(f"Fichier introuvable: {csv_path}")
        except Exception as e:
            logging.error(f"Erreur lors de la lecture du fichier CSV: {e}")
    return dataframes

def preprocess_data(df):
    """Nettoie et normalise les données du relevé bancaire."""
    required_columns = ['vendor', 'amount']
    if not all(col in df.columns for col in required_columns):
        logging.warning("Colonnes requises manquantes dans le CSV.")
        return None

    df['vendor'] = df['vendor'].astype(str).str.lower().str.strip()
    df['amount'] = df['amount'].str.replace(',', '.').astype(float)  # Remplacer les virgules par des points
    df = df.dropna(subset=['amount', 'vendor'])
    df = df.drop_duplicates()  # Supprimer les doublons
    return df

def match_transaction_with_receipts(bank_row, invoices_data, model, tolerance=0.05):
    """Trouve la facture correspondant le mieux à une transaction bancaire."""
    transaction_amount = bank_row["amount"]
    transaction_vendor = bank_row["vendor"]

    # Filtrer les factures par montant
    potential_matches = [
        invoice for invoice in invoices_data
        if abs(float(invoice["Informations"]["montant"]) - transaction_amount) <= tolerance
    ]

    logging.info(f"Transaction: {transaction_vendor} | {transaction_amount} €")
    logging.info(f"Factures potentielles: {[p['Fichier'] for p in potential_matches]}")

    if potential_matches:
        invoice_texts = [p["Informations"]["nom_commerce"] for p in potential_matches]
        invoice_embeddings = model.encode(invoice_texts, convert_to_tensor=True)
        transaction_embedding = model.encode([transaction_vendor], convert_to_tensor=True)

        similarities = cosine_similarity(transaction_embedding, invoice_embeddings).flatten()
        logging.info(f"Scores de similarité: {similarities}")

        best_match_index = np.argmax(similarities)
        if similarities[best_match_index] > 0.3:
            best_match = potential_matches[best_match_index]
            logging.info(f"Meilleure correspondance trouvée: {best_match['Fichier']} (Score {similarities[best_match_index]:.2f})")
            return {
                "Transaction": bank_row.to_dict(),
                "Image": best_match["Fichier"],
                "Score": similarities[best_match_index]
            }

    logging.info("Aucune correspondance trouvée pour cette transaction.")
    return None

def match_bank_transactions(bank_dfs, invoices_data, model):
    """Associe chaque ligne de chaque relevé bancaire avec une image."""
    all_matches = []
    for bank_df in bank_dfs:
        if bank_df is not None and not bank_df.empty:
            matches = []
            for _, row in bank_df.iterrows():
                match = match_transaction_with_receipts(row, invoices_data, model)
                if match is not None:
                    matches.append(match)
            all_matches.extend(matches)
    return all_matches

def main():
    json_path = "extracted_invoices.json"
    csv_paths = [
        "bank_statements-20250331T125655Z-001/bank_statements/releve_01.csv",
        "bank_statements-20250331T125655Z-001/bank_statements/releve_02.csv",
        "bank_statements-20250331T125655Z-001/bank_statements/releve_03.csv",
        "bank_statements-20250331T125655Z-001/bank_statements/releve_04.csv",
        "bank_statements-20250331T125655Z-001/bank_statements/releve_05.csv",
        "bank_statements-20250331T125655Z-001/bank_statements/releve_06.csv"
    ]

    invoices_data = load_json(json_path)
    bank_dfs = load_csv(csv_paths)

    if invoices_data is not None and any(df is not None and not df.empty for df in bank_dfs):
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        preprocessed_dfs = [preprocess_data(df) for df in bank_dfs if df is not None]

        if any(df is not None for df in preprocessed_dfs):
            matches = match_bank_transactions(preprocessed_dfs, invoices_data, model)

            if matches:
                for match in matches:
                    logging.info(f"Transaction: {match['Transaction']}")
                    logging.info(f"Image: {match['Image']} (Score {match['Score']:.2f})")
                    logging.info("-" * 50)
            else:
                logging.info("Aucune correspondance trouvée.")

if __name__ == "__main__":
    main()
