# utils/parser.py (v3)
import pandas as pd
import io
import streamlit as st
import traceback

def parse_bank_statement(uploaded_file):
    """
    Analyse le CSV, nettoie colonnes, convertit montant et date.
    """
    if uploaded_file is None: return None

    try:
        file_content = uploaded_file.getvalue().decode("utf-8")
        stringio = io.StringIO(file_content)
        df = pd.read_csv(stringio, sep=None, engine='python', on_bad_lines='warn')
        st.success(f"Fichier CSV '{uploaded_file.name}' lu.")

        original_columns = df.columns.tolist()
        df.columns = df.columns.str.strip().str.lower()
        cleaned_columns = df.columns.tolist()
        print(f"Colonnes originales: {original_columns}")
        print(f"Colonnes nettoyées: {cleaned_columns}")

        # --- Nettoyage/Conversion Montant ('amount') ---
        if 'amount' in df.columns:
            print("Nettoyage de la colonne 'amount'...")
            original_non_numeric = pd.to_numeric(df['amount'], errors='coerce').isna().sum()
            if pd.api.types.is_string_dtype(df['amount']):
                 df['amount_cleaned'] = df['amount'].str.replace(',', '.', regex=False)
            else:
                 df['amount_cleaned'] = df['amount']
            df['amount'] = pd.to_numeric(df['amount_cleaned'], errors='coerce')
            df.drop(columns=['amount_cleaned'], inplace=True)
            final_non_numeric = df['amount'].isna().sum()
            if final_non_numeric > original_non_numeric:
                 print(f"Avertissement: {final_non_numeric - original_non_numeric} montants n'ont pas pu être convertis.")
                 st.warning(f"{final_non_numeric - original_non_numeric} montants invalides dans CSV.")
        else:
            st.error("Colonne 'amount' essentielle manquante.")
            return None

        # --- Conversion Date ('date') ---
        if 'date' in df.columns:
             print("Conversion de la colonne 'date'...")
             try:
                 # Tenter plusieurs formats communs, 'coerce' met NaT si échec
                 # dayfirst=True est important pour les formats français JJ/MM/AAAA
                 df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True, infer_datetime_format=True)
                 date_conversion_errors = df['date'].isna().sum()
                 if date_conversion_errors > 0:
                      st.warning(f"{date_conversion_errors} dates n'ont pas pu être converties et seront ignorées.")
                 # Optionnel: supprimer lignes avec date invalide
                 # df.dropna(subset=['date'], inplace=True)
             except Exception as e:
                  st.error(f"Erreur lors de la conversion des dates: {e}")
                  # Ne pas retourner None ici, mais la colonne date peut contenir des NaT
        else:
             st.warning("Colonne 'date' non trouvée. Le filtre par date ne sera pas appliqué.")

        # Vérifier présence colonne description ('vendor' ou 'source')
        if 'vendor' not in df.columns and 'source' not in df.columns:
             st.warning("Aucune colonne de description ('vendor' ou 'source') trouvée.")

        return df

    except Exception as e:
        st.error(f"Erreur inattendue lors de la lecture/préparation du CSV : {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None
