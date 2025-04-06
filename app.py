# app.py (Version 2.0 - Pixtral OCR + Matching 2 Étapes)
import streamlit as st

# --- Configuration Streamlit ---
st.set_page_config(page_title="Matching Factures", layout="wide", initial_sidebar_state="collapsed")

import pandas as pd
# Assurez-vous que utils/parser.py (v3) existe
try:
    from utils.parser import parse_bank_statement
except ImportError:
    st.error("Erreur: Le fichier utils/parser.py (v3) est manquant ou invalide.")
    def parse_bank_statement(uploaded_file): st.error("Fonction parse_bank_statement non chargée."); return None

# LLM Clients n'est plus utilisé pour le matching ici
# try:
#     from utils.llm_clients import get_mistral_client # On a besoin du client pour l'OCR
# except ImportError:
#      st.error("Erreur: Le fichier utils/llm_clients.py est manquant ou invalide.")
#      def get_mistral_client(): st.error("Fonction get_mistral_client non chargée."); return None

from dotenv import load_dotenv
import os
import datetime
import io
import base64
import json
from PIL import Image
import re
import time
import requests # Gardé au cas où

# Imports pour le matching sémantique
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Imports pour l'appel API Mistral (Pixtral)
from mistralai.client import MistralClient # Utilisation directe ici
# Pas besoin d'importer MistralAPIException avec la v0.4.2, elle pourrait être différente
# On utilisera une gestion d'erreur plus générique si besoin.
import traceback

# Chargement des variables d'environnement (.env)
load_dotenv()

# --- Le reste du script commence ici ---
st.title("🧾 Matching Automatique Factures / Relevé Bancaire")
st.markdown("Chargez votre relevé bancaire (CSV) et vos factures (images) pour lancer le processus.")

# --- Récupération et Vérification des clés API ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") # Requis pour Pixtral OCR

# --- Fonctions de Cache pour Clients/Modèles ---
@st.cache_resource
def get_mistral_client_ocr():
    """Initialise et retourne le client Mistral (pour OCR)."""
    # Utilise MISTRAL_API_KEY
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        st.error("Clé API Mistral (MISTRAL_API_KEY) non trouvée dans .env.")
        return None
    print("--- Initialisation du client Mistral (pour OCR) ---")
    try:
        # Initialisation standard (devrait fonctionner avec v0.4.2)
        client = MistralClient(api_key=api_key)
        print("--- Client Mistral (pour OCR) initialisé ---")
        return client
    except Exception as e:
        st.error(f"Erreur init client Mistral : {e}")
        return None

@st.cache_resource
def get_sentence_transformer_model(model_name='paraphrase-MiniLM-L6-v2'):
    """Charge et retourne le modèle Sentence Transformer."""
    print(f"--- Chargement modèle Sentence Transformer : {model_name} ---")
    try:
        model = SentenceTransformer(model_name)
        print("--- Modèle Sentence Transformer chargé ---")
        return model
    except Exception as e:
        st.error(f"Erreur chargement modèle Sentence Transformer '{model_name}': {e}")
        return None

# Initialiser les clients/modèles nécessaires
mistral_client_ocr = get_mistral_client_ocr()
sentence_model = get_sentence_transformer_model()

# Afficher le statut
with st.expander("🔐 Statut des Services Externes", expanded=False):
    if mistral_client_ocr: st.success("✅ Client Mistral (OCR Pixtral) prêt.")
    else: st.warning("⚠️ Client Mistral (OCR Pixtral) non disponible.")
    if sentence_model: st.success("✅ Modèle Sentence Transformer (Matching) prêt.")
    else: st.warning("⚠️ Modèle Sentence Transformer (Matching) non disponible.")


# --- Upload des fichiers ---
st.header("1. 🔄 Chargement des Fichiers")
col1, col2 = st.columns(2)
with col1:
    uploaded_statement = st.file_uploader("**📄 Relevé Bancaire (CSV)**", type=["csv"], help="charger votre relevé bancaire.")
with col2:
    uploaded_invoices = st.file_uploader("**🖼️ Factures (images)**", type=["jpg", "jpeg", "png"], accept_multiple_files=True, help="Chargez une ou plusieurs factures.")

# --- Préparation des données ---
st.divider()
st.header("2. ⚙️ Préparation des Données")
statement_df = None
invoice_files_list = []

# Analyse du relevé (utilise parser_py_v3)
if uploaded_statement is not None:
    st.subheader("📑 Aperçu du Relevé Bancaire")
    with st.spinner("Lecture et nettoyage du fichier CSV..."):
        if 'parse_bank_statement' in globals():
             statement_df = parse_bank_statement(uploaded_statement)
        else:
             st.error("La fonction parse_bank_statement n'est pas disponible.")
             statement_df = None

    if statement_df is not None:
        st.write("Colonnes disponibles dans le relevé :", list(statement_df.columns))
        # Afficher aussi les types pour vérifier la date
        # st.write(statement_df.dtypes)
        st.dataframe(statement_df.head(), use_container_width=True)
        st.success(f"✅ {len(statement_df)} transactions prêtes.")
    elif uploaded_statement:
         st.error("Le fichier CSV n'a pas pu être traité correctement.")


# Aperçu des factures
if uploaded_invoices:
    st.subheader("🧾 Factures Chargées")
    invoice_files_list = uploaded_invoices
    invoice_names = [f.name for f in invoice_files_list]
    st.json(invoice_names, expanded=False)

# --- Fonction OCR avec Pixtral (JSON Mode) ---
def call_pixtral_ocr_json(image_bytes, client, image_media_type, model_name="pixtral-12b-2409", max_retries=3):
    """Appelle Pixtral pour extraire date, nom_commerce, montant en JSON."""
    if not client: st.error("Client Mistral non initialisé."); return None

    retries = 0
    while retries < max_retries:
        try:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            data_url = f"data:{image_media_type};base64,{base64_image}"

            # Prompt demandant le JSON (comme rap.py)
            prompt_text = ("Analyse cette facture et extrait les informations suivantes sous forme de JSON :\n"
                           "- Date de la facture\n"
                           "- Nom du commerce\n"
                           "- Montant total\n\n"
                           "Réponds uniquement avec un JSON structuré de cette manière :\n"
                           "{\n"
                           "  \"date\": \"YYYY-MM-DD\",\n"
                           "  \"nom_commerce\": \"Nom du commerce\",\n"
                           "  \"montant\": \"XX.XX\"\n" # Demander comme string pour flexibilité
                           "}\n"
                           "Ne renvoie aucun texte supplémentaire en dehors du JSON.")

            # Message API (Simple dictionnaire, OK pour mistralai 0.4.2 a priori)
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image_url", "image_url": data_url}]}]

            print(f"--- Appel API Pixtral OCR (Essai {retries+1}/{max_retries}) ---")
            # Utiliser client.chat_complete pour v0.4.2 ? Ou client.chat? A vérifier si erreur.
            # La doc 0.4.0 mentionne client.chat(...)
            chat_response = client.chat(
                model=model_name,
                messages=messages,
                # response_format n'existe peut-être pas en 0.4.2, le prompt doit suffire
                # response_format={"type": "json_object"}
            )
            print(f"--- Réponse API Pixtral OCR reçue ---")

            response_content = chat_response.choices[0].message.content
            # Essayer d'extraire le JSON même s'il y a du texte autour
            match_json = re.search(r'\{.*\}', response_content, re.DOTALL)
            if match_json:
                json_str = match_json.group(0)
                json_str_cleaned = json_str.replace(': "null"', ': null').replace(':null', ': null')
                extracted_data = json.loads(json_str_cleaned)
            else:
                 raise json.JSONDecodeError("Aucun JSON trouvé dans la réponse", response_content, 0)


            # Validation/Conversion Montant
            if extracted_data.get("montant") is not None:
                try:
                    montant_str = str(extracted_data["montant"]).replace(",",".")
                    extracted_data["montant"] = float(montant_str)
                except (ValueError, TypeError):
                    st.warning(f"Montant Pixtral ('{extracted_data.get('montant')}') invalide. Mis à null.")
                    extracted_data["montant"] = None
            else: extracted_data["montant"] = None

            # Validation/Conversion Date
            if extracted_data.get("date") is not None:
                 try:
                      # Essayer de convertir en datetime pour vérifier/normaliser
                      extracted_data["date_dt"] = pd.to_datetime(extracted_data["date"], errors='coerce', dayfirst=False, infer_datetime_format=True)
                      if pd.isna(extracted_data["date_dt"]):
                           st.warning(f"Date Pixtral ('{extracted_data.get('date')}') invalide. Gardée telle quelle.")
                           extracted_data.pop("date_dt") # Enlever si invalide
                      else:
                           # Optionnel: stocker la date normalisée
                           # extracted_data["date"] = extracted_data["date_dt"].strftime('%Y-%m-%d')
                           pass # Garder la date string originale pour l'instant
                 except Exception:
                      st.warning(f"Erreur conversion date Pixtral ('{extracted_data.get('date')}'). Gardée telle quelle.")
                      if "date_dt" in extracted_data: extracted_data.pop("date_dt")
            else: extracted_data["date"] = None


            if "nom_commerce" not in extracted_data: extracted_data["nom_commerce"] = None

            return extracted_data # Succès

        # Gestion Erreurs API pour v0.4.2 (peut être différent de v1.x)
        # On utilise une approche plus générique
        except requests.exceptions.HTTPError as http_err: # mistralai 0.4.x utilisait requests
             st.error(f"Erreur HTTP API Mistral: {http_err}")
             if http_err.response.status_code == 429:
                 retries += 1
                 retry_after = int(http_err.response.headers.get("Retry-After", 30))
                 st.warning(f"Limite API atteinte. Attente {retry_after}s ({retries}/{max_retries})...")
                 print(f"Rate limit hit (requests). Waiting {retry_after}s...")
                 if retries >= max_retries: return None
                 time.sleep(retry_after)
             else:
                 st.error(f"Erreur HTTP non récupérable: Status {http_err.response.status_code}")
                 return None # Échec définitif
        except json.JSONDecodeError as json_err:
            st.error(f"Erreur OCR (Pixtral) : Réponse non JSON ou JSON mal formé.\nErreur: {json_err}\nRéponse reçue: {response_content}")
            return {"date": None, "nom_commerce": None, "montant": None, "error": "JSONDecodeError", "raw_response": response_content}
        except Exception as e:
            st.error(f"Erreur inattendue appel Pixtral : {type(e).__name__} - {e}")
            st.error(f"Traceback: {traceback.format_exc()}")
            retries += 1
            if retries >= max_retries: return None
            time.sleep(5)

    st.error(f"Échec OCR Pixtral après {max_retries} essais.")
    return None


# --- Fonction de Matching (2 Étapes: Montant/Date + Similarité) ---
def filter_and_match_invoices(bank_row, invoices_data, model, date_delta_days=3, amount_tolerance=0.05, similarity_threshold=0.3):
    """
    Filtre les factures par montant/date puis classe par similarité sémantique.
    """
    if model is None: st.error("Modèle Sentence Transformer non chargé."); return None

    # 1. Infos de la transaction bancaire (avec gestion date NaT)
    try:
        transaction_amount = abs(float(bank_row["amount"]))
        transaction_date = bank_row["date"] # Doit être un objet datetime ou NaT (après parser v3)
        if pd.isna(transaction_date):
             # print(f"Date invalide pour transaction {bank_row.name}, matching par date impossible.")
             # On pourrait continuer sans filtre date, mais pour l'instant on ignore
             return None
    except (ValueError, TypeError, KeyError): return None # Montant ou date invalide/manquant

    desc_col = 'vendor' if 'vendor' in bank_row.index else 'source' if 'source' in bank_row.index else None
    transaction_desc = str(bank_row.get(desc_col, '')).lower().strip() if desc_col else ""
    # Si pas de description, on ne peut pas faire de matching sémantique
    if not transaction_desc: return None

    # 2. Étape 1: Filtrer les factures par Montant ET Date
    potential_matches = []
    min_date = transaction_date - pd.Timedelta(days=date_delta_days)
    max_date = transaction_date + pd.Timedelta(days=date_delta_days)

    for idx, invoice in enumerate(invoices_data):
        invoice_amount = invoice.get("montant") # Montant extrait par Pixtral
        invoice_date_str = invoice.get("date") # Date string extraite par Pixtral
        invoice_date_dt = invoice.get("date_dt") # Objet datetime (si conversion réussie dans OCR)

        # Vérifier montant
        if invoice_amount is not None:
            try:
                # Comparaison montant
                if abs(invoice_amount - transaction_amount) <= amount_tolerance:
                    # Vérification date (si date facture valide)
                    if invoice_date_dt is not None and not pd.isna(invoice_date_dt):
                         if min_date <= invoice_date_dt <= max_date:
                              invoice['original_index'] = idx
                              potential_matches.append(invoice)
                    # Si date facture invalide mais montant OK, on pourrait la garder? Pour l'instant non.
                    # else: # Si date invalide mais montant OK
                    #     invoice['original_index'] = idx
                    #     potential_matches.append(invoice) # Garder même si date invalide?

            except (TypeError): continue # Ignorer si montant pas comparable

    # 3. Étape 2: Calculer Similarité pour les factures filtrées
    if potential_matches:
        # Utiliser 'nom_commerce' retourné par Pixtral
        invoice_texts = [p.get("nom_commerce", "") for p in potential_matches]
        valid_indices = [i for i, txt in enumerate(invoice_texts) if txt] # Garder seulement celles avec un nom
        if not valid_indices: return None # Pas de noms à comparer

        potential_matches_filtered = [potential_matches[i] for i in valid_indices]
        invoice_texts_filtered = [invoice_texts[i] for i in valid_indices]

        try:
            # Calculer embeddings et similarités
            invoice_embeddings = model.encode(invoice_texts_filtered)
            transaction_embedding = model.encode([transaction_desc])
            similarities = cosine_similarity(transaction_embedding, invoice_embeddings).flatten()
            best_match_local_index = np.argmax(similarities) # Index dans la liste filtrée
            max_similarity = similarities[best_match_local_index]

            # Vérifier le seuil de similarité
            if max_similarity >= similarity_threshold:
                best_match_invoice = potential_matches_filtered[best_match_local_index]
                # Retourner les informations du meilleur match
                return {
                    "Date Relevé": transaction_date.strftime('%Y-%m-%d') if not pd.isna(transaction_date) else bank_row.get('date'),
                    "Montant Relevé": bank_row["amount"],
                    "Description Relevé": transaction_desc,
                    "Facture Fichier": best_match_invoice.get('filename'),
                    "Facture Nom Commerce": best_match_invoice.get("nom_commerce"),
                    "Facture Date": best_match_invoice.get("date"), # Date string originale
                    "Facture Montant": best_match_invoice.get("montant"),
                    "Score Similarité (%)": round(max_similarity * 100, 2)
                }
        except Exception as e:
            st.error(f"Erreur calcul similarité pour '{transaction_desc}': {e}")
            return None # Erreur pendant le calcul
    return None # Aucun match trouvé après filtrage ou similarité trop basse


# --- Lancement du Matching ---
st.divider()
st.header("3. 🚀 Lancer le Matching")

ready_to_match = (
    statement_df is not None and not statement_df.empty and
    invoice_files_list and
    mistral_client_ocr is not None and # Vérifier client OCR
    sentence_model is not None     # Vérifier modèle matching
)

if st.button("🔍 Démarrer le Matching", disabled=not ready_to_match, use_container_width=True):
    if ready_to_match:
        with st.spinner("Analyse OCR (Pixtral) & Matching (Filtre + Similarité)..."):

            # --- Étape 1: OCR avec Pixtral ---
            st.info("Étape 1 : Analyse OCR des factures via Pixtral...")
            all_invoice_data = []
            progress_bar_ocr = st.progress(0)
            status_text_ocr = st.empty()
            ocr_errors = 0
            pixtral_model_name = "pixtral-12b-2409" # Modèle spécifié dans rap.py

            for i, inv_file in enumerate(invoice_files_list):
                status_text_ocr.text(f"🔎 OCR en cours : {inv_file.name} ({i+1}/{len(invoice_files_list)})")
                try:
                    image_bytes = inv_file.getvalue()
                    image_media_type = inv_file.type
                    # Appel de la fonction OCR Pixtral JSON
                    extracted_data = call_pixtral_ocr_json(image_bytes, mistral_client_ocr, image_media_type, pixtral_model_name)

                    if extracted_data and extracted_data.get("error") is None :
                        extracted_data['filename'] = inv_file.name
                        all_invoice_data.append(extracted_data)
                    else:
                        ocr_errors += 1
                        raw_resp = extracted_data.get('raw_response', 'N/A') if isinstance(extracted_data, dict) else 'N/A'
                        st.warning(f"Échec OCR ou réponse invalide pour {inv_file.name}. Réponse: {raw_resp[:100]}...")
                except Exception as e:
                     ocr_errors += 1
                     st.error(f"Erreur imprévue traitement OCR de {inv_file.name}: {e}")
                     st.error(f"Traceback: {traceback.format_exc()}")
                finally:
                     progress_bar_ocr.progress((i + 1) / len(invoice_files_list))

            status_text_ocr.text("Analyse OCR terminée.")
            if ocr_errors > 0: st.warning(f"⚠️ {ocr_errors} facture(s) n'ont pas pu être analysée(s) correctement par l'OCR.")
            st.success(f"✅ {len(all_invoice_data)} factures traitées par OCR.")
            # Afficher données extraites pour debug si besoin
            # with st.expander("Voir données OCR"): st.json(all_invoice_data)


            # --- Étape 2: Matching en deux étapes ---
            if not all_invoice_data:
                 st.warning("Aucune donnée OCR valide. Impossible de lancer le matching.")
            else:
                st.info("Étape 2 : Recherche de correspondances (Filtre Montant/Date + Similarité Nom)...")
                matches = []
                progress_bar_match = st.progress(0)
                status_text_match = st.empty()

                # Vérifier colonnes avant de continuer
                required_cols_bank = ['amount', 'date'] # Date est maintenant requise pour le filtre delta
                if not all(col in statement_df.columns for col in required_cols_bank):
                     st.error(f"Colonnes requises {required_cols_bank} manquantes dans le relevé. Matching annulé.")
                     st.stop()
                # Vérifier si la colonne date a bien été convertie
                if not pd.api.types.is_datetime64_any_dtype(statement_df['date']):
                     st.error("La colonne 'date' du relevé n'a pas pu être convertie en type date. Matching annulé.")
                     st.stop()


                num_transactions = len(statement_df)
                for i, (_, row) in enumerate(statement_df.iterrows()):
                     status_text_match.text(f"🔗 Matching transaction {i+1}/{num_transactions}...")
                     try:
                         # Appel de la nouvelle fonction de matching en 2 étapes
                         match = filter_and_match_invoices(row, all_invoice_data, sentence_model, date_delta_days=3) # Delta de 3 jours
                         if match is not None:
                              matches.append(match)
                     except Exception as e:
                          st.error(f"Erreur imprévue lors du matching transaction {i+1}: {e}")
                          st.error(f"Traceback: {traceback.format_exc()}")
                     finally:
                          progress_bar_match.progress((i + 1) / num_transactions)

                status_text_match.text("Matching terminé.")
                match_results = pd.DataFrame(matches)

                # --- Étape 3: Affichage des Résultats ---
                if not match_results.empty:
                    st.success(f"🎯 {len(match_results)} correspondance(s) trouvée(s) après filtrage et analyse de similarité.")
                    match_results.sort_values(by="Score Similarité (%)", ascending=False, inplace=True)
                    # Adapter noms colonnes affichage
                    display_columns = {
                        "Date Relevé": "Date Relevé", "Montant Relevé": "Montant Relevé", "Description Relevé": "Description Relevé",
                        "Facture Fichier": "Fichier Facture", "Facture Nom Commerce": "Nom Commerce (Facture)",
                        "Facture Date": "Date (Facture)", "Facture Montant": "Montant (Facture)",
                        "Score Similarité (%)": "Score Similarité (%)"
                    }
                    match_results_display = match_results.rename(columns=display_columns)
                    st.dataframe(match_results_display, use_container_width=True)

                    try:
                        csv = match_results.to_csv(index=False).encode('utf-8')
                        st.download_button(label="📥 Télécharger les résultats en CSV", data=csv, file_name="resultats_matching_final.csv", mime="text/csv", key='download_button')
                    except Exception as e:
                        st.error(f"Erreur préparation téléchargement CSV: {e}")

                else:
                    st.warning("⚠️ Aucune correspondance trouvée avec les critères (Montant/Date/Similarité).")

    # Gestion erreur si bouton cliqué mais pas prêt
    elif not ready_to_match:
        error_message = "❌ Impossible de démarrer : "
        missing_items = []
        if statement_df is None or statement_df.empty: missing_items.append("Relevé bancaire valide")
        if not invoice_files_list: missing_items.append("Factures")
        if not mistral_client_ocr: missing_items.append("Client Mistral OCR (Clé API?)")
        if not sentence_model: missing_items.append("Modèle de matching")
        st.error(error_message + ", ".join(missing_items) + ".")

# Message si fichiers pas prêts
elif not ready_to_match and (uploaded_statement or uploaded_invoices):
    missing = []
    if uploaded_statement is None: missing.append("le relevé bancaire (CSV)")
    elif statement_df is None or statement_df.empty: missing.append("un relevé bancaire valide")
    if not uploaded_invoices: missing.append("les factures (images)")
    st.warning(f"Veuillez charger {', '.join(missing)} pour lancer.")

# --- Footer ---
st.divider()
st.caption("Application de démonstration – V2.0 (Pixtral OCR JSON + Matching 2 Étapes)")

