# utils/llm_clients.py (v2 - Appel Mistral pour Matching)
import streamlit as st
import os
from dotenv import load_dotenv
import json
import time
from mistralai.client import MistralClient
# Essayer d'importer l'exception de base si possible, sinon utiliser Exception générale
try:
    from mistralai import MistralAPIException
except ImportError:
    try: # Essayer l'ancien chemin
        from mistralai.exceptions import MistralAPIException
    except ImportError: # Si toujours pas trouvé, utiliser Exception
        MistralAPIException = Exception # Définir comme alias pour Exception générale

import httpx # Pour attraper les erreurs HTTP spécifiques
import traceback

# Charger les clés API depuis .env
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Initialiser le client Mistral (mis en cache dans app.py via une fonction get)
@st.cache_resource
def get_mistral_client():
    """Initialise et retourne le client Mistral."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        st.error("Clé API Mistral (MISTRAL_API_KEY) non trouvée dans .env.")
        return None
    print("--- Initialisation du client Mistral ---")
    try:
        client = MistralClient(api_key=api_key)
        print("--- Client Mistral initialisé ---")
        return client
    except Exception as e:
        st.error(f"Erreur init client Mistral : {e}")
        return None

# Fonction pour appeler Mistral pour le matching
def get_match_from_mistral(client, statement_line_data, invoice_data_list, model="mistral-large-latest", max_retries=3):
    """
    Utilise l'API Mistral pour trouver la meilleure facture correspondant à une ligne de relevé.

    Args:
        client: Le client Mistral initialisé.
        statement_line_data (dict): Infos de la ligne de relevé (date, amount, description).
        invoice_data_list (list): Liste de dictionnaires contenant les infos des factures (OCR EasyOCR).
        model (str): Nom du modèle Mistral à utiliser.
        max_retries (int): Nombre max de tentatives en cas de rate limit.

    Returns:
        dict: Résultat JSON du LLM ou None si échec.
               Ex: {"matching_invoice_index": <int>, "confidence": <float>, "reasoning": "..."}
    """
    if not client: st.error("Client Mistral non initialisé."); return None
    if not invoice_data_list: st.warning("Aucune donnée de facture fournie pour le matching."); return None

    # Préparer les données pour le prompt (similaire à Groq)
    try:
        s_amount = abs(float(statement_line_data.get('amount', 0)))
        s_date = statement_line_data.get('date', 'N/A')
        s_desc_col = 'vendor' if 'vendor' in statement_line_data else 'source'
        s_desc = statement_line_data.get(s_desc_col, 'N/A')
        statement_str = f"Ligne Relevé: Date='{s_date}', Description='{s_desc}', Montant='{s_amount:.2f}'"

        invoices_str_parts = []
        for i, inv in enumerate(invoice_data_list):
            # Utiliser les clés extraites par EasyOCR + extract_invoice_data_from_text
            inv_amount = inv.get('montant')
            inv_name = inv.get('nom_commerce', 'N/A')
            inv_date = inv.get('date', 'N/A')
            if inv_amount is not None:
                 try:
                     inv_amount_float = float(inv_amount)
                     invoices_str_parts.append(f"  Facture Index {i}: Nom='{inv_name}', Date='{inv_date}', Montant='{inv_amount_float:.2f}'")
                 except (ValueError, TypeError):
                     invoices_str_parts.append(f"  Facture Index {i}: Nom='{inv_name}', Date='{inv_date}', Montant='{inv_amount}' (Invalide)")
            else:
                 invoices_str_parts.append(f"  Facture Index {i}: Nom='{inv_name}', Date='{inv_date}', Montant='N/A'")
        invoices_str = "\n".join(invoices_str_parts)
        if not invoices_str: st.warning("Aucune facture avec montant valide à comparer."); return None

    except Exception as e:
        st.error(f"Erreur préparation données pour prompt Mistral: {e}"); return None

    # Créer le prompt (similaire à Groq)
    prompt = f"""
    Tâche : Tu es un assistant comptable expert en rapprochement bancaire. Ton but est de trouver la facture la plus probable correspondant à la ligne de relevé bancaire fournie.

    Ligne de Relevé Bancaire :
    {statement_str}

    Factures Disponibles (extraites par OCR, avec leur index de liste commençant à 0) :
    {invoices_str}

    Instructions :
    1. Compare le montant de la ligne de relevé (valeur absolue) avec le montant de chaque facture disponible. Tolérance +/- 0.05.
    2. Compare la description de la ligne de relevé avec le nom du commerce de la facture. Prends en compte variations, abréviations.
    3. La proximité des dates est un indice secondaire.
    4. Évalue la probabilité de correspondance pour chaque facture potentielle (montant similaire).
    5. Choisis la facture avec la plus haute probabilité.
    6. Réponds **UNIQUEMENT** avec un objet JSON valide contenant les clés suivantes :
       - "matching_invoice_index": L'index (nombre entier) de la facture la mieux adaptée (commençant à 0). Si aucune ne correspond bien, retourne -1.
       - "confidence": Un score de confiance (nombre flottant entre 0.0 et 1.0). Si index = -1, confidence = 0.0.
       - "reasoning": Une courte phrase (string) expliquant ton choix ou l'absence de match.

    Format JSON attendu :
    {{
      "matching_invoice_index": <index ou -1>,
      "confidence": <score flottant>,
      "reasoning": "Explication concise."
    }}
    Ne fournis aucun texte avant ou après l'objet JSON.
    """

    messages = [{"role": "user", "content": prompt}]
    retries = 0
    response_content = "No response" # Initialiser au cas où l'appel échoue avant d'assigner

    while retries < max_retries:
        try:
            print(f"--- Appel API Mistral Matching (Essai {retries+1}/{max_retries}) ---")
            chat_response = client.chat(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1
            )
            print(f"--- Réponse API Mistral Matching reçue ---")
            response_content = chat_response.choices[0].message.content
            response_content_cleaned = response_content.replace(': "null"', ': null').replace(':null', ': null')
            result = json.loads(response_content_cleaned)

            # Validation basique
            if isinstance(result.get("matching_invoice_index"), int) and \
               isinstance(result.get("confidence"), (float, int)) and \
               isinstance(result.get("reasoning"), str):
                return result # Succès
            else:
                st.warning(f"Réponse JSON de Mistral invalide/incomplète: {result}")
                raise ValueError("Format JSON de Mistral invalide")

        # Gestion Erreurs API (Rate limit, etc.)
        # Tenter d'attraper MistralAPIException si elle a pu être importée, sinon httpx.HTTPStatusError
        except (MistralAPIException, httpx.HTTPStatusError) as e:
            http_status = -1
            headers = {}
            # Essayer d'extraire le code de statut et les en-têtes
            if isinstance(e, MistralAPIException) and hasattr(e, 'http_status'):
                 http_status = e.http_status
                 if hasattr(e, 'response') and e.response and hasattr(e.response, 'headers'):
                      headers = e.response.headers
            elif isinstance(e, httpx.HTTPStatusError):
                 http_status = e.response.status_code
                 headers = e.response.headers

            st.error(f"Erreur API Mistral: Status {http_status} - {e}")
            if http_status == 429: # Rate Limit
                retries += 1
                retry_after = int(headers.get("Retry-After", 30))
                st.warning(f"Limite API Mistral atteinte. Attente {retry_after}s ({retries}/{max_retries})...")
                print(f"Mistral Rate limit hit. Waiting {retry_after}s...")
                if retries >= max_retries:
                     st.error("Limite API Mistral atteinte et max retries dépassé.")
                     return None
                time.sleep(retry_after)
            else: # Autre erreur HTTP
                st.error(f"Erreur API Mistral non récupérable (Status {http_status}).")
                return None
        except json.JSONDecodeError:
            st.error(f"Erreur Matching (Mistral) : Réponse non JSON.\nRéponse: {response_content}")
            return None
        except Exception as e: # Autres erreurs (connexion, timeout...)
            st.error(f"Erreur inattendue appel Mistral Matching : {type(e).__name__} - {e}")
            st.error(f"Traceback: {traceback.format_exc()}")
            retries += 1
            if retries >= max_retries: return None
            st.warning(f"Attente 5s avant nouvel essai ({retries}/{max_retries})...")
            time.sleep(5)

    st.error(f"Échec appel API Mistral Matching après {max_retries} essais.")
    return None

