import base64
import os
import json
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from mistralai import Mistral

def encode_image(image_path):
    """Encode une image en base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Erreur : Le fichier {image_path} est introuvable.")
        return None
    except Exception as e:
        print(f"Erreur lors de l'encodage de l'image : {e}")
        return None

def get_invoice_details(image_path, api_key, model):
    """Obtenir les détails d'une facture en utilisant une image encodée."""
    base64_image = encode_image(image_path)
    if not base64_image:
        return None

    # Initialiser le client Mistral
    client = Mistral(api_key=api_key)

    # Définir les messages pour le chat
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": ("Analyse cette facture et extrait les informations suivantes sous forme de JSON :\n"
                             "- Date de la facture\n"
                             "- Nom du commerce\n"
                             "- Montant total\n\n"
                             "Réponds uniquement avec un JSON structuré de cette manière :\n"
                             "{\n"
                             "  \"date\": \"YYYY-MM-DD\",\n"
                             "  \"nom_commerce\": \"Nom du commerce\",\n"
                             "  \"montant\": \"XX.XX\"\n"
                             "}\n"
                             "Ne renvoie aucun texte supplémentaire en dehors du JSON.")
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            ]
        }
    ]

    while True:
        try:
            # Obtenir la réponse du chat
            chat_response = client.chat.complete(
                model=model,
                messages=messages,
                response_format={"type": "json_object"}
            )
            # Extraire et retourner le contenu JSON
            return json.loads(chat_response.choices[0].message.content)
        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 429:
                print("Limite de taux atteinte. Attente avant de réessayer...")
                retry_after = int(http_err.response.headers.get("Retry-After", 60))
                time.sleep(retry_after)  # Attendre le temps spécifié dans l'en-tête Retry-After
            else:
                print(f"Erreur HTTP lors de l'appel à l'API Mistral pour {image_path}: {http_err}")
                return None
        except Exception as e:
            print(f"Erreur lors de l'appel à l'API Mistral pour {image_path}: {e}")
            return None

def process_receipts(directory_path, api_key, model, output_file):
    """Traite toutes les images dans le dossier des reçus et sauvegarde les résultats."""
    receipts_data = []

    # Charger les données existantes si le fichier existe
    if os.path.exists(output_file):
        with open(output_file, "r") as file:
            receipts_data = json.load(file)
            print("Données existantes chargées.")

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf')):
            image_path = os.path.join(directory_path, filename)
            # Vérifier si les données sont déjà extraites
            if not any(entry["Fichier"] == filename for entry in receipts_data):
                print(f"Traitement de {filename}...")
                invoice_details = get_invoice_details(image_path, api_key, model)
                if invoice_details:
                    receipts_data.append({
                        "Fichier": filename,
                        "Informations": invoice_details
                    })

    # Sauvegarder les résultats
    with open(output_file, "w") as file:
        json.dump(receipts_data, file, indent=4)
    print(f"Données sauvegardées dans {output_file}")

    return pd.DataFrame(receipts_data)

def main():
    # Charger les variables d'environnement depuis le fichier .env
    load_dotenv()

    # Chemins vers les fichiers
    receipts_directory = "receipts-20250331T125700Z-001/receipts"
    output_file = "extracted_invoices.json"

    # Récupérer la clé API depuis les variables d'environnement
    api_key = os.getenv("MISTRAL_KEY")
    if not api_key:
        print("Erreur : La clé API Mistral n'est pas définie.")
        return

    # Spécifier le modèle
    model = "pixtral-12b-2409"

    # Traiter les reçus et sauvegarder les résultats
    receipts_df = process_receipts(receipts_directory, api_key, model, output_file)
    print("Traitement terminé.")

if __name__ == "__main__":
    main()
