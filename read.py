import json

def view_saved_data(file_path):
    """Afficher les données sauvegardées dans un fichier JSON."""
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            print(json.dumps(data, indent=4, ensure_ascii=False))
    except FileNotFoundError:
        print(f"Erreur : Le fichier {file_path} est introuvable.")
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")

# Chemin vers le fichier JSON
file_path = "extracted_invoices.json"

# Afficher les données
view_saved_data(file_path)
