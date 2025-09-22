# main.py
from pipeline import run_diagnostic
from data_cleaner import clean_simple

def main():
    # Définis les chemins (tu pourras remplacer par argparse plus tard)
    coco_json_path = "/home/gaetansimplon/wildfire_detection/data/_annotations.coco.json"
    image_dir = "/home/gaetansimplon/wildfire_detection/data"

    # Étape 1 : diagnostic
    diagnostic = run_diagnostic(coco_json_path, image_dir)

    # Étape 2 : confirmation pour nettoyage
    answer = input("\nSouhaites-tu lancer le nettoyage automatique ? (o/n) : ").strip().lower()
    if answer == "o":
        output_json_path = "/home/gaetansimplon/wildfire_detection/data/_annotations_clean.coco.json"
        clean_simple(coco_json_path, image_dir, output_json_path)
        print("✅ Nettoyage terminé, fichier propre :", output_json_path)
    else:
        print("❌ Nettoyage annulé")

if __name__ == "__main__":
    main()
