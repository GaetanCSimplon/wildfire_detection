# pipeline.py
from data_loader import load_coco_json, coco_to_df
from data_cleaner import (
    get_image_extensions,
    check_image_annotation_coherence,
    img_without_ann,
    annotations_without_images,
    detect_invalid_bboxes,
    clean_simple
)
from pathlib import Path

def run_diagnostic(coco_json_path, image_dir):
    # 1. Charger les données
    coco_data = load_coco_json(coco_json_path)
    df_images, df_annotations, df_categories = coco_to_df(coco_data)

    # 2. Vérifier les extensions
    exts = get_image_extensions(image_dir)
    print("Extensions d'images détectées :", exts)

    # 3. Vérifier cohérence images/annotations
    missing_in_folder, missing_in_coco = check_image_annotation_coherence(df_images, image_dir)
    print("Images absentes du dossier mais présentes dans JSON :", missing_in_folder)
    print("Fichiers présents dans le dossier mais pas listés dans JSON :", missing_in_coco)

    # 4. Images sans annotations
    unannotated = img_without_ann(df_images, df_annotations)
    print("Images sans annotations :", unannotated[['file_name', 'id']].values)

    # 5. Annotations sans image
    anomalies_ann = annotations_without_images(df_images, df_annotations)
    print("Annotations sans image :", anomalies_ann)

    # 6. Valeurs aberrantes dans les bounding boxes
    invalid_bboxes = detect_invalid_bboxes(df_annotations)
    print("Annotations avec bbox invalides :", invalid_bboxes)

    return {
        "extensions": exts,
        "missing_in_folder": missing_in_folder,
        "missing_in_coco": missing_in_coco,
        "unannotated": unannotated,
        "annotations_without_images": anomalies_ann,
        "invalid_bboxes": invalid_bboxes
    }

if __name__ == "__main__":
    # Exemple d’appel
    coco_json_path = "/home/gaetansimplon/wildfire_detection/data/_annotations.coco.json"
    image_dir = "/home/gaetansimplon/wildfire_detection/data"
    diagnostic = run_diagnostic(coco_json_path, image_dir)
        # Demande confirmation
    answer = input("\nSouhaites-tu lancer le nettoyage automatique ? (o/n) : ").strip().lower()

    if answer == "o":
        output_json_path = "/home/fadilatou/PROJETS/wildfire_detection/data/_annotations_clean.coco.json"
        clean_simple(coco_json_path, image_dir, output_json_path)
        print("Nettoyage terminé, fichier propre :", output_json_path)
    else:
        print("Nettoyage annulé")