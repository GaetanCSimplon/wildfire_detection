from pathlib import Path
import pandas as pd
import json
import os

def get_image_extensions(image_dir):
    """
    Retourne la liste des extensions trouvées dans un dossier d'images.
    """
    extensions = [p.suffix for p in Path(image_dir).iterdir() if p.is_file()]
    return list(set(extensions))

def check_image_annotation_coherence(img_df, image_dir):
    """
    Retourne deux listes :
    - images absentes du dossier mais présentes dans img_df
    - fichiers présents dans le dossier mais pas listés dans img_df
    """
    # liste des noms d'image dans le coco.json
    coco_images = set(img_df['file_name'])

    # liste des fichiers présents dans le dossier
    actual_images = {p.name for p in Path(image_dir).iterdir() if p.is_file()}

    missing_in_folder = coco_images - actual_images
    missing_in_coco = actual_images - coco_images

    return missing_in_folder, missing_in_coco

def img_without_ann(img_df, ann_df):
    """
    Retourne une liste des images sans annotations.
    """
    ann_img_ids = ann_df['image_id'].unique()
    unann = img_df[~img_df['id'].isin(ann_img_ids)]
    return unann

def annotations_without_images(img_df, ann_df):
    """
    Retourne les annotations dont l'image_id n'existe pas dans img_df.
    """
    valid_ids = img_df['id'].unique()
    anomalies = ann_df[~ann_df['image_id'].isin(valid_ids)]
    return anomalies

def detect_invalid_bboxes(ann_df):
    """
    Retourne les annotations avec largeur ou hauteur incohérente.
    """
    # dans COCO, bbox = [x_min, y_min, width, height]
    invalid = ann_df[
        (ann_df['bbox'].apply(lambda x: x[2] <= 0)) | 
        (ann_df['bbox'].apply(lambda x: x[3] <= 0))
    ]
    return invalid

def clean_simple(coco_json_path, image_dir, output_json_path):
    # 1. Charger les données COCO
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']

    # 2. Construire des listes utiles
    files_in_folder = {p.name for p in Path(image_dir).iterdir() if p.is_file()}
    files_in_json = {img['file_name'] for img in images}

    # 3. Supprimer du dossier les fichiers qui ne sont pas dans le JSON
    for file_name in files_in_folder - files_in_json:
        os.remove(Path(image_dir)/file_name)
        print("Supprimé du dossier:", file_name)

    # 4. Supprimer du JSON les images absentes du dossier
    valid_images = [img for img in images if img['file_name'] in files_in_folder]
    valid_ids = {img['id'] for img in valid_images}

    # 5. Supprimer du JSON les annotations qui ne correspondent plus à aucune image
    valid_annotations = [ann for ann in annotations if ann['image_id'] in valid_ids and ann['bbox'][2] > 0 and ann['bbox'][3] > 0]

    # 6. Réécrire le JSON propre
    coco_data['images'] = valid_images
    coco_data['annotations'] = valid_annotations

    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print("Fichier COCO nettoyé sauvegardé :", output_json_path)