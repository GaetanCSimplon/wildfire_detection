from pathlib import Path

def get_file_extensions_from_dir(folder_path):
    """
    Renvoie les extensions des fichiers présents dans un dossier.

    Paramètres :
        folder_path (str ou Path) : chemin vers le dossier

    Retour :
        set : ensemble des extensions trouvées (sans le point)
    """
    folder = Path(folder_path)

    extensions = {file.suffix[1:] for file in folder.iterdir() if file.is_file()}
    return extensions


# Vérification de la cohérence entre images disponibles et les images présent dans l'annotations
def coherence_images(image_dir, coco_data):
    """
    Vérifie la cohérence entre les images du dossier et celles listées dans le fichier COCO.
    Retourne les images manquantes et les images en trop.
    """
    from pathlib import Path
    # Liste des fichiers images dans le dossier
    image_files = set([f.name for f in Path(image_dir).glob("*.jpg")])
    # Liste des fichiers attendus dans le COCO json
    coco_images = set([img['file_name'] for img in coco_data['images']])

    manquant = coco_images - image_files
    extra = image_files - coco_images
    return {"missing_in_folder": len(manquant), "extra_in_folder": len(extra)}

# Les images sans annotations

def get_images_without_annotations(coco_data):
    """
    Renvoie la liste des images sans annotations.
    
    Paramètres :
        coco_data (dict) : données complètes du fichier coco.json chargé avec json.load()
    
    Retour :
        list[str] : noms des fichiers images sans annotations
    """
    # Récupérer tous les ids d'images
    all_image_ids = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # Récupérer les ids d'images annotées
    annotated_ids = {ann['image_id'] for ann in coco_data['annotations']}
    
    # Trouver les ids qui ne sont jamais annotés
    images_without_ann = [fname for img_id, fname in all_image_ids.items() if img_id not in annotated_ids]
    
    return images_without_ann

