# Import des modules nécessaires
import json            # Pour lire le fichier COCO (JSON)
import os              # Pour gérer les chemins de fichiers
import shutil          # Pour copier les images
from pathlib import Path  # Pour créer facilement des dossiers
from sklearn.model_selection import train_test_split  # Pour découper le dataset

# Fonction principale
def prepare_data(coco_json, images_dir, output_dir):
    """
    Convertit un dataset COCO en dataset YOLO et le découpe en train/val/test.
    
    coco_json : chemin vers le fichier COCO (.json)
    images_dir : dossier contenant les images
    output_dir : dossier où seront créés train/val/test avec images + labels
    """
    
    # Lecture du fichier COCO
    with open(coco_json) as f:
        coco = json.load(f)
    
    # Mapping image_id -> nom de fichier
    # Exemple : {1: 'img_001.jpg', 2: 'img_002.jpg', ...}
    id_to_file = {img['id']: img['file_name'] for img in coco['images']}
    
    # Mapping image_id -> (largeur, hauteur)
    # Pour la normalisation des bbox plus tard
    id_to_size = {img['id']: (img['width'], img['height']) for img in coco['images']}
    
    # On regroupe les annotations par image
    # ann_per_img[image_id] = liste d'annotations
    ann_per_img = {}
    for ann in coco['annotations']:
        ann_per_img.setdefault(ann['image_id'], []).append(ann)
    
    # Liste de tous les IDs d'images
    all_ids = list(id_to_file.keys())
    
    # -------------------------------
    # 1. Découpage train / val / test
    # -------------------------------
    # 70% train
    train_ids, temp_ids = train_test_split(all_ids, train_size=0.7, random_state=42)
    # Les 30% restants sont pour val et test : 20% val, 10% test
    val_ids, test_ids = train_test_split(temp_ids, train_size=2/3, random_state=42)

    splits = {'train': train_ids, 'val': val_ids, 'test': test_ids}
    
    # -------------------------------
    # 2. Création des dossiers et conversion COCO -> YOLO
    # -------------------------------
    for split, ids in splits.items():
        # Création des sous-dossiers images/labels
        images_out = Path(output_dir)/split/'images'
        labels_out = Path(output_dir)/split/'labels'
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        # Boucle sur chaque image de ce split
        for img_id in ids:
            # Nom de l'image
            fname = id_to_file[img_id]
            # Taille de l'image (largeur, hauteur)
            w, h = id_to_size[img_id]
            # Copie de l'image dans le bon dossier
            shutil.copy(os.path.join(images_dir, fname), images_out/fname)

            # Liste qui contiendra les lignes du fichier label YOLO
            yolo_lines = []
            for ann in ann_per_img.get(img_id, []):  # Boucle sur toutes les annotations de l'image
                cls = ann['category_id']           # ID de la classe
                x, y, bw, bh = ann['bbox']        # bbox COCO : x,y,width,height en pixels
                
                # Conversion COCO -> YOLO
                x_c = (x + bw/2) / w               # centre X normalisé (0-1)
                y_c = (y + bh/2) / h               # centre Y normalisé (0-1)
                bw /= w                            # largeur normalisée
                bh /= h                            # hauteur normalisée
                
                # Création de la ligne YOLO
                # Format : class x_center y_center width height
                yolo_lines.append(f"{cls} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

            # Écriture du fichier texte YOLO
            # Même nom que l'image mais extension .txt
            label_file = labels_out/f"{Path(fname).stem}.txt"
            label_file.write_text("\n".join(yolo_lines))