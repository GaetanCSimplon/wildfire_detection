import json
import pandas as pd

# Fonction de chargement de données coco.json
def load_coco_json(path):
    """Loads data from coco.json files

    Args:
        path (str): path to the coco.json file

    Returns:
        data (json): json content
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data

# coco.json => dataframe
def coco_to_df(coco_data):
    """ Transform json's lists into dataframe

    Args:
        coco_data (json): json's lists

    Returns:
        df: Dataframe with associated annotations, categories and images infos
    """
    # Mise en dataframe des listes du json
    df_images = pd.DataFrame(coco_data['images'])
    df_annotations = pd.DataFrame(coco_data['annotations'])
    df_categories = pd.DataFrame(coco_data['categories'])
    
    # Fusion annotations -> images
    df = df_annotations.merge(
        df_images[['id', 'file_name', 'width', 'height']], # colonnes à récupérer
        left_on='image_id', # colonne de df_annotations correspondant à l'image
        right_on='id', # colonne de df_images qui correspond à l'image
        suffixes=('_ann', '_img') # pour éviter les doublons de noms de colonnes
    ).drop(columns=['id_img']) # suppresion de la colonne 'id' de l'image après fusion
    
    # Ajout des catégories des annotations
    df = df.merge(
        df_categories[['id', 'name']], # colonnes à récupérer
        left_on='category_id', # colonne de df_annotations
        right_on='id', # colonne de df_categories
        suffixes=('', '_cat') # évite les doublons
    ).rename(columns={'name': 'category_name'}).drop(columns=['category_id']) # renommage et suppression de la colonne category_id
    
    return df
    
