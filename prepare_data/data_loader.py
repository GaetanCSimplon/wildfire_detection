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
def coco_to_df(coco:  dict):
    """ Transform json's lists into dataframe

    Args:
        coco_data (json): json's lists

    Returns:
        df: Dataframe with associated annotations, categories and images infos
    """
    # Mise en dataframe des listes du json
    df_images = pd.DataFrame(coco.get('images', []))
    df_annotations = pd.DataFrame(coco.get('annotations', []))
    df_categories = pd.DataFrame(coco.get('categories', []))
    
    return df_images, df_annotations, df_categories
    