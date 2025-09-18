import json
import pandas as pd

# Fonction de chargement de données coco.json
def load_coco_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

coco_data = load_coco_json('/home/gaetansimplon/wildfire_detection/data/_annotations.coco.json')

# Affichage des clés
print(coco_data.keys())

# Affichage 1ère image
print('Image 1 :', coco_data['images'][:1])
# Affichage 1ère annotation
print('Annotation 1 :', coco_data['annotations'][:1])
# Affichage catégories
print('Catégories : ', coco_data['categories'])


# coco.json => dataframe
def coco_to_df(coco_data):
    
    df_images = pd.DataFrame(coco_data['images'])
    df_annotations = pd.DataFrame(coco_data['annotations'])
    df_categories = pd.DataFrame(coco_data['categories'])
    
    df = df_annotations.merge(
        df_images[['id', 'file_name', 'width', 'height']],
        left_on='image_id', right_on='id', suffixes=('_ann', '_img')
    ).drop(columns=['id_img'])
    
    df = df.merge(
        df_categories[['id', 'name']],
        left_on='category_id', right_on='id', suffixes=('', '_cat')
    ).rename(columns={'name': 'category_name'}).drop(columns=['category_id'])
    
    return df
    
df = coco_to_df(coco_data)

display(df.head())
print(df.columns)

print(df)