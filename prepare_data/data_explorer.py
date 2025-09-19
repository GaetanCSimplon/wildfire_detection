import pandas as pd
import numpy
import matplotlib.pyplot as plt
import seaborn as sns

## --- Partie Statistique --- ##

# Résumé des données
def dataset_summary(df):
    # Sécurité : afficher les colonnes présentes
    print("Colonnes disponibles :", df.columns.tolist())

    num_img = df['image_id'].nunique()
    num_ann = len(df)
    mean_ann = df.groupby('image_id').size().mean()

    print(f"Nombre total d’images : {num_img}")
    print(f"Nombre total d’annotations : {num_ann}")
    print(f"Nombre moyen d’annotations par image : {mean_ann:.2f}")


# Distribution des annotations par image
def ann_per_img(df):
    annotations_count = df.groupby('image_id').size()
    print("Statistiques des annotations par image : ")
    print(annotations_count.describe())
    

    
    
## --- Partie Visualisation Graphique --- ##


def plot_ann_per_image(df):
    # Compter le nombre d'annotations par image
    ann_counts = df.groupby('image_id').size().reset_index(name='nb_ann')

    # Tracer l'histogramme
    plt.figure(figsize=(8,5))
    sns.countplot(x='nb_ann', data=ann_counts, color='skyblue')

    # Habillage
    plt.title("Nombre d'images selon le nombre d'annotations")
    plt.xlabel("Nombre d'annotations par image")
    plt.ylabel("Nombre d'images")
    plt.tight_layout()
    plt.show()

