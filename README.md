# wildfire_detection
Développement d'un modèle de détection des incendies

# Installer les dépendances 
```python
pip install requirements.txt
```

# Structure du projet

|- data -> contient le dataset
|- data_split -> dataset divisé en train/validation/test
|- prepare_data
  |- notebooks
    |- exploration.ipynb
    |- model.ipynb
    |- yolo_get.ipynb
  |- data_cleaner.py -> fonctions de nettoyage des données
  |- data_explorer.py -> fonctions d'exploration des données
  |- data_loader.py -> fonctions de chargement du dataset
  |- data_preparation.py -> fonctions de conversion format coco -> yolo
  |- pipeline.py
  |- main.py
  |- visualisation.py -> visualisation via fiftyone
|- train_results -> contient les csv des résultats d'entrainement (via google colab)
|- run_data_prep.py -> pipeline de préparation des données pour le format yolo
|- requirements.txt
