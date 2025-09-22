import fiftyone as fo

dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path='/home/gaetansimplon/wildfire_detection/data', # Dossier contenant les images
    labels_path='/home/gaetansimplon/wildfire_detection/data/_annotations_clean.coco.json'
)

session = fo.launch_app(dataset)