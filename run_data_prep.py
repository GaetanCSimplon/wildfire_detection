from prepare_data.data_preparation import prepare_data

prepare_data(
    coco_json="/home/fadilatou/PROJETS/wildfire_detection/data/_annotations_clean.coco.json",
    images_dir="/home/fadilatou/PROJETS/wildfire_detection/data",
    output_dir="/home/fadilatou/PROJETS/wildfire_detection/data_split"
)
