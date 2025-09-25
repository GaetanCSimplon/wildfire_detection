from prepare_data.data_preparation import prepare_data

prepare_data(
    coco_json="/home/gaetansimplon/wildfire_detection/data/_annotations_clean.coco.json",
    images_dir="/home/gaetansimplon/wildfire_detection/data",
    output_dir="/home/gaetansimplon/wildfire_detection/data_split"
)