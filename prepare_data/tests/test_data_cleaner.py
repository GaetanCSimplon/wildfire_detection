import pandas as pd
from prepare_data.data_cleaner import (detect_invalid_bboxes, get_image_extensions,
                                       check_image_annotation_coherence, clean_simple,
                                       img_without_ann, annotations_without_images
                                       )



def test_get_images_no_annotation_unannotated_returns():
  """Cas : certaines images annotées, certaines non : seules les non annotées doivent apparaître."""
  df_images = pd.DataFrame([
      {"id": 1, "file_name": "img1.jpg"},
      {"id": 2, "file_name": "img2.jpg"},
      {"id": 3, "file_name": "img3.jpg"},
  ])
  df_annotations = pd.DataFrame([
      {"id": 10, "image_id": 1},
  ])
  result = img_without_ann(df_images, df_annotations)
  assert len(result) == 2 # On attend 2 images non annotées
  assert set(result["file_name"]) == {"img2.jpg", "img3.jpg"} # img non annotées qui doivent être renvoyées
  
def test_detect_invalid_bboxes():
    df = pd.DataFrame({
        "bbox": [
            [0, 0, 10, 10],   # valide
            [5, 5, -1, 10],   # invalide (largeur <= 0)
            [2, 2, 5, -3]     # invalide (hauteur <= 0)
        ]
    })
    result = detect_invalid_bboxes(df)
    assert len(result) == 2

def test_annotations_without_images():
    df_img = pd.DataFrame([
      {"id": 1, "file_name": "img1.jpg"},
      {"id": 2, "file_name": "img2.jpg"},
      {"id": 3, "file_name": "img3.jpg"},
    ])
    df_ann = pd.DataFrame([
        {"id": 1, "image_id": 1},
        {"id": 2, "image_id": 2},
        {"id": 3, 'image_id': None},
    ])
    orphaned = annotations_without_images(df_img, df_ann)
    assert len(orphaned) == 1