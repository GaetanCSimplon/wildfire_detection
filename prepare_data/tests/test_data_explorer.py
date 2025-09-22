import pandas as pd
import pytest
from prepare_data.data_explorer import count_images, count_ann, stats_summary



def test_count_images():
    df = pd.DataFrame({
        "image_id":[1,2,3,4]
    })
    img_count = count_images(df)
    assert img_count == 4
    
def test_ann_count():
    df = pd.DataFrame({
        "id_ann": [1,2,3]
    })
    ann_count = count_ann(df)
    assert ann_count != 4

def test_stats_summary():
    df_img = pd.DataFrame({
        "image_id":[1,2,3],
    })
    df_ann = pd.DataFrame({
        "id": [1,2,3],
        "image_id":[1,2,3],
    })
    summary = stats_summary(df_img, df_ann)
    # Vérifie que les clés existent et que les valeurs sont correctes
    assert "img_count" in summary
    assert "ann_count" in summary
    assert "ann_per_img" in summary

    assert isinstance(summary['img_count'], int)
    assert isinstance(summary['ann_count'], int)
    assert isinstance(summary['ann_per_img'], float)



    