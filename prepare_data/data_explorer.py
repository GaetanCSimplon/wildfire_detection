import pandas as pd

def count_images(img_df: pd.DataFrame):
    return len(img_df)

def count_ann(ann_df: pd.DataFrame):
    return len(ann_df)

def cat_list(cat_df: pd.DataFrame):
    return cat_df['name'].tolist()

def img_per_cat(ann_df: pd.DataFrame):
    grouped = ann_df[['image_id', 'category_id']].drop_duplicates().groupby('category_id').size().reset_index(name='nb_img')
    return grouped

def ann_stats_per_img(ann_df: pd.DataFrame):
    return ann_df.groupby('image_id').size().describe()


def avg_ann_per_img(ann_df: pd.DataFrame):
    return ann_df.groupby('image_id')['id'].nunique().mean()

def stats_summary(img_df: pd.DataFrame, ann_df: pd.DataFrame):
    img_count = count_images(img_df)
    ann_count = count_images(ann_df)
    ann_per_img = avg_ann_per_img(ann_df)

    print(f"Nombre d'images : {count_images(img_df)}")
    print(f"Nombre d'annotations : {count_images(ann_df)}")
    print(f"Nombre moyen d'annotations par image : {avg_ann_per_img(ann_df)}")
    return {
        "img_count": img_count,
        "ann_count": ann_count,
        "ann_per_img": ann_per_img
    }