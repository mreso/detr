import pytest
from collections import namedtuple
from datasets.coco_data_pipe import (
        build_web_datapipe,
        load_annotation_from_url,
        WebCocoImgUrlLister,
        WebCocoAnnotationLoader,
    )


BASE_URL = "https://mscoco-dataset.s3.us-west-2.amazonaws.com/"

@pytest.fixture(scope="session")
def val_coco():
    yield load_annotation_from_url(BASE_URL, "val")


def test_web_coco_img_url_lister(val_coco):
    url = BASE_URL
    image_set = "val"

    assert len(val_coco.imgs) == 5000

    dp = WebCocoImgUrlLister(url, image_set, val_coco)

    files = [e.split("/")[-1] for e in dp]

    assert "000000000139.jpg" in files

def test_build_web_datapipe():

    Args = namedtuple('args', ['coco_path', 'world_size', 'batch_size', 'masks', 'rank'])

    batch_size = 2

    args = Args(BASE_URL, 2, batch_size, False, 0)

    dp = build_web_datapipe("val", args)

    item = next(iter(dp))
    
    assert len(item) == 2
    assert isinstance(item[1][0], dict)