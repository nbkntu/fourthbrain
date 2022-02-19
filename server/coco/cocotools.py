from pycocotools.coco import COCO

class CocoUtils:
    def __init__(self,
                data_dir = './coco/annotations_trainval2017',
                data_type = 'val2017'):
        ann_file = '{}/annotations/instances_{}.json'.format(data_dir, data_type)
        self.coco = COCO(ann_file)

    def load_annotations(self, image_id, category_id):
        ann_ids = self.coco.getAnnIds(
            imgIds=[image_id],
            catIds=[category_id])
        anns = self.coco.loadAnns(ann_ids)
        return anns
