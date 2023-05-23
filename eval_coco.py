from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

pred = './coco_val.bbox.json'
gt = 'data/coco/annotations/instances_val2017.json'

cocoGt = COCO(gt)
cocoPred = cocoGt.loadRes(pred)

cocoEval = COCOeval(cocoGt, cocoPred, 'bbox')

cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
