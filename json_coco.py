# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser
from pathlib import Path

import mmengine
import torch
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config, ConfigDict
from mmengine.utils import ProgressBar, path

from mmyolo.utils import switch_to_deploy
from mmyolo.utils.misc import get_file_list


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'img', help='Image path, include image file, dir and URL.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        'gt',
        default='coco/annotations/instances_val2017.json',
        help='gt image to index')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--task', default='val_2017', help='Task for inference')
    parser.add_argument(
        '--deploy',
        action='store_true',
        help='Switch model to deployment mode')
    parser.add_argument(
        '--tta',
        action='store_true',
        help='Whether to use test time augmentation')
    parser.add_argument(
        '--score-thr', type=float, default=0.001, help='Bbox score threshold')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config = args.config

    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None

    if args.tta:
        assert 'tta_model' in config, 'Cannot find ``tta_model`` in config.' \
                                      " Can't use tta !"
        assert 'tta_pipeline' in config, 'Cannot find ``tta_pipeline`` ' \
                                         "in config. Can't use tta !"
        config.model = ConfigDict(**config.tta_model, module=config.model)
        test_data_cfg = config.test_dataloader.dataset
        while 'dataset' in test_data_cfg:
            test_data_cfg = test_data_cfg['dataset']

        # batch_shapes_cfg will force control the size of the output image,
        # it is not compatible with tta.
        if 'batch_shapes_cfg' in test_data_cfg:
            test_data_cfg.batch_shapes_cfg = None
        test_data_cfg.pipeline = config.tta_pipeline

    # TODO: TTA mode will error if cfg_options is not set.
    #  This is an mmdet issue and needs to be fixed later.
    # build the model from a config file and a checkpoint file
    model = init_detector(
        config, args.checkpoint, device=args.device, cfg_options={})

    if args.deploy:
        switch_to_deploy(model)

    path.mkdir_or_exist(args.out_dir)

    # get file list
    files, source_type = get_file_list(args.img)

    # # get model class name
    # dataset_classes = model.dataset_meta.get('classes')
    with open(args.gt) as f:
        gt = mmengine.load(f, file_format='json')
    name2id = {g['file_name']: g['id'] for g in gt['images']}
    # start detector inference
    progress_bar = ProgressBar(len(files))
    coco_results = []
    for file in files:
        result = inference_detector(model, file)

        # Get candidate predict info with score threshold
        pred_instances = result.pred_instances[
            result.pred_instances.scores > args.score_thr]

        bboxes = pred_instances.bboxes
        x1y1, x2y2 = bboxes.chunk(2, 1)
        wh = x2y2 - x1y1
        bboxes = torch.cat([x1y1, wh], dim=1)
        labels = pred_instances.labels
        scores = pred_instances.scores

        for bbox, label, score in zip(bboxes, labels, scores):
            bbox = [round(float(x), 5) for x in bbox]
            label = int(label)
            score = round(float(score), 5)
            pred_data = {
                'image_id': name2id[os.path.basename(file)],
                'category_id': label,
                'bbox': bbox,
                'score': score
            }

            coco_results.append(pred_data)
        progress_bar.update()
    with open(os.path.join(args.out_dir, f'{args.task}.json'), 'w') as f:
        mmengine.dump(coco_results, f, file_format='json')


if __name__ == '__main__':
    main()
