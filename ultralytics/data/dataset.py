# Ultralytics YOLO 🚀, AGPL-3.0 license

from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm
import psutil

import glob
import os
from waymo_open_dataset import v2
from waymo_open_dataset.v2.perception.utils import lidar_utils as _lidar_utils
import ultralytics.datasets.wod.utils.wod_reader as wod_reader
import ultralytics.datasets.wod.config.config as config

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM_BAR_FORMAT, is_dir_writeable
from .augment import Compose, Format, Instances, LetterBox, classify_albumentations, classify_transforms, v8_transforms
from .base import BaseDataset
from .utils import HELP_URL, LOGGER, get_hash, img2label_paths, verify_image_label


class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        use_segments (bool, optional): If True, segmentation masks are used as labels. Defaults to False.
        use_keypoints (bool, optional): If True, keypoints are used as labels. Defaults to False.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """
    cache_version = '1.0.2'  # dataset labels *.cache version, >= 1.0.0 for YOLOv8
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(self, *args, data=None, use_segments=False, use_keypoints=False, **kwargs):
        self.use_segments = use_segments
        self.use_keypoints = use_keypoints
        self.data = data
        assert not (self.use_segments and self.use_keypoints), 'Can not use both segments and keypoints.'
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path('./labels.cache')):
        """Cache dataset labels, check images and read shapes.
        Args:
            path (Path): path where to save the cache file (default: Path('./labels.cache')).
        Returns:
            (dict): labels.
        """
        x = {'labels': []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f'{self.prefix}Scanning {path.parent / path.stem}...'
        total = len(self.im_files)
        nkpt, ndim = self.data.get('kpt_shape', (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
            raise ValueError("'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                             "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image_label,
                                iterable=zip(self.im_files, self.label_files, repeat(self.prefix),
                                             repeat(self.use_keypoints), repeat(len(self.data['names'])), repeat(nkpt),
                                             repeat(ndim)))
            pbar = tqdm(results, desc=desc, total=total, bar_format=TQDM_BAR_FORMAT)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x['labels'].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb[:, 0:1],  # n, 1
                            bboxes=lb[:, 1:],  # n, 4
                            segments=segments,
                            keypoints=keypoint,
                            normalized=True,
                            bbox_format='xywh'))
                if msg:
                    msgs.append(msg)
                pbar.desc = f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            pbar.close()

        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.im_files)
        x['results'] = nf, nm, ne, nc, len(self.im_files)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        if is_dir_writeable(path.parent):
            if path.exists():
                path.unlink()  # remove *.cache file if exists
            np.save(str(path), x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{self.prefix}New cache created: {path}')
        else:
            LOGGER.warning(f'{self.prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.')
        return x

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')
        try:
            import gc
            gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
            cache, exists = np.load(str(cache_path), allow_pickle=True).item(), True  # load dict
            gc.enable()
            assert cache['version'] == self.cache_version  # matches current version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            tqdm(None, desc=self.prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        if nf == 0:  # number of labels found
            raise FileNotFoundError(f'{self.prefix}No labels found in {cache_path}, can not start training. {HELP_URL}')

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels = cache['labels']
        assert len(labels), f'No valid labels found, please check your dataset. {HELP_URL}'
        self.im_files = [lb['im_file'] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb['cls']), len(lb['bboxes']), len(lb['segments'])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f'WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, '
                f'len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. '
                'To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.')
            for lb in labels:
                lb['segments'] = []
        if len_cls == 0:
            raise ValueError(f'All labels empty in {cache_path}, can not start training without labels. {HELP_URL}')
        return labels

    # TODO: use hyp config to set all these augmentations
    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """custom your label format here."""
        # NOTE: cls is not with bboxes now, classification and semantic segmentation need an independent cls label
        # we can make it also support classification and semantic segmentation by add or remove some dict keys there.
        bboxes = label.pop('bboxes')
        segments = label.pop('segments')
        keypoints = label.pop('keypoints', None)
        bbox_format = label.pop('bbox_format')
        normalized = label.pop('normalized')
        label['instances'] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == 'img':
                value = torch.stack(value, 0)
            if k in ['masks', 'keypoints', 'bboxes', 'cls']:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch['batch_idx'] = list(new_batch['batch_idx'])
        for i in range(len(new_batch['batch_idx'])):
            new_batch['batch_idx'][i] += i  # add target image index for build_targets()
        new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)
        return new_batch


class WodDataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        use_segments (bool, optional): If True, segmentation masks are used as labels. Defaults to False.
        use_keypoints (bool, optional): If True, keypoints are used as labels. Defaults to False.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """
    cache_version = '1.0.2'  # dataset labels *.cache version, >= 1.0.0 for YOLOv8
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(self, *args, data=None, use_segments=False, use_keypoints=False, mode='', **kwargs):
        self.cfg = config.load()
        self.use_segments = use_segments
        self.use_keypoints = use_keypoints
        self.data = data
        # self.dataset_dir = str(data.get('path')) + "/training"
        self.dataset_dir = str(data.get(mode))
        self.context_names = self.get_context_names(self.dataset_dir)
        self.laser_name = 1
        assert not (self.use_segments and self.use_keypoints), 'Can not use both segments and keypoints.'
        super().__init__(*args, **kwargs)

    def get_context_names(self, dataset_dir):
        # return [os.path.splitext(os.path.basename(name))[0][len("training_lidar_"):]
        #         for name in glob.glob(dataset_dir + "/lidar/*.*")]
        return [os.path.splitext(os.path.basename(name))[0] for name in glob.glob(dataset_dir + "/lidar/*.*")]


    def get_img_files(self, dataset_dir):
        """Read lidar file names"""
        try:
            context_names = []  # image files
            im_files = []
            for p in dataset_dir if isinstance(dataset_dir, list) else [dataset_dir]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    # context_names = [os.path.splitext(os.path.basename(name))[0][len("training_lidar_"):] for name in
                    #      glob.glob(str(p / 'lidar' / '*.*'), recursive=True)]
                    context_names = [os.path.splitext(os.path.basename(name))[0] for name in glob.glob(str(p / 'lidar' / '*.*'), recursive=True)]
                else:
                    raise FileNotFoundError(f'{self.prefix}{p} does not exist')
            ##
            for context_name in context_names:
                lidar_lidar_box_df = wod_reader.read_lidar_lidar_box_df(dataset_dir, context_name, self.laser_name)

                for i, (_, r) in enumerate(lidar_lidar_box_df.iterrows()):
                    lidar = v2.LiDARComponent.from_dict(r)
                    im_files.append(lidar.key.segment_context_name + "#" + str(lidar.key.laser_name) + "#" + str(lidar.key.frame_timestamp_micros))
            ##
            assert im_files, f'{self.prefix}No images found'
        except Exception as e:
            raise FileNotFoundError(f'{self.prefix}Error loading data from {dataset_dir}/lidar\n{HELP_URL}') from e
        if self.fraction < 1:
            im_files = im_files[:round(len(im_files) * self.fraction)]
        return im_files

    # def load_image(self, i):
    #     im = self.ims[i]
    #     h0, w0 = im.shape[:2]  # orig hw
    #     # Add to buffer if training with augmentations
    #     if self.augment:
    #         self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
    #         self.buffer.append(i)
    #         if len(self.buffer) >= self.max_buffer_length:
    #             j = self.buffer.pop(0)
    #             self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

    def cache_images(self, cache):
        """Cache images to memory or disk."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        with ThreadPool(NUM_THREADS) as pool:
            total = len(self.context_names)
            pbar = tqdm(self.context_names, total=total, bar_format=TQDM_BAR_FORMAT, disable=LOCAL_RANK > 0)
            idx = 0
            for context_name in pbar:
                lidar_lidar_box_df = wod_reader.read_lidar_lidar_box_df(self.dataset_dir, context_name, self.laser_name)
                lidar_calibration_df = wod_reader.read_lidar_calibration_df(self.dataset_dir, context_name, self.laser_name)
                lidar_pose_df = wod_reader.read_lidar_pose_df(self.dataset_dir, context_name, self.laser_name)
                vehicle_pose_df = wod_reader.read_vehicle_pose_df(self.dataset_dir, context_name)

                df = lidar_lidar_box_df.merge(lidar_calibration_df)
                df = v2.merge(df, lidar_pose_df)
                df = v2.merge(df, vehicle_pose_df)

                for i, (_, r) in enumerate(df.iterrows()):
                    lidar = v2.LiDARComponent.from_dict(r)
                    lidar_calibration = v2.LiDARCalibrationComponent.from_dict(r)
                    lidar_pose = v2.LiDARPoseComponent.from_dict(r)
                    vehicle_pose = v2.VehiclePoseComponent.from_dict(r)

                    pcl = _lidar_utils.convert_range_image_to_point_cloud(lidar.range_image_return1, lidar_calibration,
                                                                          lidar_pose.range_image_return1, vehicle_pose)
                    bev_img = self.pcl_to_bev(pcl)
                    bev_img = cv2.rotate(bev_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    # img = Image.fromarray(bev_img).convert('RGB')

                    if cache == 'disk':
                        b += self.npy_files[i].stat().st_size
                    else:  # 'ram'
                        self.ims[idx], self.im_hw0[idx], self.im_hw[idx] = bev_img, (640, 640), (640, 640)  # im, hw_orig, hw_resized = load_image(self, i)
                        b += self.ims[idx].nbytes
                    idx += 1
                    pbar.desc = f'{self.prefix}Caching images ({b / gb:.1f}GB {cache})'
            pbar.close()

    # def cache_images_to_disk(self, i):
    #     """Saves an image as an *.npy file for faster loading."""
    #     f = self.npy_files[i]
    #     if not f.exists():
    #         np.save(f.as_posix(), cv2.imread(self.im_files[i]))

    def check_cache_ram(self, safety_margin=0.5):
        """Check image caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            ratio = self.imgsz / max(640, 640)  # max(h, w)  # ratio
            b += 50 * 1024 * ratio ** 2
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required < mem.available  # to cache or not to cache, that is the question
        if not cache:
            LOGGER.info(f'{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images '
                        f'with {int(safety_margin * 100)}% safety margin but only '
                        f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '
                        f"{'caching images ✅' if cache else 'not caching images ⚠️'}")
        return cache

    def pcl_to_bev(self, pcl):
        pcl_npa = pcl.numpy()
        mask = np.where((pcl_npa[:, 0] >= self.cfg.range_x[0]) & (pcl_npa[:, 0] <= self.cfg.range_x[1]) &
                        (pcl_npa[:, 1] >= self.cfg.range_y[0]) & (pcl_npa[:, 1] <= self.cfg.range_y[1]) &
                        (pcl_npa[:, 2] >= self.cfg.range_z[0]) & (pcl_npa[:, 2] <= self.cfg.range_z[1]))
        pcl_npa = pcl_npa[mask]

        # compute bev-map discretization by dividing x-range by the bev-image height
        bev_discrete = (self.cfg.range_x[1] - self.cfg.range_x[0]) / self.cfg.bev_height

        # create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates
        pcl_cpy = np.copy(pcl_npa)
        pcl_cpy[:, 0] = np.int_(np.floor(pcl_cpy[:, 0] / bev_discrete))

        # transform all metrix y-coordinates as well but center the forward-facing x-axis in the middle of the image
        pcl_cpy[:, 1] = np.int_(np.floor(pcl_cpy[:, 1] / bev_discrete) + (self.cfg.bev_width + 1) / 2)

        # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
        pcl_cpy[:, 2] = pcl_cpy[:, 2] - self.cfg.range_z[0]

        # re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then by decreasing height
        idx_height = np.lexsort((-pcl_cpy[:, 2], pcl_cpy[:, 1], pcl_cpy[:, 0]))
        lidar_pcl_hei = pcl_cpy[idx_height]

        # extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
        _, idx_height_unique = np.unique(lidar_pcl_hei[:, 0:2], axis=0, return_index=True)
        lidar_pcl_hei = lidar_pcl_hei[idx_height_unique]

        # assign the height value of each unique entry in lidar_top_pcl to the height map and
        # make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
        height_map = np.zeros((self.cfg.bev_height + 1, self.cfg.bev_width + 1))
        height_map[np.int_(lidar_pcl_hei[:, 0]), np.int_(lidar_pcl_hei[:, 1])] = lidar_pcl_hei[:, 2] / float(
            np.abs(self.cfg.range_z[1] - self.cfg.range_z[0]))

        # sort points such that in case of identical BEV grid coordinates, the points in each grid cell are arranged based on their intensity
        pcl_cpy[pcl_cpy[:, 2] > 1.0, 2] = 1.0
        idx_intensity = np.lexsort((-pcl_cpy[:, 2], pcl_cpy[:, 1], pcl_cpy[:, 0]))
        pcl_cpy = pcl_cpy[idx_intensity]

        # only keep one point per grid cell
        _, indices = np.unique(pcl_cpy[:, 0:2], axis=0, return_index=True)
        lidar_pcl_int = pcl_cpy[indices]

        # create the intensity map
        intensity_map = np.zeros((self.cfg.bev_height + 1, self.cfg.bev_width + 1))
        intensity_map[np.int_(lidar_pcl_int[:, 0]), np.int_(lidar_pcl_int[:, 1])] = lidar_pcl_int[:, 2] / (
                np.amax(lidar_pcl_int[:, 2]) - np.amin(lidar_pcl_int[:, 2]))

        # Compute density layer of the BEV map
        density_map = np.zeros((self.cfg.bev_height + 1, self.cfg.bev_width + 1))
        _, _, counts = np.unique(lidar_pcl_int[:, 0:2], axis=0, return_index=True, return_counts=True)
        normalized_counts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
        density_map[np.int_(lidar_pcl_int[:, 0]), np.int_(lidar_pcl_int[:, 1])] = normalized_counts

        bev_map = np.zeros((3, self.cfg.bev_height, self.cfg.bev_width))
        bev_map[2, :, :] = density_map[:self.cfg.bev_height, :self.cfg.bev_width]  # r_map
        bev_map[1, :, :] = height_map[:self.cfg.bev_height, :self.cfg.bev_width]  # g_map
        bev_map[0, :, :] = intensity_map[:self.cfg.bev_height, :self.cfg.bev_width]  # b_map

        bev_map = (np.transpose(bev_map, (1, 2, 0)) * 255).astype(np.uint8)

        return bev_map

    def create_label(self, lidar, lidar_box):
        discrete = (self.cfg.range_x[1] - self.cfg.range_x[0]) / self.cfg.bev_width
        bboxes = []

        for i, (object_id, object_type, x, size_x, y, size_y, yaw) in enumerate(zip(
                lidar_box.key.laser_object_id, lidar_box.type, lidar_box.box.center.x, lidar_box.box.size.x,
                lidar_box.box.center.y,
                lidar_box.box.size.y, lidar_box.box.heading
        )):
            # Draw the object bounding box.
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)

            x = x / discrete
            y = (-y / discrete) + self.cfg.bev_width / 2

            size_x = size_x / discrete
            size_y = size_y / discrete

            size_x = size_x * cos_yaw + size_y * sin_yaw
            size_y = - size_x * sin_yaw + size_y * cos_yaw

            bboxes.append([x, y, size_x, size_y])
        return dict(
            im_file=lidar.key.segment_context_name + "#" + str(lidar.key.laser_name) + "#" + str(lidar.key.frame_timestamp_micros),
            shape=(self.cfg.bev_width, self.cfg.bev_height),
            ori_shape=(self.cfg.bev_width, self.cfg.bev_height),
            resized_shape=(self.cfg.bev_width, self.cfg.bev_height),
            cls=np.array(lidar_box.type).reshape(len(lidar_box.type), 1),  # n, 1
            bboxes=np.array(bboxes),  # n, 4
            segments=[],
            # keypoints=None,
            normalized=True,
            bbox_format='xywh')

    def cache_labels(self, path=Path('./labels.cache')):
        """Cache dataset labels, check images and read shapes.
        Args:
            path (Path): path where to save the cache file (default: Path('./labels.cache')).
        Returns:
            (dict): labels.
        """
        x = {'labels': []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f'{self.prefix}Scanning {path.parent}/lidar...'
        total = len(self.context_names)
        nkpt, ndim = self.data.get('kpt_shape', (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
            raise ValueError("'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                             "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'")
        with ThreadPool(NUM_THREADS) as pool:
            pbar = tqdm(self.context_names, desc=desc, total=total, bar_format=TQDM_BAR_FORMAT)
            for context_name in pbar:
                lidar_lidar_box_df = wod_reader.read_lidar_lidar_box_df(self.dataset_dir, context_name, self.laser_name)
                lidar_calibration_df = wod_reader.read_lidar_calibration_df(self.dataset_dir, context_name, self.laser_name)
                lidar_pose_df = wod_reader.read_lidar_pose_df(self.dataset_dir, context_name, self.laser_name)
                vehicle_pose_df = wod_reader.read_vehicle_pose_df(self.dataset_dir, context_name)

                df = lidar_lidar_box_df.merge(lidar_calibration_df)
                df = v2.merge(df, lidar_pose_df)
                df = v2.merge(df, vehicle_pose_df)

                for i, (_, r) in enumerate(df.iterrows()):
                    lidar = v2.LiDARComponent.from_dict(r)
                    lidar_box = v2.LiDARBoxComponent.from_dict(r)

                    nf += 1
                    x['labels'].append(self.create_label(lidar, lidar_box))
                nm = 0
                ne = 0
                nc = 0
                msg = ''
                if msg:
                    msgs.append(msg)
                pbar.desc = f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            pbar.close()

        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}')
        x['hash'] = get_hash(self.im_files)
        x['results'] = nf, nm, ne, nc, len(self.context_names)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        if is_dir_writeable(path.parent):
            if path.exists():
                path.unlink()  # remove *.cache file if exists
            np.save(str(path), x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{self.prefix}New cache created: {path}')
        else:
            LOGGER.warning(f'{self.prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.')
        return x

    def get_labels(self):
        """Returns dictionary of labels for WOD training."""
        # self.label_files = img2label_paths(self.im_files)
        # cache_path = Path(str(self.data.get('path')) + "/training/labels").with_suffix('.cache')
        cache_path = Path(str(self.dataset_dir) + "/labels").with_suffix('.cache')
        try:
            import gc
            gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
            cache, exists = np.load(str(cache_path), allow_pickle=True).item(), True  # load dict
            gc.enable()
            assert cache['version'] == self.cache_version  # matches current version
            assert cache['hash'] == get_hash(self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            tqdm(None, desc=self.prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        if nf == 0:  # number of labels found
            raise FileNotFoundError(f'{self.prefix}No labels found in {cache_path}, can not start training. {HELP_URL}')

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels = cache['labels']
        self.im_files = [lb['im_file'] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb['cls']), len(lb['bboxes']), len(lb['segments'])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f'WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, '
                f'len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. '
                'To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.')
            for lb in labels:
                lb['segments'] = []
        if len_cls == 0:
            raise ValueError(f'All labels empty in {cache_path}, can not start training without labels. {HELP_URL}')
        return labels

    # def get_image_and_label(self, index):
    #     self.load_image(index)
    #     """Get and return label information from the dataset."""
    #     label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
    #     label['ratio_pad'] = (label['resized_shape'][0] / label['ori_shape'][0],
    #                           label['resized_shape'][1] / label['ori_shape'][1])  # for evaluation
    #     if self.rect:
    #         label['rect_shape'] = self.batch_shapes[self.batch[index]]
    #     return self.update_labels_info(self.labels[index])

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """custom your label format here."""
        # NOTE: cls is not with bboxes now, classification and semantic segmentation need an independent cls label
        # we can make it also support classification and semantic segmentation by add or remove some dict keys there.
        bboxes = label.pop('bboxes')
        segments = label.pop('segments')
        # keypoints = label.pop('keypoints', None)
        keypoints = None
        bbox_format = label.pop('bbox_format')
        normalized = label.pop('normalized')
        # label['instances'] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        label['instances'] = Instances(bboxes, segments, bbox_format=bbox_format, normalized=normalized)
        return label

    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop('shape') for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        # self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == 'img':
                value = torch.stack(value, 0)
            if k in ['masks', 'keypoints', 'bboxes', 'cls']:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch['batch_idx'] = list(new_batch['batch_idx'])
        for i in range(len(new_batch['batch_idx'])):
            new_batch['batch_idx'][i] += i  # add target image index for build_targets()
        new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)
        return new_batch


# Classification dataloaders -------------------------------------------------------------------------------------------
class ClassificationDataset(torchvision.datasets.ImageFolder):
    """
    YOLO Classification Dataset.

    Args:
        root (str): Dataset path.

    Attributes:
        cache_ram (bool): True if images should be cached in RAM, False otherwise.
        cache_disk (bool): True if images should be cached on disk, False otherwise.
        samples (list): List of samples containing file, index, npy, and im.
        torch_transforms (callable): torchvision transforms applied to the dataset.
        album_transforms (callable, optional): Albumentations transforms applied to the dataset if augment is True.
    """

    def __init__(self, root, args, augment=False, cache=False):
        """
        Initialize YOLO object with root, image size, augmentations, and cache settings.

        Args:
            root (str): Dataset path.
            args (Namespace): Argument parser containing dataset related settings.
            augment (bool, optional): True if dataset should be augmented, False otherwise. Defaults to False.
            cache (bool | str | optional): Cache setting, can be True, False, 'ram' or 'disk'. Defaults to False.
        """
        super().__init__(root=root)
        if augment and args.fraction < 1.0:  # reduce training fraction
            self.samples = self.samples[:round(len(self.samples) * args.fraction)]
        self.cache_ram = cache is True or cache == 'ram'
        self.cache_disk = cache == 'disk'
        self.samples = [list(x) + [Path(x[0]).with_suffix('.npy'), None] for x in self.samples]  # file, index, npy, im
        self.torch_transforms = classify_transforms(args.imgsz)
        self.album_transforms = classify_albumentations(
            augment=augment,
            size=args.imgsz,
            scale=(1.0 - args.scale, 1.0),  # (0.08, 1.0)
            hflip=args.fliplr,
            vflip=args.flipud,
            hsv_h=args.hsv_h,  # HSV-Hue augmentation (fraction)
            hsv_s=args.hsv_s,  # HSV-Saturation augmentation (fraction)
            hsv_v=args.hsv_v,  # HSV-Value augmentation (fraction)
            mean=(0.0, 0.0, 0.0),  # IMAGENET_MEAN
            std=(1.0, 1.0, 1.0),  # IMAGENET_STD
            auto_aug=False) if augment else None

    def __getitem__(self, i):
        """Returns subset of data and targets corresponding to given indices."""
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram and im is None:
            im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f))
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        if self.album_transforms:
            sample = self.album_transforms(image=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))['image']
        else:
            sample = self.torch_transforms(im)
        return {'img': sample, 'cls': j}

    def __len__(self) -> int:
        return len(self.samples)


# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()
