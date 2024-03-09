import math
import os
import sys
import argparse
import logging
from tqdm import tqdm
import glob
import cv2
import numpy as np
from PIL import Image

import ultralytics.utils.wod_reader as wod_reader
import ultralytics.datasets.wod.config.config as config
from waymo_open_dataset import v2
from waymo_open_dataset.v2.perception.utils import lidar_utils as _lidar_utils

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
cfg = config.load()
logging.basicConfig(level=logging.INFO)


def prepare_yolov8_input_dir(path):
    if not os.path.exists(path + '/images'):
        logging.info('Create images directory')
        os.makedirs(path + '/images/training')
        os.makedirs(path + '/images/testing')
        os.makedirs(path + '/images/validation')
    if not os.path.exists(path + '/labels'):
        logging.info('Create labels directory')
        os.makedirs(path + '/labels/training')
        os.makedirs(path + '/labels/testing')
        os.makedirs(path + '/labels/validation')


def pcl_to_bev(pcl):
    cfg = config.load()
    pcl_npa = pcl.numpy()
    mask = np.where((pcl_npa[:, 0] >= cfg.range_x[0]) & (pcl_npa[:, 0] <= cfg.range_x[1]) &
                    (pcl_npa[:, 1] >= cfg.range_y[0]) & (pcl_npa[:, 1] <= cfg.range_y[1]) &
                    (pcl_npa[:, 2] >= cfg.range_z[0]) & (pcl_npa[:, 2] <= cfg.range_z[1]))
    pcl_npa = pcl_npa[mask]

    # compute bev-map discretization by dividing x-range by the bev-image height
    bev_discrete = (cfg.range_x[1] - cfg.range_x[0]) / cfg.bev_height

    # create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates
    pcl_cpy = np.copy(pcl_npa)
    pcl_cpy[:, 0] = np.int_(np.floor(pcl_cpy[:, 0] / bev_discrete))

    # transform all metrix y-coordinates as well but center the forward-facing x-axis in the middle of the image
    pcl_cpy[:, 1] = np.int_(np.floor(pcl_cpy[:, 1] / bev_discrete) + (cfg.bev_width + 1) / 2)

    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    pcl_cpy[:, 2] = pcl_cpy[:, 2] - cfg.range_z[0]

    # re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then by decreasing height
    idx_height = np.lexsort((-pcl_cpy[:, 2], pcl_cpy[:, 1], pcl_cpy[:, 0]))
    lidar_pcl_hei = pcl_cpy[idx_height]

    # extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    _, idx_height_unique = np.unique(lidar_pcl_hei[:, 0:2], axis=0, return_index=True)
    lidar_pcl_hei = lidar_pcl_hei[idx_height_unique]

    # assign the height value of each unique entry in lidar_top_pcl to the height map and
    # make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    height_map = np.zeros((cfg.bev_height + 1, cfg.bev_width + 1))
    height_map[np.int_(lidar_pcl_hei[:, 0]), np.int_(lidar_pcl_hei[:, 1])] = lidar_pcl_hei[:, 2] / float(
        np.abs(cfg.range_z[1] - cfg.range_z[0]))

    # sort points such that in case of identical BEV grid coordinates, the points in each grid cell are arranged based on their intensity
    pcl_cpy[pcl_cpy[:, 2] > 1.0, 2] = 1.0
    idx_intensity = np.lexsort((-pcl_cpy[:, 2], pcl_cpy[:, 1], pcl_cpy[:, 0]))
    pcl_cpy = pcl_cpy[idx_intensity]

    # only keep one point per grid cell
    _, indices = np.unique(pcl_cpy[:, 0:2], axis=0, return_index=True)
    lidar_pcl_int = pcl_cpy[indices]

    # create the intensity map
    intensity_map = np.zeros((cfg.bev_height + 1, cfg.bev_width + 1))
    intensity_map[np.int_(lidar_pcl_int[:, 0]), np.int_(lidar_pcl_int[:, 1])] = lidar_pcl_int[:, 2] / (
            np.amax(lidar_pcl_int[:, 2]) - np.amin(lidar_pcl_int[:, 2]))

    # Compute density layer of the BEV map
    density_map = np.zeros((cfg.bev_height + 1, cfg.bev_width + 1))
    _, _, counts = np.unique(lidar_pcl_int[:, 0:2], axis=0, return_index=True, return_counts=True)
    normalized_counts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    density_map[np.int_(lidar_pcl_int[:, 0]), np.int_(lidar_pcl_int[:, 1])] = normalized_counts

    bev_map = np.zeros((3, cfg.bev_height, cfg.bev_width))
    bev_map[2, :, :] = density_map[:cfg.bev_height, :cfg.bev_width]  # r_map
    bev_map[1, :, :] = height_map[:cfg.bev_height, :cfg.bev_width]  # g_map
    bev_map[0, :, :] = intensity_map[:cfg.bev_height, :cfg.bev_width]  # b_map

    bev_map = (np.transpose(bev_map, (1, 2, 0)) * 255).astype(np.uint8)

    return bev_map


def crop_bbox(x, y, size_x, size_y):
    if x - size_x / 2 < 0:
        if x < 0:
            size_x = x + size_x / 2
        if x > 0:
            size_x = (size_x / 2 - x) + size_x / 2
        x = size_x / 2
    if x + size_x / 2 > cfg.bev_width:
        if x < cfg.bev_width:
            size_x = (cfg.bev_width - x) + size_x / 2
        if x > cfg.bev_width:
            size_x = size_x / 2 - (x - cfg.bev_width)
        x = cfg.bev_width - size_x / 2
    if y - size_y / 2 < 0:
        if y < 0:
            size_y = y + size_y / 2
        if y > 0:
            size_y = (size_y / 2 - y) + size_y / 2
        y = size_y / 2
    if y + size_y / 2 > cfg.bev_height:
        if y < cfg.bev_height:
            size_y = (cfg.bev_height - y) + size_y / 2
        if y > cfg.bev_height:
            size_y = size_y / 2 - (y - cfg.bev_height)
        y = cfg.bev_height - size_y / 2

    return x, y, size_x, size_y


def rotate_point(x, y, cx, cy, theta):
    # Rotate point (x, y) around center (cx, cy) by angle (in radians)
    x_rel = x - cx
    y_rel = y - cy
    x_new = x_rel * math.cos(theta) - y_rel * math.sin(theta) + cx
    y_new = x_rel * math.sin(theta) + y_rel * math.cos(theta) + cy
    return x_new, y_new


def calculate_rotated_bbox_coordinates(cx, cy, width, height, theta):
    # Calculate the coordinates of the four corners of the rectangle
    x1 = cx - width / 2
    y1 = cy - height / 2
    x2 = cx + width / 2
    y2 = cy - height / 2
    x3 = cx + width / 2
    y3 = cy + height / 2
    x4 = cx - width / 2
    y4 = cy + height / 2

    # Rotate the four corners around the center
    x1_new, y1_new = rotate_point(x1, y1, cx, cy, theta)
    x2_new, y2_new = rotate_point(x2, y2, cx, cy, theta)
    x3_new, y3_new = rotate_point(x3, y3, cx, cy, theta)
    x4_new, y4_new = rotate_point(x4, y4, cx, cy, theta)

    return x1_new, y1_new, x2_new, y2_new, x3_new, y3_new, x4_new, y4_new


def create_label_list(lidar_box):
    discrete = (cfg.range_x[1] - cfg.range_x[0]) / cfg.bev_width
    bboxes = []
    clss = []

    for i, (object_id, object_type, x, size_x, y, size_y, yaw) in enumerate(zip(
            lidar_box.key.laser_object_id, lidar_box.type, lidar_box.box.center.x, lidar_box.box.size.x,
            lidar_box.box.center.y,
            lidar_box.box.size.y, lidar_box.box.heading
    )):
        x = x / discrete
        y = (-y / discrete) + cfg.bev_width / 2

        size_x = size_x / discrete
        size_y = size_y / discrete

        if ((x + size_x / 2) < 0 or (x - size_x / 2) > cfg.bev_width) or (
                (y + + size_y / 2) < 0 or (y - size_y / 2) > cfg.bev_height):
            continue

        x, y, size_x, size_y = crop_bbox(x, y, size_x, size_y)
        x1, y1, x2, y2, x3, y3, x4, y4 = calculate_rotated_bbox_coordinates(x, y, size_x, size_y, -yaw)

        bboxes.append(
            [x1 / cfg.bev_width, y1 / cfg.bev_height, x2 / cfg.bev_width, y2 / cfg.bev_height, x3 / cfg.bev_width,
             y3 / cfg.bev_height, x4 / cfg.bev_width, y4 / cfg.bev_height])
        clss.append(object_type)

    return clss, bboxes


def convert(dataset_root_dir, output_dir, dir, testing=False):
    # dataset_dir = dataset_root_dir + "/data/" + dir
    dataset_dir = dataset_root_dir + "/" + dir
    context_names = [os.path.splitext(os.path.basename(name))[0] for name in glob.glob(dataset_dir + "/lidar/*.*")]
    laser_name = 1
    pbar = tqdm(context_names, total=len(context_names), disable=LOCAL_RANK > 0)
    for context_name in pbar:
        lidar_df = wod_reader.read_lidar_df(dataset_dir, context_name, laser_name)
        lidar_box_df = wod_reader.read_lidar_box_df(dataset_dir, context_name, laser_name)
        lidar_calibration_df = wod_reader.read_lidar_calibration_df(dataset_dir, context_name, laser_name)
        lidar_pose_df = wod_reader.read_lidar_pose_df(dataset_dir, context_name, laser_name)
        vehicle_pose_df = wod_reader.read_vehicle_pose_df(dataset_dir, context_name)

        df = lidar_df[lidar_df['key.laser_name'] == laser_name]
        if not testing:
            df = v2.merge(df, lidar_box_df, right_group=True)
        df = v2.merge(df, lidar_calibration_df)
        df = v2.merge(df, lidar_pose_df)
        df = v2.merge(df, vehicle_pose_df)

        for i, (_, r) in enumerate(df.iterrows()):
            lidar = v2.LiDARComponent.from_dict(r)
            if not testing:
                lidar_box = v2.LiDARBoxComponent.from_dict(r)
            lidar_calibration = v2.LiDARCalibrationComponent.from_dict(r)
            lidar_pose = v2.LiDARPoseComponent.from_dict(r)
            vehicle_pose = v2.VehiclePoseComponent.from_dict(r)

            pcl = _lidar_utils.convert_range_image_to_point_cloud(lidar.range_image_return1, lidar_calibration,
                                                                  lidar_pose.range_image_return1, vehicle_pose)
            bev_img = pcl_to_bev(pcl)
            bev_img = cv2.rotate(bev_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img = Image.fromarray(bev_img).convert('RGB')
            img.save(output_dir + "/images/" + dir + os.sep + context_name + "_" + str(i) + ".png")
            if not testing:
                clss, bboxes = create_label_list(lidar_box)
                with open(output_dir + "/labels/" + dir + os.sep + context_name + "_" + str(i) + ".txt",
                          'w') as f:
                    for j, (cls, bbox) in enumerate(zip(clss, bboxes)):
                        f.write(str(cls) + ' ' + ' '.join(str(e) for e in bbox) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--wod_dataset', help="WOD dataset root dir path")
    parser.add_argument('-o', '--pc_obb_dataset', help="Output PC OBB dataset root dir path")
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    prepare_yolov8_input_dir(args.pc_obb_dataset)
    convert(args.wod_dataset, args.pc_obb_dataset, "training")
    convert(args.wod_dataset, args.pc_obb_dataset, "validation")
    convert(args.wod_dataset, args.pc_obb_dataset, "testing", True)


if __name__ == '__main__':
    logging.info('Start conversion')
    main()
    logging.info('Conversion finished')
