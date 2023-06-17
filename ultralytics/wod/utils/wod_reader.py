import tensorflow as tf
import dask.dataframe as dd
from waymo_open_dataset import v2


def read_df(dataset_dir: str, context_name: str, tag: str) -> dd.DataFrame:
    """Creates a Dask DataFrame for the component specified by its tag."""
    paths = tf.io.gfile.glob(f'{dataset_dir}/{tag}/training_{tag}_{context_name}.parquet')
    return dd.read_parquet(paths)


def read_cam_img_cam_box_df(dataset_dir: str, context_name: str, camera_name: int):
    cam_img_df = read_df(dataset_dir, context_name, 'camera_image')
    cam_box_df = read_df(dataset_dir, context_name, 'camera_box')

    # Join all DataFrames using matching columns
    cam_img_df = cam_img_df[cam_img_df['key.camera_name'] == camera_name]
    cam_img_cam_box_df = v2.merge(cam_img_df, cam_box_df, right_group=True)

    return cam_img_cam_box_df


def read_lidar_df(dataset_dir: str, context_name: str, laser_name: int):
    lidar_df = read_df(dataset_dir, context_name, 'lidar')
    lidar_df = lidar_df[lidar_df['key.laser_name'] == laser_name]

    return lidar_df


def read_lidar_lidar_box_df(dataset_dir: str, context_name: str, laser_name: int):
    lidar_df = read_df(dataset_dir, context_name, 'lidar')
    lidar_box_df = read_df(dataset_dir, context_name, 'lidar_box')

    # Join all DataFrames using matching columns
    lidar_df = lidar_df[lidar_df['key.laser_name'] == laser_name]
    lidar_lidar_box_df = v2.merge(lidar_df, lidar_box_df, right_group=True)

    return lidar_lidar_box_df


def read_lidar_calibration_df(dataset_dir: str, context_name: str, laser_name: int):
    lidar_calibration_df = read_df(dataset_dir, context_name, 'lidar_calibration')
    lidar_calibration_df = lidar_calibration_df[lidar_calibration_df['key.laser_name'] == laser_name]

    return lidar_calibration_df


def read_lidar_pose_df(dataset_dir: str, context_name: str, laser_name: int):
    lidar_pose_df = read_df(dataset_dir, context_name, 'lidar_pose')
    lidar_pose_df = lidar_pose_df[lidar_pose_df['key.laser_name'] == laser_name]

    return lidar_pose_df


def read_vehicle_pose_df(dataset_dir: str, context_name: str):
    vehicle_pose_df = read_df(dataset_dir, context_name, 'vehicle_pose')

    return vehicle_pose_df
