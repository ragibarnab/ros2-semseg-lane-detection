from os import path

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    launch_path = path.realpath(__file__)
    launch_dir = path.join(path.dirname(launch_path), '..')
    param_dir = path.join(launch_dir,"param")


    lane_detection = Node(
        package='lane_detection',
        executable='lane_detection_exe',
        name='lane_detection_node',
        parameters=[
            (path.join(param_dir, "perception", "front_camera.param.yaml")),
            (path.join(param_dir, "perception", "lane_detection.param.yaml"))
        ],
        remappings=[
            ('/color_image', '/sensors/zed/left_rgb'),
        ]
    )

    lane_viz = Node(
        package='lane_viz',
        executable='lane_viz_exe',
        name='lane_viz_node',
        remappings=[
            ('/lane_detections', '/lane_detections'),
            ('/lane_viz', '/lane_viz'),
        ]
    )

    return LaunchDescription([
        lane_detection,
        lane_viz
    ])