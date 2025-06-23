from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='camera_manager',
            executable='camera_manager_node',
            name='camera_manager',
            output='screen'
        ),
    ])