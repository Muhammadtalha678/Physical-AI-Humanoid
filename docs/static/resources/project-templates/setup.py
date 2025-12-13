import os
from setuptools import setup
from glob import glob
import subprocess

package_name = 'robot_control_package'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Robot control package for ROS 2',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_publisher = robot_control_package.sensor_publisher:main',
            'command_subscriber = robot_control_package.command_subscriber:main',
            'status_service = robot_control_package.status_service:main',
        ],
    },
)