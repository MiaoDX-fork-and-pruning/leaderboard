#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Simple Autonomous Agent Example with Obstacle Avoidance
"""

import carla
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
import numpy as np

def get_entry_point():
    return 'SimpleAgent'

class SimpleAgent(AutonomousAgent):
    """
    A simple autonomous agent example with basic obstacle avoidance.
    """

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters.
        """
        self.track = Track.SENSORS  # Set the track to SENSORS
        self._prev_timestamp = 0  # Initialize previous timestamp

    def sensors(self):
        """
        Define the sensor suite required by the agent.
        """
        sensors = [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 300, 'height': 200, 'fov': 100, 'id': 'Center'},
            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'id': 'LIDAR'},
            {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'},
            {'type': 'sensor.other.imu', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'id': 'IMU'}
        ]
        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        control = carla.VehicleControl()
        control.throttle = 0.5  # Set throttle to 50%
        control.brake = 0.0  # No braking

        # Process LIDAR data to detect obstacles
        lidar_data = input_data['LIDAR'][1]  # LIDAR data is the second element in the tuple

        # Check for obstacles in front of the vehicle
        obstacle_detected = self.detect_obstacle(lidar_data)

        if obstacle_detected:
            # If obstacle detected, steer slightly to the left
            control.steer = -0.2  # Steer left
        else:
            # Otherwise, keep driving straight
            control.steer = 0.0  # No steering

        return control

    def detect_obstacle(self, lidar_data):
        """
        Detect obstacles using LIDAR data.
        """
        # Define a threshold distance for detecting obstacles (in meters)
        obstacle_threshold = 5.0

        # Process LIDAR points to check for obstacles in front of the vehicle
        for point in lidar_data:
            # Convert LIDAR point to Cartesian coordinates
            x, y, z = point[:3]

            # Check if the point is within a certain range in front of the vehicle
            if -0.5 < y < 0.5 and 0 < x < obstacle_threshold:
                return True  # Obstacle detected

        return False  # No obstacle detected

    def destroy(self):
        """
        Cleanup.
        """
        print("Destroying the agent...")
