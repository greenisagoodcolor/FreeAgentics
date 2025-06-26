import unittest
from datetime import datetime, timedelta

import numpy as np

from agents.base.data_model import (
    Agent,
    AgentCapability,
    AgentGoal,
    AgentPersonality,
    AgentResources,
    AgentStatus,
    Orientation,
    Position,
    ResourceAgent,
    SocialAgent,
    SocialRelationship,
)

"""
Unit tests for Agent Data Model
This module contains comprehensive tests for the Agent data model classes.
"""


class TestPosition(unittest.TestCase):
    """Test Position class"""

    def test_position_creation(self) -> None:
        """Test position creation with default and custom values"""
        pos1 = Position(1.0, 2.0)
        self.assertEqual(pos1.x, 1.0)
        self.assertEqual(pos1.y, 2.0)
        self.assertEqual(pos1.z, 0.0)

        pos2 = Position(1.0, 2.0, 3.0)
        self.assertEqual(pos2.z, 3.0)

    def test_position_to_array(self) -> None:
        """Test conversion to numpy array"""
        pos = Position(1.0, 2.0, 3.0)
        arr = pos.to_array()
        np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.0]))

    def test_distance_calculation(self) -> None:
        """Test distance calculation between positions"""
        pos1 = Position(0.0, 0.0, 0.0)
        pos2 = Position(3.0, 4.0, 0.0)
        self.assertAlmostEqual(pos1.distance_to(pos2), 5.0)

        pos3 = Position(1.0, 1.0, 1.0)
        pos4 = Position(2.0, 2.0, 2.0)
        expected_distance = np.sqrt(3)
        self.assertAlmostEqual(pos3.distance_to(pos4), expected_distance)


class TestOrientation(unittest.TestCase):
    """Test Orientation class"""

    def test_orientation_creation(self) -> None:
        """Test orientation creation with default quaternion"""
        orient = Orientation()
        self.assertEqual(orient.w, 1.0)
        self.assertEqual(orient.x, 0.0)
        self.assertEqual(orient.y, 0.0)
        self.assertEqual(orient.z, 0.0)

    def test_orientation_to_euler(self) -> None:
        """Test quaternion to Euler angle conversion"""
        orient = Orientation()
        roll, pitch, yaw = orient.to_euler()
        self.assertAlmostEqual(roll, 0.0)
        self.assertAlmostEqual(pitch, 0.0)
        self.assertAlmostEqual(yaw, 0.0)


if __name__ == "__main__":
    unittest.main()
