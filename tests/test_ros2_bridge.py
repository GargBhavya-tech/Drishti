"""
tests/test_ros2_bridge.py

ros2_bridge/drishti_bridge.py is an explicitly UNTESTED-end-to-end
architecture skeleton (see its own module docstring) -- there is no ROS
2 install in this project's dev environment. These tests check the
parts that ARE meaningfully testable without one: the module imports
cleanly, the topic-name contract is stable, and attempting to actually
construct the node without ROS 2 fails with a clear, actionable error
rather than a bare ImportError or a silent no-op.
"""

from __future__ import annotations

import pytest

from ros2_bridge import drishti_bridge as bridge


def test_module_imports_without_ros2_installed():
    # If this test file collected at all, the module already imported
    # successfully regardless of whether rclpy is present -- this
    # assertion just makes that intent explicit.
    assert bridge is not None


def test_topic_contract_is_stable():
    assert bridge.POINTCLOUD_TOPIC == "/ouster/points"
    assert bridge.COSTMAP_TOPIC == "/drishti/foveated_costmap"
    assert bridge.CMD_VEL_TOPIC == "/drishti/cmd_vel"
    assert bridge.SAFE_SPEED_TOPIC == "/drishti/safe_speed_limit"


def test_require_ros2_raises_actionable_error_when_unavailable(monkeypatch):
    monkeypatch.setattr(bridge, "_ROS2_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="interface skeleton"):
        bridge.require_ros2()


def test_require_ros2_passes_when_available(monkeypatch):
    monkeypatch.setattr(bridge, "_ROS2_AVAILABLE", True)
    bridge.require_ros2()  # must not raise


def test_node_construction_fails_clearly_without_ros2(monkeypatch):
    monkeypatch.setattr(bridge, "_ROS2_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="interface skeleton"):
        bridge.DrishtiBridgeNode()
