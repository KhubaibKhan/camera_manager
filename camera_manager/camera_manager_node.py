#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import pyrealsense2 as rs
import threading
import numpy as np
from cv_bridge import CvBridge
from pathlib import Path


class CameraManagerNode(Node):
    def __init__(self):
        super().__init__('camera_manager')
        self.bridge = CvBridge()
        # Subscribe to comma-separated list of serials to activate
        self.create_subscription(String, 'camera_select', self.select_callback, 10)

        # Discover RealSense devices
        self.ctx = rs.context()
        devs = self.ctx.query_devices()
        if len(devs) == 0:
            self.get_logger().error('No RealSense devices found!')
            raise RuntimeError('No devices')
        
        # global lock
        self.mutex = threading.Lock()

        # Prepare pipelines and a separate publisher dictionary
        self.pipelines = {}
        self.pub_dict = {}
        for dev in devs:
            serial = dev.get_info(rs.camera_info.serial_number)
            # Query supported video profiles to find native resolutions
            color_profiles = [p.as_video_stream_profile()
                              for s in dev.query_sensors()
                              for p in s.get_stream_profiles()
                              if p.stream_type() == rs.stream.color and p.is_video_stream_profile()]
            depth_profiles = [p.as_video_stream_profile()
                              for s in dev.query_sensors()
                              for p in s.get_stream_profiles()
                              if p.stream_type() == rs.stream.depth and p.is_video_stream_profile()]
            # Pick highest-res color & depth
            best_color = max(color_profiles, key=lambda p: p.width() * p.height())
            best_depth = max(depth_profiles, key=lambda p: p.width() * p.height())
            cw, ch, cf = best_color.width(), best_color.height(), int(best_color.fps())
            dw, dh, df = best_depth.width(), best_depth.height(), int(best_depth.fps())

            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, cf)
            cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, df)
            pipe = rs.pipeline()

            self.pipelines[serial] = {
                'config': cfg,
                'pipeline': pipe,
                'running': False
            }
            # sanitize serial token
            safe = serial if serial[0].isalpha() else f"id_{serial}"
            color_topic = f'camera/{safe}/color/image_raw'
            depth_topic = f'camera/{safe}/depth/image_raw'
            # Use pub_dict instead of a Node property
            self.pub_dict[serial] = {
                'color': self.create_publisher(Image, color_topic, 10),
                'depth': self.create_publisher(Image, depth_topic, 10)
            }

        self.active_serials = set()
        thread = threading.Thread(target=self.capture_loop, daemon=True)
        thread.start()

    def select_callback(self, msg: String):
        requested = {tok.strip() for tok in msg.data.split(',') if tok.strip()}
        with self.mutex:
            # ── ① On-demand creation for any new .bag tokens ───────────────
            for tok in requested:
                if tok.endswith('.bag') and Path(tok).is_file() and tok not in self.pipelines:
                    cfg, pipe = rs.config(), rs.pipeline()
                    cfg.enable_device_from_file(str(Path(tok).expanduser()),
                                                repeat_playback=False)
                    safe = Path(tok).stem
                    self.pipelines[tok] = {
                        'config':   cfg,
                        'pipeline': pipe,
                        'running':  False,
                        'dtype':    'bag'
                    }
                    self.pub_dict[tok] = {
                        'color': self.create_publisher(
                            Image, f'camera/{safe}/color/image_raw', 10),
                        'depth': self.create_publisher(
                            Image, f'camera/{safe}/depth/image_raw', 10)
                    }

            # ──Start requested cameras (live or bag) ───────────────────
            for tok in requested - self.active_serials:
                if tok in self.pipelines:
                    self.get_logger().info(f"Starting camera {tok}")
                    profile = self.pipelines[tok]['pipeline'].start(
                        self.pipelines[tok]['config'])
                    # if this is a playback bag, switch to non-real-time mode
                    if self.pipelines[tok].get('dtype') == 'bag':
                        profile.get_device().as_playback().set_real_time(False)
                    self.pipelines[tok]['running'] = True

            #Stop pipelines no longer requested
            for tok in self.active_serials - requested:
                bundle = self.pipelines.get(tok)
                if bundle and bundle['running']:
                    bundle['running'] = False
                    bundle['pipeline'].stop()
                    self.get_logger().info(f"Stopped {tok}")

            # Update active set ──────────────────────────────────────────
            self.active_serials = requested

    def capture_loop(self):
        while rclpy.ok():
            with self.mutex:                    # single atomic snapshot
                active_now = [s for s in self.active_serials
                               if self.pipelines[s]['running']]
            for serial in list(self.active_serials):
                if not self.pipelines[serial]['running']:
                    continue
                pipe = self.pipelines[serial]['pipeline']
                try:
                    frames = pipe.wait_for_frames()
                except rs.error as e:
                    # if a bag playback hits EOF, stop that pipeline cleanly
                    if 'end of file' in str(e).lower():
                        self.get_logger().info(f"Bag finished: {serial}")
                        self.pipelines[serial]['running'] = False
                        continue
                    # otherwise log any other frame error
                    self.get_logger().error(f"Error capturing frames from {serial}: {e}")
                    continue
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if color_frame:
                    img = np.asanyarray(color_frame.get_data())
                    color_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
                    self.pub_dict[serial]['color'].publish(color_msg)
                if depth_frame:
                    depth_img = np.asanyarray(depth_frame.get_data())
                    depth_msg = self.bridge.cv2_to_imgmsg(depth_img, encoding='16UC1')
                    self.pub_dict[serial]['depth'].publish(depth_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CameraManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
