#!/usr/bin/env python3
import os, sys, cv2, threading, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

SHOW_W, SHOW_H = 1280, 720
CAMERA_ID_TO_NAME = {0: "camera_front_4k", 1: "camera_front", 2: "camera_rear", 3: "camera_front_left",
                     4: "camera_front_right", 5: "camera_rear_left", 6: "camera_rear_right"}

class KeySaveViewer(Node):
    def __init__(self, cam_id: int):
        super().__init__(f'camera{cam_id}_viewer')
        self.cam_id = cam_id
        cam_name = CAMERA_ID_TO_NAME[cam_id]
        self.save_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'data',
                         f'{cam_name}', 'image_undistort.png'))
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        self.br = CvBridge()
        self.latest_raw = None
        self.lock = threading.Lock()
        self.quit_ev = threading.Event()          # ---- 退出标志 ----

        topic = f'/sensing/camera/camera{cam_id}/camera_image'
        self.sub = self.create_subscription(Image, topic, self.callback, 10)

        self.key_thread = threading.Thread(target=self.key_loop, daemon=True)
        self.key_thread.start()
        self.get_logger().info(f'ESC=quit   s=save -> {self.save_path}')

    def callback(self, msg: Image):
        try:
            raw = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge exception: {e}')
            return
        with self.lock:
            self.latest_raw = raw.copy()
        display = cv2.resize(raw, (SHOW_W, SHOW_H))
        cv2.imshow('CameraImage', display)

    # 键盘线程：超时轮询，既能取键又能定期检查退出
    def key_loop(self):
        while not self.quit_ev.is_set() and rclpy.ok():
            key = cv2.waitKey(50) & 0xFF          # 50 ms 超时
            if key == 27:                         # ESC
                self.get_logger().info('ESC pressed – exit.')
                self.quit_ev.set()
                os._exit(0)
            elif key == ord('s'):
                with self.lock:
                    if self.latest_raw is not None:
                        ret = cv2.imwrite(self.save_path, self.latest_raw)
                        self.get_logger().info(
                            f'Saved: {self.save_path}  (ret={ret})')
                    else:
                        self.get_logger().warn('No image yet!')

def main():
    if len(sys.argv) != 2:
        print('Usage: python3 view_cam_key.py <camera_id 0-6>')
        sys.exit(1)
    cam_id = int(sys.argv[1])
    assert cam_id in range(0, 7)

    rclpy.init()
    node = KeySaveViewer(cam_id)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Ctrl-C caught – shutting down …')
    finally:
        node.quit_ev.set()            # 通知键盘线程
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
