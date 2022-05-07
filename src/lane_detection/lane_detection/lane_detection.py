import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import ros2_numpy as rnp
from lane_detection_msgs.msg import LaneDetectionArray, LaneDetection

import torch
from collections import OrderedDict
from .erfnet import ERFNet
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_grad_enabled(False)

class LaneDetectionNode(Node):

    def __init__(self):
        super().__init__('lane_detection_node')

        # parameter declarations
        self.declare_parameter('image_size_x', 1280)
        self.declare_parameter('image_size_y', 720)
        self.declare_parameter('focal_x', 1280/2)
        self.declare_parameter('focal_y', 720/2)
        self.declare_parameter('center_x', 1280/2)
        self.declare_parameter('center_y', 720/2)
        self.declare_parameter('rotation_x', 0.0)
        self.declare_parameter('rotation_y', 0.0)
        self.declare_parameter('rotation_z', 0.0)
        self.declare_parameter('translation_x', 0.0)
        self.declare_parameter('translation_y', 0.0)
        self.declare_parameter('translation_z', 0.0)
        self.declare_parameter('weights_file', 'data/perception/erfnet_baseline_tusimple.pt')
        self.declare_parameter('lane_conf_thresh', 0.5)
        self.declare_parameter('prob_conf_thresh', 0.3)
        self.declare_parameter('size_thresh', 0)
        self.declare_parameter('rsquared_thresh', 0.9)

        # get image resolution
        self.image_size_x = self.get_parameter('image_size_x').get_parameter_value().integer_value
        self.image_size_y = self.get_parameter('image_size_y').get_parameter_value().integer_value

        # get intrinsics
        focal_x = self.get_parameter('focal_x').get_parameter_value().double_value
        focal_y = self.get_parameter('focal_y').get_parameter_value().double_value
        center_x = self.get_parameter('center_x').get_parameter_value().double_value
        center_y = self.get_parameter('center_y').get_parameter_value().double_value

        intrinsic_matrix = np.array([
            [focal_x, 0, center_x, 0],
            [0, focal_y, center_y, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.inv_int = np.linalg.inv(intrinsic_matrix)

        # get extrinsics
        rot_x = self.get_parameter('rotation_x').get_parameter_value().double_value
        rot_y = self.get_parameter('rotation_y').get_parameter_value().double_value
        rot_z = self.get_parameter('rotation_z').get_parameter_value().double_value
        trans_x = self.get_parameter('translation_x').get_parameter_value().double_value
        trans_y = self.get_parameter('translation_y').get_parameter_value().double_value
        trans_z = self.get_parameter('translation_z').get_parameter_value().double_value

        ## matrix to convert to ros coordinate frame convention
        camera_to_world_matrix = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ])

        ## matrix to transform to base link
        rotation_matrix = R.from_euler('xyz', [rot_x, rot_y, rot_z], degrees=True).as_matrix()
        translation_vector = np.array([[trans_x], [trans_y], [trans_z]])
        bl_transform_matrix = np.concatenate([rotation_matrix, translation_vector], axis=1)
        bl_transform_matrix = np.concatenate((bl_transform_matrix,[[0,0,0,1]]), axis=0)
        self.extrinsic_matrix = np.matmul(camera_to_world_matrix, bl_transform_matrix)

        # lane detection param initialization
        self.camera_height = -trans_z
        self.ground_normal = np.array([[0,1,0,0]])
        self.weights_file = self.get_parameter('weights_file').get_parameter_value().string_value
        self.lane_conf_thresh = self.get_parameter('lane_conf_thresh').get_parameter_value().double_value
        self.prob_conf_thresh = self.get_parameter('prob_conf_thresh').get_parameter_value().double_value
        self.size_thresh = self.get_parameter('size_thresh').get_parameter_value().integer_value
        self.rsquared_thresh = self.get_parameter('rsquared_thresh').get_parameter_value().double_value

        # initialize model
        self.model = self.get_model()
        self.get_logger().info('Successfully loaded model')
        
        self.color_image_sub = self.create_subscription(
            Image, 
            '/color_image',
            self.color_image_callback,
            10
        )

        self.lane_detections_pub = self.create_publisher(
            LaneDetectionArray,
            '/lane_detections',
            10
        )

    def get_model(self):
        model = ERFNet(num_classes=7)
        checkpoint = torch.load(self.weights_file, map_location=device)
        checkpoint['model'] = OrderedDict(
            (k.replace('aux_head', 'lane_classifier') if 'aux_head' in k else k, v)
            for k, v in checkpoint['model'].items())
        model.load_state_dict(checkpoint['model'], strict=True)
        model.to(device)
        model.eval()
        return model
                

    def color_image_callback(self, color_image_msg: Image):
        # pre-processing 
        image = rnp.numpify(color_image_msg)
        image = image[..., :3]
        image = cv2.resize(image, (640, 360))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image = image / 255.0
        tensor = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).to(device)

        # inference
        outputs = self.model(tensor)

        # post-processing
        prob_map = torch.nn.functional.interpolate(outputs['out'], size=(720, 1280), \
            mode='bilinear', align_corners=True).softmax(dim=1)
        existence_conf = outputs['lane'].sigmoid()[0].cpu().numpy()
        existence = existence_conf > self.lane_conf_thresh
        existing_lane_labels = existence.nonzero()[0]

        prob_map, label_map = torch.max(prob_map.squeeze(), dim=0)
        prob_map = prob_map.detach().cpu().numpy()
        label_map = label_map.detach().cpu().numpy()
        label_map[prob_map < self.prob_conf_thresh] = 0

        # get lane by doing polynomial fitting
        lane_detection_array = LaneDetectionArray()
        for lane_label in existing_lane_labels:
            # project lane into 3-d space, assuming lanes are at world_z = 0
            uv = np.argwhere(label_map==lane_label+1)
            probs = prob_map[uv[:,0], uv[:, 1]]
            uv[:, [0, 1]] = uv[:, [1, 0]]
            uv = np.hstack([uv, np.ones_like(uv)])
            inv_int_uv = np.matmul(self.inv_int, uv.transpose())
            z_estimate = self.camera_height / np.matmul(self.ground_normal, inv_int_uv)
            world_proj = np.matmul(inv_int_uv.transpose(), self.extrinsic_matrix) * z_estimate.transpose()
            xyp = np.hstack([world_proj[:, :2], probs[..., np.newaxis]])

            if xyp.size > self.size_thresh:
                # polynomial fitting
                coeffs = np.polyfit(xyp[:, 0], xyp[:, 1], deg=3, w=xyp[:, 2])
                polynomial = np.poly1d(coeffs)

                # get rsquared, filter out lanes that are not good fits
                yhat = polynomial(xyp[:, 0])  
                ybar = np.sum(xyp[:,1])/len(xyp[:,1])          # or sum(y)/len(y)
                ssreg = np.sum((yhat-ybar)**2)                 # or sum([ (yihat - ybar)**2 for yihat in yhat])
                sstot = np.sum((xyp[:,1] - ybar)**2)
                determination = ssreg / sstot
                if (determination < self.rsquared_thresh):
                    continue
                
                # make msg
                lane_detection = LaneDetection()
                lane_detection.label = int(lane_label)
                lane_detection.confidence = existence_conf[lane_label].item()
                lane_detection.coeffs = [p for p in coeffs]
                lane_detection_array.lane_detections.append(lane_detection)
        
        # publish lane detections
        if len(lane_detection_array.lane_detections) != 0:
            lane_detection_array.header.stamp = self.get_clock().now().to_msg()
            lane_detection_array.header.frame_id = 'base_link'
            self.lane_detections_pub.publish(lane_detection_array)


def main(args=None):
    rclpy.init(args=args)

    inference_node = LaneDetectionNode()

    rclpy.spin(inference_node)

    inference_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()