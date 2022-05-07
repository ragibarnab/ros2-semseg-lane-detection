import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from lane_detection_msgs.msg import LaneDetectionArray

import numpy as np


class LaneVizNode(Node):
    def __init__(self):
        super().__init__('lane_viz_node')

        self.lane_detections_sub = self.create_subscription(
            LaneDetectionArray, 
            '/lane_detections',
            self.lane_viz_callback,
            10
        )
        
        self.lane_viz_pub = self.create_publisher(
            MarkerArray,
            '/lane_viz',
            10
        )

    def lane_viz_callback(self, lane_detections_msg: LaneDetectionArray):
        
        marker_array = MarkerArray()
        for lane_detection in lane_detections_msg.lane_detections:
            if lane_detection.label not in [2,3]:
                continue

            marker = Marker()
            marker.header.frame_id = lane_detections_msg.header.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = lane_detection.label
            marker.type = 4
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker.lifetime = Duration(seconds=0.1).to_msg()
            marker.scale.x = 0.25
            marker.ns = 'lane_viz'

            coeffs = np.asarray(lane_detection.coeffs)
            polynomial = np.poly1d(coeffs)
            x_coords = np.arange(5,20,0.1)
            y_coords = polynomial(x_coords)


            for x,y in np.vstack([x_coords,y_coords]).transpose():
                point = Point()
                point.x = x
                point.y = y
                point.z = 0.0
                marker.points.append(point)
        
            marker_array.markers.append(marker)
        self.lane_viz_pub.publish(marker_array)



def main(args=None):
    rclpy.init(args=args)

    lane_viz_node = LaneVizNode()

    rclpy.spin(lane_viz_node)

    lane_viz_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()