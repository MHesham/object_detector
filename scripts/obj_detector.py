#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from object_detection_msgs.msg import DetectedObject
from object_detection_msgs.msg import DetectedObjectsArray
from azure.cognitiveservices.vision.customvision.prediction import prediction_endpoint
from azure.cognitiveservices.vision.customvision.prediction.prediction_endpoint import models
from cv_bridge import CvBridge, CvBridgeError
import cv2

prediction_key = "YOUR_PREDICTION_KEY"
project_id = "2a538b0b-5978-4bd4-9e62-f603c0f287e9"
prediction_threshold = .40
od_loop_freq_hz = 2


def send_box_overlays(cv2_img, detected_objects_list):
    cv2_overlay_img = cv2_img.copy()
    img_h, img_w, _ = cv2_img.shape

    for box in detected_objects_list:
        p1_x = int(img_w * (box.x_center - box.width / 2.0))
        p1_y = int(img_h * (box.y_center - box.height / 2.0))
        p2_x = int(p1_x + (img_w * box.width))
        p2_y = int(p1_y + (img_h * box.height))
        cv2.rectangle(cv2_overlay_img, (p1_x, p1_y),
                      (p2_x, p2_y), (255, 0, 0), 2)
        txt = '{}:{:.2f}'.format(box.label, box.probability)
        cv2.putText(cv2_overlay_img, txt, (p1_x, p1_y),
                    cv2.FONT_HERSHEY_DUPLEX, .5, (0, 0, 255), 1)
    try:
        # Convert CV2 image to ROS Image message
        ros_overlay_img_msg = bridge.cv2_to_imgmsg(cv2_overlay_img, 'bgr8')
    except CvBridgeError, e:
        print(e)
        pass

    overlay_img_pub.publish(ros_overlay_img_msg)


def od_loop():

    data = rospy.wait_for_message(
        '/camera/rgb/image_raw', Image, timeout=5.0)

    img_h, img_w = data.height, data.width
    rospy.logdebug('received frame hxw:%dx%d step:%d',
                   img_h, img_w, data.step)
    try:
        # Convert ROS Image message to CV2 image
        cv2_img = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError, e:
        print(e)
        pass

    # Convert the image from CV2 format to JPG which is expecited by Azure
    # predictor
    _, buf = cv2.imencode('.jpg', cv2_img)
    results = predictor.predict_image(project_id, buf)

    detected_objects_list = []
    for prediction in results.predictions:
        if prediction.probability < prediction_threshold:
            continue

        [box_left, box_top, box_w, box_h] = prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height
        do = DetectedObject()
        do.label = prediction.tag_name
        do.probability = prediction.probability
        # (x,y) are the center of the box relative to the Image, where the image
        # (0,0) is the top-left corner of the image
        # x,y,width and height are all normalized with respect to actual image
        # width and height.
        do.x_center = box_left + box_w / 2.0
        do.y_center = box_top + box_h / 2.0
        do.width = box_w
        do.height = box_h

        detected_objects_list.append(do)
        rospy.loginfo("{0}: {1:.2f}% ({2:.2f},{3:.2f}) ({4:.2f}x{5:.2f})".format(
            prediction.tag_name, prediction.probability * 100, do.x_center, do.y_center, box_h, box_w))

    boxes_pub.publish(detected_objects_list)
    send_box_overlays(cv2_img, detected_objects_list)


if __name__ == '__main__':
    try:
        predictor = prediction_endpoint.PredictionEndpoint(prediction_key)
        assert(predictor)
        bridge = CvBridge()
        rospy.init_node('object_detector', anonymous=True)
        boxes_pub = rospy.Publisher(
            "/detected_objects/boxes", DetectedObjectsArray, queue_size=1)
        overlay_img_pub = rospy.Publisher(
            "/detected_objects/overlay_img", Image, queue_size=1)
        rate = rospy.Rate(od_loop_freq_hz)

        while not rospy.is_shutdown():
            od_loop()
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
