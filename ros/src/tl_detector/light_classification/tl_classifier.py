from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import cv2

GREEN = 0
RED = 1
YELLOW = 2
UNKNOWN = 3


def LightToColor(light):
    if light == GREEN:
        return (0, 255, 0)
    elif light == YELLOW:
        return (0, 255, 255)
    elif light == RED:
        return (0, 0, 255)
    else:
        return (0, 0, 0)


def draw(img, box, color):
    (height, width, channels) = img.shape
    ymin, xmin, ymax, xmax = box
    left_top = (int(xmin * width), int(ymin * height))
    right_bottom = (int(xmax * width), int(ymax * height))

    cv2.rectangle(img, left_top, right_bottom, LightToColor(color), 3)


class TLClassifier(object):
    def __init__(self):
        # TODO load classifier
        GRAPH = 'inference_graph/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.session = tf.Session()
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            self.tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)

            self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        self.index = 0
        print(tf.__version__)

    def run_interface_for_single_image(self, image):
        output_dict = self.session.run(self.tensor_dict,
                                       feed_dict={self.image_tensor: np.expand_dims(image, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(
            output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        return output_dict

    def predict_color(self, image, box):
        (height, width, channels) = image.shape
        ymin, xmin, ymax, xmax = box
        ymin *= height
        ymax *= height
        xmin *= width
        xmax *= width
        box_height = ymax - ymin
        box_width = xmax - xmin
        box_height_tri = box_height // 3
        is_color = [0, 0, 0]
        for i in range(3):
            offset_y = ymin + box_height_tri * i
            for y in range(5):
                for x in range(5):
                    xx = int(xmin + box_width // 2 - 2 + x)
                    yy = int(offset_y + box_height_tri // 2 - 2 + y)
                    if xx < 0:
                        xx = 0
                    if xx >= width:
                        xx = width - 1
                    if yy < 0:
                        yy = 0
                    if yy >= height:
                        yy = height - 1

                    if i == 0:  # red
                        if image[yy, xx, 0] < 127 and image[yy, xx, 1] < 127 and image[yy, xx, 2] > 127:
                            is_color[RED] += 1
                    elif i == 1:  # yello
                        if image[yy, xx, 0] < 127 and image[yy, xx, 1] > 127 and image[yy, xx, 2] > 127:
                            is_color[YELLOW] += 1
                    else:   # green
                        if image[yy, xx, 0] < 127 and image[yy, xx, 1] > 127 and image[yy, xx, 2] < 127:
                            is_color[GREEN] += 1
        winner = 0
        for i in range(3):
            if is_color[winner] < is_color[i]:
                winner = i
        count = 0
        for i in range(3):
            if is_color[winner] == is_color[i]:
                count += 1

        if count >= 2:
            return UNKNOWN
        else:
            return winner

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # TODO implement light color prediction
        #cv2.imwrite("/mnt/datasets/images/" + str(self.index).zfill(8) + ".png", image)

        output_dict = self.run_interface_for_single_image(image)
        result = TrafficLight.UNKNOWN
        vote = [0, 0, 0, 0]

        for i in range(3):
            box = output_dict['detection_boxes'][i]
            score = output_dict['detection_scores'][i]
            if score > 0.3:
                color = self.predict_color(image, box)
                vote[color] += 1

               # draw(image, box, color)
                print("{0} {1} {2}".format(box, score, color))

        winner = 0
        for i in range(4):
            if vote[winner] < vote[i]:
                winner = i
        result_dict = [TrafficLight.GREEN, TrafficLight.RED,
                       TrafficLight.YELLOW, TrafficLight.UNKNOWN]

        result = result_dict[winner]

        # print(result)
#        image = cv2.resize(image, None, fx=0.5, fy=0.5)
#        cv2.imshow("a", image)

#        cv2.waitKey(1)
        return result
