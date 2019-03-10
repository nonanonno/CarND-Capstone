from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np


class TLClassifier(object):
    def __init__(self):
        # TODO load classifier
        GRAPH = '/home/shuh/workspace/udacity/CarND-Capstone/ros/interface_graph/frozen_inference_graph.pb'
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
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # TODO implement light color prediction
        output_dict = self.run_interface_for_single_image(image)
        result = TrafficLight.UNKNOWN
        if output_dict['detection_scores'][0] > 0.5:
            if output_dict['detection_classes'][0] == 1:
                result = TrafficLight.GREEN
            elif output_dict['detection_classes'][0] == 2:
                result = TrafficLight.RED
            elif output_dict['detection_classes'][0] == 3:
                result = TrafficLight.YELLOW
        print(result)
        return result
