import numpy as np
import cv2
from django.shortcuts import render, redirect
from django.http.response import StreamingHttpResponse, HttpResponse
from django.template.response import TemplateResponse
from django.views.decorators.http import condition
import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import pytesseract
from utils import label_map_util
from utils import visualization_utils as vis_util
import pyttsx3


obj = ['Background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'Null', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'Null', 'backpack', 'umbrella', 'Null', 'Null', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'Null', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'Null', 'dining table', 'Null', 'Null', 'toilet', 'Null', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'Null', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def codeRun():

    arch = 'resnet18'
    engine =pyttsx3.init()
    model_file = 'whole_%s_places365_python36.pth.tar' % arch
    if not os.access(model_file, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget' + weight_url)



    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract'

    MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
    # MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

    NUM_CLASSES = 90


    if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
        print ('Downloading the model')
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())
        print ('Download complete')
    else:
        print ('Model already exists')

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    #Convert the coco model into label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    cap = cv2.VideoCapture(0)
    

    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            retr = True
            while (retr):
                engine.setProperty('rate',300)
                retr,image_np = cap.read()

                text=pytesseract.image_to_string(image_np)
                te = "".join(text.split())
                if len(te):
                    print("start",text,"end",len(text),len(te))
                    ret, jpeg = cv2.imencode('.jpg', image_np)    
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')      
                    # cv2.imshow('image',cv2.resize(image_np,(1024,768)))
                    print(text)
                    engine.say(text)
                    engine.runAndWait()
                else:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=5)
                    engine.setProperty('rate',300)
                    for i,b in enumerate(boxes[0]):
                        if scores[0][i] >= 0.7:
                            mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                            mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                            apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                            cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                            print(apx_distance)
                            if apx_distance <=0.2:
                                if mid_x > 0.3 and mid_x < 0.7:
                                    cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                                    
                                    txt = "Warning " + obj[int(classes[0][i])] + " very close"
                                    print(classes[0][i],txt)
                                    ret, jpeg = cv2.imencode('.jpg', image_np)    
                                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')      
                                    engine.say(txt)

                                else:
                                    print(classes[0][i])
                                    txt = obj[int(classes[0][i])] + " at safe distance"
                                    ret, jpeg = cv2.imencode('.jpg', image_np)    
                                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')      
                                    print(txt)
                                    engine.say(txt)
                            else:
                                print(classes[0][i])
                                txt = obj[int(classes[0][i])] + " at safe distance"
                                print(classes[0][i],txt)
                                ret, jpeg = cv2.imencode('.jpg', image_np)    
                                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')      
                                engine.say(txt)
                            engine.runAndWait()
                        else:
                            ret, jpeg = cv2.imencode('.jpg', image_np)    
                            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')      
                    
                    ret, jpeg = cv2.imencode('.jpg', image_np)    
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')      
                        
               
def homePage(request):
    return render(request, 'index.html')

def smartVision(request):
    return render(request, 'module.html')


    
def stream(request):
    return StreamingHttpResponse(codeRun(),content_type='multipart/x-mixed-replace; boundary=frame')