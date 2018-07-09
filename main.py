from __future__ import print_function
import numpy as np
import cv2
import argparse
import face_recognition

parser = argparse.ArgumentParser()
parser.add_argument('--with_draw', help='do draw?', default='True')
args = parser.parse_args()

net = cv2.dnn.readNetFromCaffe('./models/deploy.prototxt.txt', './models/res10_300x300_ssd_iter_140000.caffemodel')

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def preprocess(img):
    ### analysis
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(1):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray_img.mean() < 130:
            img = adjust_gamma(img, 1.5)
        else:
            break
    return img

vc = cv2.VideoCapture('./data/TAEYANG_ONLY_LOOK_AT_ME_MV.mp4')

length = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
print ('length :', length)

if args.with_draw == 'True':
    cv2.namedWindow('show', 0)

for idx in range(length):
    bgr_img = vc.read()[1]
    if bgr_img is None:
        break
    if idx%3 != 0: continue
    if idx < 200: continue
    
    start = cv2.getTickCount()
    
    ### preprocess
    bgr_img = preprocess(bgr_img)

    ### detection
    border = (bgr_img.shape[1] - bgr_img.shape[0])//2
    bgr_img = cv2.copyMakeBorder(bgr_img,
                                 border, # top
                                 border, # bottom
                                 0, # left
                                 0, # right
                                 cv2.BORDER_CONSTANT,
                                 value=(0,0,0))

    (h, w) = bgr_img.shape[:2]
    
    blob = cv2.dnn.blobFromImage(cv2.resize(bgr_img, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    ### bbox
    list_bboxes = []
    list_confidence = []
    # list_dlib_rect = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.6:
                continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (l, t, r, b) = box.astype("int") # l t r b
        
        original_vertical_length = b-t
        t = int(t + (original_vertical_length)*0.15)
        b = int(b - (original_vertical_length)*0.05)

        margin = ((b-t) - (r-l))//2
        l = l - margin if (b-t-r+l)%2 == 0 else l - margin - 1
        r = r + margin
        refined_box = [l, t, r, b]
        list_bboxes.append(refined_box)
        list_confidence.append(confidence)

    time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
    print ('%d, elapsed time: %.3fms'%(idx,time))

    ### draw rectangle bbox
    if args.with_draw == 'True':
        for bbox, confidence in zip(list_bboxes, list_confidence):
            l, t, r, b = bbox
            
            cv2.rectangle(bgr_img, (l, t), (r, b),
                (0, 255, 0), 2)
            text = "face: %.2f" % confidence
            text_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y = t #- 1 if t - 1 > 1 else t + 1
            cv2.rectangle(bgr_img, (l,y-text_size[1]),(l+text_size[0], y+base_line), (0,255,0), -1)
            cv2.putText(bgr_img, text, (l, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow('show', bgr_img)
        key = cv2.waitKey(1)
        if key == 27:
            break