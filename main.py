from __future__ import print_function
import numpy as np
import face_recognition
import argparse
import cv2
import os
import pickle
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

parser = argparse.ArgumentParser()
parser.add_argument('--with_draw', help='do draw?', default='True')
args = parser.parse_args()

net = cv2.dnn.readNetFromCaffe('./models/deploy.prototxt.txt', './models/res10_300x300_ssd_iter_140000.caffemodel')

knn_clf = pickle.load(open('./models/fr_knn.pkl', 'rb'))

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

# vc = cv2.VideoCapture('./data/TAEYANG_ONLY_LOOK_AT_ME_MV.mp4')
vc = cv2.VideoCapture('./data/gd_and_ty.mp4')

length = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
print ('length :', length)

if args.with_draw == 'True':
    cv2.namedWindow('show', 0)

for idx in range(length):
    img_bgr = vc.read()[1]
    if img_bgr is None:
        break
    # if idx%3 != 0: continue
    # if idx < 200: continue
    
    start = cv2.getTickCount()
    
    ### preprocess
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_bgr_ori = img_bgr.copy()
    img_bgr = preprocess(img_bgr)

    ### detection
    border = (img_bgr.shape[1] - img_bgr.shape[0])//2
    img_bgr = cv2.copyMakeBorder(img_bgr,
                                 border, # top
                                 border, # bottom
                                 0, # left
                                 0, # right
                                 cv2.BORDER_CONSTANT,
                                 value=(0,0,0))

    (h, w) = img_bgr.shape[:2]
    
    blob = cv2.dnn.blobFromImage(cv2.resize(img_bgr, (300, 300)), 1.0,
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
        t = int(t + (original_vertical_length)*0.15) - border
        b = int(b - (original_vertical_length)*0.05) - border

        margin = ((b-t) - (r-l))//2
        l = l - margin if (b-t-r+l)%2 == 0 else l - margin - 1
        r = r + margin
        refined_box = [t,r,b,l]
        list_bboxes.append(refined_box)
        list_confidence.append(confidence)

    ### facenet
    face_encodings = face_recognition.face_encodings(img_rgb, list_bboxes)
    closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
    is_recognized = [closest_distances[0][i][0] <= 0.4 for i in range(len(list_bboxes))]
    list_reconized_face = [(pred, loc, conf) if rec else ("unknown", loc, conf) for pred, loc, rec, conf in zip(knn_clf.predict(face_encodings), list_bboxes, is_recognized, list_confidence)]
    # print (list_reconized_face)

    time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
    print ('%d, elapsed time: %.3fms'%(idx,time))

    ### blurring
    img_bgr_blur = img_bgr_ori.copy()
    for name, bbox, conf in list_reconized_face:
        t,r,b,l = bbox
        if name == 'unknown':
            face = img_bgr_blur[t:b, l:r]
            small = cv2.resize(face, None, fx=.05, fy=.05, interpolation=cv2.INTER_NEAREST)
            blurred_face = cv2.resize(small, (face.shape[:2]), interpolation=cv2.INTER_NEAREST)
            img_bgr_blur[t:b, l:r] = blurred_face

    ### draw rectangle bbox
    if args.with_draw == 'True':
        source_img = Image.fromarray(img_bgr_ori)
        draw = ImageDraw.Draw(source_img)
        for name, bbox, confidence in list_reconized_face:
            t,r,b,l = bbox
            # print (int((r-l)/img_bgr_ori.shape[1]*100))
            font_size = int((r-l)/img_bgr_ori.shape[1]*100)

            draw.rectangle(((l,t),(r,b)), outline=(0,255,128))

            draw.rectangle(((l,t-font_size-2),(r,t+2)), fill=(0,255,128))
            draw.text((l, t - font_size), name, font=ImageFont.truetype('./BMDOHYEON_TTF.TTF', font_size), fill=(0,0,0,0))

        show = np.asarray(source_img)
        cv2.imshow('show', show)
        cv2.imshow('blur', img_bgr_blur)
        key = cv2.waitKey(30)
        if key == 27:
            break


### opencv text, box drawing
# cv2.rectangle(img_bgr_blur, (l, t), (r, b), (0, 255, 0), 2)

# cv2.rectangle(img_bgr_ori, (l, t), (r, b), (0, 255, 128), 2)
# text = "%s: %.2f" % (name,confidence)
# text_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
# y = t #- 1 if t - 1 > 1 else t + 1
# cv2.rectangle(img_bgr_ori, 
#             (l,y-text_size[1]),(l+text_size[0], y+base_line), (0,255,0), -1)
# cv2.putText(img_bgr_ori, text, (l, y), 
#             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)