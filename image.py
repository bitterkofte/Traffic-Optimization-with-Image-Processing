import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob, os

model = cv2.dnn.readNetFromDarknet("sam.cfg","sam.weights") #moodelin ağırlığının ve cfg dosyasının çekilmesi
layers = model.getLayerNames()      #ağırlıktaki katmanların çıkarılması
unconnect = model.getUnconnectedOutLayers() #yolo katmanlarının seçimi
unconnect = unconnect-1


output_layers = []
for i in unconnect:
    output_layers.append(layers[int(i)])

classFile = 'obj.names'
classNames=[]
with open(classFile,'rt') as f:     #sınıf isimlerinin names dosyasından okunması
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)

sayi = 0
img = cv2.imread('sinan.jpg')       #test edilecek fotoğrafın çekilmesi
#img2 = cv2.imread('merc.png')       #test edilecek fotoğrafın çekilmesi

img_width = img.shape[1]            #genişlik ve yüksekliğin belirlenmesi
img_height = img.shape[0]

img_blob = cv2.dnn.blobFromImage(img,1/255,(416,416),swapRB=True)       #çekilen resmin standardizasyonu

model.setInput(img_blob)
detection_layers = model.forward(output_layers)     #nesne tespiti katmanı dizisinin oluşturulması

ids_list = []
boxes_list = []
confidences_list = []

for detection_layer in detection_layers:
    for object_detection in detection_layer:
        scores = object_detection[5:]
        predicted_id = np.argmax(scores)
        confidence =scores[predicted_id]


        if confidence > 0.10:   #güven skoruna göre sınırlayıcı kutuların belirlenmesi

            label = classNames[predicted_id]
            bounding_box = object_detection[0:4] * np.array([img_width,img_height,img_width,img_height])
            (box_center_x, box_center_y ,box_width ,box_height) = bounding_box.astype("int")
            start_x = int(box_center_x- (box_width/2))
            start_y = int(box_center_y - (box_height/2))

            ids_list.append(predicted_id)
            confidences_list.append(float(confidence))
            boxes_list.append([start_x,start_y,int(box_width),int(box_height)])

max_ids = cv2.dnn.NMSBoxes(boxes_list,confidences_list,0.5,0.4)     #çakışan kutuların giderilmesi

for max_id in max_ids:
    max_class_id=max_id
    box = boxes_list[max_class_id]

    start_x = box[0]
    start_y = box[1]
    box_width = box[2]
    box_height= box[3]

    predicted_id = ids_list[max_class_id]
    label = classNames[predicted_id]
    #print(classNames[predicted_id])
    sayi = sayi + 1
    confidence=confidences_list[max_class_id]

    end_x = start_x + box_width
    end_y = start_y+box_height

    cv2.rectangle(img,(start_x,start_y),(end_x,end_y),(255,0,0),2)    #sınırlayıcı kutuların çizimi

    cv2.putText(img,label,(start_x,start_y-10),cv2.FONT_HERSHEY_DUPLEX,0.7,(255,0,0),2,1)    #nesnenin isminin yazdırılması
    cv2.putText(img, f"{confidence*100:.2f}%", (start_x, end_y+20), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,1)   #nesnenin güven skoru

yol_bir=sayi
print('Yoldaki araç sayısı', yol_bir)
#cv2.waitKey(0)

#---------------------------------------------------------Print---------------------------------------------------------
cv2.imshow("Yol 1",img)
cv2.imwrite('Output/oto.jpg', img)

cv2.waitKey(0)