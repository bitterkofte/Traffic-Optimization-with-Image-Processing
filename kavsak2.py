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

#1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
sayi = 0
img = cv2.imread('kavsak1_input.png')       #test edilecek fotoğrafın çekilmesi
#img2 = cv2.imread('merc.png')       #test edilecek fotoğrafın çekilmesi

y=550
x=0
h=660
w=300
crop_image1 = img[x:w, y:h]
#cv2.imshow("Cropped", crop_image1)
img = crop_image1

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
print('K2 yolundaki araç sayısı:',yol_bir)
#cv2.waitKey(0)

#2222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222

sayi = 0
img2 = cv2.imread('kavsak1_input.png')       #test edilecek fotoğrafın çekilmesi
#img2 = cv2.imread('merc.png')       #test edilecek fotoğrafın çekilmesi

y=650
x=0
h=750
w=320
crop_image1 = img2[x:w, y:h]
#cv2.imshow("Cropped", crop_image1)
img2 = crop_image1

img2_width = img2.shape[1]            #genişlik ve yüksekliğin belirlenmesi
img2_height = img2.shape[0]

img2_blob = cv2.dnn.blobFromImage(img2,1/255,(416,416),swapRB=True)       #çekilen resmin standardizasyonu

model.setInput(img2_blob)
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
            bounding_box = object_detection[0:4] * np.array([img2_width,img2_height,img2_width,img2_height])
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

    cv2.rectangle(img2,(start_x,start_y),(end_x,end_y),(255,0,0),2)    #sınırlayıcı kutuların çizimi

    cv2.putText(img2,label,(start_x,start_y-10),cv2.FONT_HERSHEY_DUPLEX,0.7,(255,0,0),2,1)    #nesnenin isminin yazdırılması
    cv2.putText(img2, f"{confidence*100:.2f}%", (start_x, end_y+20), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,1)   #nesnenin güven skoru


yol_iki=sayi
print('K1 yolundaki araç sayısı:',yol_iki)

#3333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
sayi = 0
img3 = cv2.imread('kavsak1_input.png')       #test edilecek fotoğrafın çekilmesi

y=700
x=280
h=1200
w=340
crop_image1 = img3[x:w, y:h]
#cv2.imshow("Cropped", crop_image1)
img3 = crop_image1

img3_width = img3.shape[1]            #genişlik ve yüksekliğin belirlenmesi
img3_height = img3.shape[0]

img3_blob = cv2.dnn.blobFromImage(img3,1/255,(416,416),swapRB=True)       #çekilen resmin standardizasyonu

model.setInput(img3_blob)
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
            bounding_box = object_detection[0:4] * np.array([img3_width,img3_height,img3_width,img3_height])
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

    cv2.rectangle(img3,(start_x,start_y),(end_x,end_y),(255,0,0),2)    #sınırlayıcı kutuların çizimi

    cv2.putText(img3,label,(start_x,start_y-10),cv2.FONT_HERSHEY_DUPLEX,0.7,(255,0,0),2,1)    #nesnenin isminin yazdırılması
    cv2.putText(img3, f"{confidence*100:.2f}%", (start_x, end_y+20), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,1)   #nesnenin güven skoru


yol_uc=sayi
print('D2 yolundaki araç sayısı:',yol_uc)

#4444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444

sayi = 0
img4 = cv2.imread('kavsak1_input.png')       #test edilecek fotoğrafın çekilmesi

y=700
x=330
h=1250
w=400
crop_image1 = img4[x:w, y:h]
#cv2.imshow("Cropped", crop_image1)
img4 = crop_image1

img4_width = img4.shape[1]            #genişlik ve yüksekliğin belirlenmesi
img4_height = img4.shape[0]

img4_blob = cv2.dnn.blobFromImage(img4,1/255,(416,416),swapRB=True)       #çekilen resmin standardizasyonu

model.setInput(img4_blob)
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
            bounding_box = object_detection[0:4] * np.array([img4_width,img4_height,img4_width,img4_height])
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

    cv2.rectangle(img4,(start_x,start_y),(end_x,end_y),(255,0,0),2)    #sınırlayıcı kutuların çizimi

    cv2.putText(img4,label,(start_x,start_y-10),cv2.FONT_HERSHEY_DUPLEX,0.7,(255,0,0),2,1)    #nesnenin isminin yazdırılması
    cv2.putText(img4, f"{confidence*100:.2f}%", (start_x, end_y+20), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,1)   #nesnenin güven skoru


yol_dort=sayi
print('D1 yolundaki araç sayısı:',yol_dort)

#5555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555

sayi = 0
img5 = cv2.imread('kavsak1_input.png')       #test edilecek fotoğrafın çekilmesi

y=670
x=400
h=900
w=870
crop_image1 = img5[x:w, y:h]
#cv2.imshow("Cropped", crop_image1)
img5 = crop_image1

img5_width = img5.shape[1]            #genişlik ve yüksekliğin belirlenmesi
img5_height = img5.shape[0]

img5_blob = cv2.dnn.blobFromImage(img5,1/255,(416,416),swapRB=True)       #çekilen resmin standardizasyonu

model.setInput(img5_blob)
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
            bounding_box = object_detection[0:4] * np.array([img5_width,img5_height,img5_width,img5_height])
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

    cv2.rectangle(img5,(start_x,start_y),(end_x,end_y),(255,0,0),2)    #sınırlayıcı kutuların çizimi

    cv2.putText(img5,label,(start_x,start_y-10),cv2.FONT_HERSHEY_DUPLEX,0.7,(255,0,0),2,1)    #nesnenin isminin yazdırılması
    cv2.putText(img5, f"{confidence*100:.2f}%", (start_x, end_y+20), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,1)   #nesnenin güven skoru


yol_bes=sayi
print('G2 yolundaki araç sayısı:',yol_bes)

#6666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666

sayi = 0
img6 = cv2.imread('kavsak1_input.png')       #test edilecek fotoğrafın çekilmesi

y=630
x=400
h=700
w=870
crop_image1 = img6[x:w, y:h]
#cv2.imshow("Cropped", crop_image1)
img6 = crop_image1

img6_width = img6.shape[1]            #genişlik ve yüksekliğin belirlenmesi
img6_height = img6.shape[0]

img6_blob = cv2.dnn.blobFromImage(img6,1/255,(416,416),swapRB=True)       #çekilen resmin standardizasyonu

model.setInput(img6_blob)
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
            bounding_box = object_detection[0:4] * np.array([img6_width,img6_height,img6_width,img6_height])
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

    cv2.rectangle(img6,(start_x,start_y),(end_x,end_y),(255,0,0),2)    #sınırlayıcı kutuların çizimi

    cv2.putText(img6,label,(start_x,start_y-10),cv2.FONT_HERSHEY_DUPLEX,0.7,(255,0,0),2,1)    #nesnenin isminin yazdırılması
    cv2.putText(img6, f"{confidence*100:.2f}%", (start_x, end_y+20), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,1)   #nesnenin güven skoru


yol_alti=sayi
print('G1 yolundaki araç sayısı:',yol_alti)

#7777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777

sayi = 0
img7 = cv2.imread('kavsak1_input.png')       #test edilecek fotoğrafın çekilmesi

y=400
x=370
h=655
w=430
crop_image1 = img7[x:w, y:h]
#cv2.imshow("Cropped", crop_image1)
img7 = crop_image1

img7_width = img7.shape[1]            #genişlik ve yüksekliğin belirlenmesi
img7_height = img7.shape[0]

img7_blob = cv2.dnn.blobFromImage(img7,1/255,(416,416),swapRB=True)       #çekilen resmin standardizasyonu

model.setInput(img7_blob)
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
            bounding_box = object_detection[0:4] * np.array([img7_width,img7_height,img7_width,img7_height])
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

    cv2.rectangle(img7,(start_x,start_y),(end_x,end_y),(255,0,0),2)    #sınırlayıcı kutuların çizimi

    cv2.putText(img7,label,(start_x,start_y-10),cv2.FONT_HERSHEY_DUPLEX,0.7,(255,0,0),2,1)    #nesnenin isminin yazdırılması
    cv2.putText(img7, f"{confidence*100:.2f}%", (start_x, end_y+20), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,1)   #nesnenin güven skoru


yol_yedi=sayi
print('B2 yolundaki araç sayısı:',yol_yedi)

#8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888

sayi = 0
img8 = cv2.imread('kavsak1_input.png')       #test edilecek fotoğrafın çekilmesi

y=0
x=320
h=655
w=385
crop_image = img8[x:w, y:h]
#cv2.imshow("Cropped", crop_image)
img8 = crop_image

img8_width = img8.shape[1]            #genişlik ve yüksekliğin belirlenmesi
img8_height = img8.shape[0]

img8_blob = cv2.dnn.blobFromImage(img8,1/255,(416,416),swapRB=True)       #çekilen resmin standardizasyonu

model.setInput(img8_blob)
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
            bounding_box = object_detection[0:4] * np.array([img8_width,img8_height,img8_width,img8_height])
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

    cv2.rectangle(img8,(start_x,start_y),(end_x,end_y),(255,0,0),2)    #sınırlayıcı kutuların çizimi

    cv2.putText(img8,label,(start_x,start_y-10),cv2.FONT_HERSHEY_DUPLEX,0.7,(255,0,0),2,1)    #nesnenin isminin yazdırılması
    cv2.putText(img8, f"{confidence*100:.2f}%", (start_x, end_y+20), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,1)   #nesnenin güven skoru


yol_sekiz=sayi
print('B1 yolundaki araç sayısı:',yol_sekiz)

#------------------------------------------------------Calculation------------------------------------------------------

yol = [yol_bir, yol_iki, yol_uc, yol_dort, yol_bes, yol_alti, yol_yedi, yol_sekiz]
duble = [yol[4]+yol[0], yol[4]+yol[5], yol[0]+yol[1], yol[2]+yol[6], yol[2]+yol[3], yol[6]+yol[7], yol[3]+yol[7], yol[1]+yol[5]]
yer = ['K2-G2', 'G2-G1', 'K2-K1', 'D2-B2', 'D2-D1', 'B2-B1', 'B1-D1', 'G1-K1']

print(duble)

max=0
for i in range(0, 8, 1):
    if (max < duble[i]):
        max = duble[i]
        index = i

print('En yoğun yol grubu', yer[index], '\nAraç sayısı', max)

#---------------------------------------------------------Print---------------------------------------------------------

#cv2.imshow("Yol 1",img)
#cv2.imshow("Yol 2",img2)
#cv2.imshow("Yol 3",img3)
#cv2.imshow("Yol 4",img4)

yon = index + 1

kavsak = cv2.imread("kavsak2.jpg")
#shapek = kavsak.shape
#print('resim çözünürlüğü',shapek)

if(yon == 1): #Kuzey2 + Kuzey1
    cv2.circle(kavsak, (294, 244), 10, (0, 255, 0), -1) #K2
    cv2.circle(kavsak, (359, 244), 10, (0, 255, 0), -1) #K1
    cv2.circle(kavsak, (473, 293), 10, (0, 0, 255), -1) #D2
    cv2.circle(kavsak, (473, 360), 10, (0, 0, 255), -1) #D1
    cv2.circle(kavsak, (362, 473), 10, (0, 0, 255), -1) #G2
    cv2.circle(kavsak, (425, 473), 10, (0, 0, 255), -1) #G1
    cv2.circle(kavsak, (245, 427), 10, (0, 0, 255), -1) #B2
    cv2.circle(kavsak, (245, 362), 10, (0, 0, 255), -1) #B1

if(yon == 2): #Güney2 + Güney1
    cv2.circle(kavsak, (294, 244), 10, (0, 0, 255), -1) #K2
    cv2.circle(kavsak, (359, 244), 10, (0, 0, 255), -1) #K1
    cv2.circle(kavsak, (473, 293), 10, (0, 0, 255), -1) #D2
    cv2.circle(kavsak, (473, 360), 10, (0, 0, 255), -1) #D1
    cv2.circle(kavsak, (362, 473), 10, (0, 255, 0), -1) #G2
    cv2.circle(kavsak, (425, 473), 10, (0, 255, 0), -1) #G1
    cv2.circle(kavsak, (245, 427), 10, (0, 0, 255), -1) #B2
    cv2.circle(kavsak, (245, 362), 10, (0, 0, 255), -1) #B1

if(yon == 3): #Kuzey2 +Güney2
    cv2.circle(kavsak, (294, 244), 10, (0, 255, 0), -1) #K2
    cv2.circle(kavsak, (359, 244), 10, (0, 0, 255), -1) #K1
    cv2.circle(kavsak, (473, 293), 10, (0, 0, 255), -1) #D2
    cv2.circle(kavsak, (473, 360), 10, (0, 0, 255), -1) #D1
    cv2.circle(kavsak, (362, 473), 10, (0, 255, 0), -1) #G2
    cv2.circle(kavsak, (425, 473), 10, (0, 0, 255), -1) #G1
    cv2.circle(kavsak, (245, 427), 10, (0, 0, 255), -1) #B2
    cv2.circle(kavsak, (245, 362), 10, (0, 0, 255), -1) #B1

if (yon == 4): #Batı2 + Batı1
    cv2.circle(kavsak, (294, 244), 10, (0, 0, 255), -1) #K2
    cv2.circle(kavsak, (359, 244), 10, (0, 0, 255), -1) #K1
    cv2.circle(kavsak, (473, 293), 10, (0, 0, 255), -1) #D2
    cv2.circle(kavsak, (473, 360), 10, (0, 0, 255), -1) #D1
    cv2.circle(kavsak, (362, 473), 10, (0, 0, 255), -1) #G2
    cv2.circle(kavsak, (425, 473), 10, (0, 0, 255), -1) #G1
    cv2.circle(kavsak, (245, 427), 10, (0, 255, 0), -1) #B2
    cv2.circle(kavsak, (245, 362), 10, (0, 255, 0), -1) #B1

if (yon == 5): #Doğu2 + Doğu1
    cv2.circle(kavsak, (294, 244), 10, (0, 0, 255), -1) #K2
    cv2.circle(kavsak, (359, 244), 10, (0, 0, 255), -1) #K1
    cv2.circle(kavsak, (473, 293), 10, (0, 255, 0), -1) #D2
    cv2.circle(kavsak, (473, 360), 10, (0, 255, 0), -1) #D1
    cv2.circle(kavsak, (362, 473), 10, (0, 0, 255), -1) #G2
    cv2.circle(kavsak, (425, 473), 10, (0, 0, 255), -1) #G1
    cv2.circle(kavsak, (245, 427), 10, (0, 0, 255), -1) #B2
    cv2.circle(kavsak, (245, 362), 10, (0, 0, 255), -1) #B1

if (yon == 6): #Batı2 + Doğu2
    cv2.circle(kavsak, (294, 244), 10, (0, 0, 255), -1) #K2
    cv2.circle(kavsak, (359, 244), 10, (0, 0, 255), -1) #K1
    cv2.circle(kavsak, (473, 293), 10, (0, 255, 0), -1) #D2
    cv2.circle(kavsak, (473, 360), 10, (0, 0, 255), -1) #D1
    cv2.circle(kavsak, (362, 473), 10, (0, 0, 255), -1) #G2
    cv2.circle(kavsak, (425, 473), 10, (0, 0, 255), -1) #G1
    cv2.circle(kavsak, (245, 427), 10, (0, 255, 0), -1) #B2
    cv2.circle(kavsak, (245, 362), 10, (0, 0, 255), -1) #B1

if (yon == 7): #Batı1 + Doğu1
    cv2.circle(kavsak, (294, 244), 10, (0, 0, 255), -1) #K2
    cv2.circle(kavsak, (359, 244), 10, (0, 0, 255), -1) #K1
    cv2.circle(kavsak, (473, 293), 10, (0, 0, 255), -1) #D2
    cv2.circle(kavsak, (473, 360), 10, (0, 255, 0), -1) #D1
    cv2.circle(kavsak, (362, 473), 10, (0, 0, 255), -1) #G2
    cv2.circle(kavsak, (425, 473), 10, (0, 0, 255), -1) #G1
    cv2.circle(kavsak, (245, 427), 10, (0, 0, 255), -1) #B2
    cv2.circle(kavsak, (245, 362), 10, (0, 255, 0), -1) #B1

if (yon == 8): #Güney1 + Kuzey1
    cv2.circle(kavsak, (294, 244), 10, (0, 0, 255), -1) #K2
    cv2.circle(kavsak, (359, 244), 10, (0, 255, 0), -1) #K1
    cv2.circle(kavsak, (473, 293), 10, (0, 0, 255), -1) #D2
    cv2.circle(kavsak, (473, 360), 10, (0, 0, 255), -1) #D1
    cv2.circle(kavsak, (362, 473), 10, (0, 0, 255), -1) #G2
    cv2.circle(kavsak, (425, 473), 10, (0, 255, 0), -1) #G1
    cv2.circle(kavsak, (245, 427), 10, (0, 0, 255), -1) #B2
    cv2.circle(kavsak, (245, 362), 10, (0, 0, 255), -1) #B1

cv2.imshow("kavsak",kavsak)

cv2.imwrite('Output/cikti.jpg', kavsak)

cv2.waitKey(0)