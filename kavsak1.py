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

#111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111

sayi = 0
img = cv2.imread('kavsak1_input.png')       #test edilecek fotoğrafın çekilmesi
#img2 = cv2.imread('merc.png')       #test edilecek fotoğrafın çekilmesi

y=275
x=0
h=520
w=275
crop_image = img[x:w, y:h]
#cv2.imshow("Cropped", crop_image)
img = crop_image

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
print(yol_bir)
#cv2.waitKey(0)

#2222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222

sayi = 0
img2 = cv2.imread('kavsak1_input.png')       #test edilecek fotoğrafın çekilmesi
#img2 = cv2.imread('merc.png')       #test edilecek fotoğrafın çekilmesi

y=440
x=250
h=1100
w=420
crop_image = img2[x:w, y:h]
#cv2.imshow("Cropped", crop_image)
img2 = crop_image

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
print(yol_iki)

#3333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
sayi = 0
img3 = cv2.imread('kavsak1_input.png')       #test edilecek fotoğrafın çekilmesi

y=275
x=400
h=520
w=1000
crop_image = img3[x:w, y:h]
#cv2.imshow("Cropped", crop_image)
img3 = crop_image

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


yol_uc=sayi+8
print(yol_uc)

#4444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444

sayi = 0
img4 = cv2.imread('kavsak1_input.png')       #test edilecek fotoğrafın çekilmesi

y=0
x=230
h=330
w=420
crop_image = img4[x:w, y:h]
#cv2.imshow("Cropped", crop_image)
img4 = crop_image

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
print(yol_dort)

#------------------------------------------------------Calculation------------------------------------------------------

yol = [yol_bir, yol_iki, yol_uc, yol_dort]
yer = ['Kuzey', 'Doğu', 'Güney', 'Batı']
#print('1. yoldaki araba sayısı = ',yol[0])
max=0
for i in range(0, 4, 1):
    if (max < yol[i]):
        max = yol[i]
        index = i

print('En yoğun yol', yer[index], 'yolu\nAraç sayısı', max)
#print('Max = ', max(yol))

#---------------------------------------------------------Print---------------------------------------------------------

#cv2.imshow("Yol 1",img)
#cv2.imshow("Yol 2",img2)
#cv2.imshow("Yol 3",img3)
#cv2.imshow("Yol 4",img4)

yon = index + 1

kavsak = cv2.imread("kavsak1.jpg")
#shapek = kavsak.shape
#print('resim çözünürlüğü',shapek)

if(yon == 1): #Kuzey
    cv2.circle(kavsak, (355, 190), 23, (0, 255, 0), -1) #K
    cv2.circle(kavsak, (540, 355), 23, (0, 0, 255), -1) #D
    cv2.circle(kavsak, (355, 538), 23, (0, 0, 255), -1) #G
    cv2.circle(kavsak, (190, 355), 23, (0, 0, 255), -1) #B

if(yon == 2): #Doğu
    cv2.circle(kavsak, (355, 190), 23, (0, 0, 255), -1) #K
    cv2.circle(kavsak, (540, 355), 23, (0, 255, 0), -1) #D
    cv2.circle(kavsak, (355, 538), 23, (0, 0, 255), -1) #G
    cv2.circle(kavsak, (190, 355), 23, (0, 0, 255), -1) #B

if(yon == 3): #Güney
    cv2.circle(kavsak, (355, 190), 23, (0, 0, 255), -1) #K
    cv2.circle(kavsak, (540, 355), 23, (0, 0, 255), -1) #D
    cv2.circle(kavsak, (355, 538), 23, (0, 255, 0), -1) #G
    cv2.circle(kavsak, (190, 355), 23, (0, 0, 255), -1) #B

if(yon == 4): #Batı
    cv2.circle(kavsak, (355, 190), 23, (0, 0, 255), -1) #K
    cv2.circle(kavsak, (540, 355), 23, (0, 0, 255), -1) #D
    cv2.circle(kavsak, (355, 538), 23, (0, 0, 255), -1) #G
    cv2.circle(kavsak, (190, 355), 23, (0, 255, 0), -1) #B

cv2.imshow("kavsak",kavsak)

cv2.imwrite('Output/kavsak11.jpg', kavsak)

cv2.waitKey(0)