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
img = cv2.imread('kavsak3_input.png')       #test edilecek fotoğrafın çekilmesi
#img2 = cv2.imread('merc.png')       #test edilecek fotoğrafın çekilmesi
y=300
x=70
h=500
w=370
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
print("\nKuzeydoğu yolu araç sayısı:",yol_bir)
#cv2.waitKey(0)

#2222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222

sayi = 0
img2 = cv2.imread('kavsak3_input.png')       #test edilecek fotoğrafın çekilmesi
#img2 = cv2.imread('merc.png')       #test edilecek fotoğrafın çekilmesi

y=300
x=430
h=500
w=1100
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
print("Güneydoğu yolu araç sayısı:",yol_iki)

#3333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
sayi = 0
img3 = cv2.imread('kavsak3_input.png')       #test edilecek fotoğrafın çekilmesi

y=400
x=330
h=1100
w=500
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
print("Güneybatı yolu araç sayısı:",yol_uc)

#------------------------------------------------------Calculation------------------------------------------------------

yol = [yol_bir, yol_iki, yol_uc]
yer = ['Kuzeydoğu', 'Güneydoğu', 'Güneybatı']
#print('1. yoldaki araba sayısı = ',yol[0])
max=0
for i in range(0, 3, 1):
    if (max < yol[i]):
        max = yol[i]
        index = i


print('\nEn yoğun yol', yer[index], 'yolu,\nAraç sayısı', max)
#print('Max = ', max(yol))

#---------------------------------------------------------Print---------------------------------------------------------

cv2.imshow(yer[0],img)
cv2.imshow(yer[1],img2)
cv2.imshow(yer[2],img3)

yon = index + 1

kavsak = cv2.imread("kavsak3.jpg")
#shapek = kavsak.shape
#print('resim çözünürlüğü',shapek)


if(yon == 1): #Kuzey
    cv2.circle(kavsak, (403, 237), 16, (0, 255, 0), -1) #K
    cv2.circle(kavsak, (475, 406), 16, (0, 0, 255), -1) #D
    cv2.circle(kavsak, (310, 474), 16, (0, 0, 255), -1) #G

if(yon == 2): #Doğu
    cv2.circle(kavsak, (403, 237), 16, (0, 0, 255), -1) #K
    cv2.circle(kavsak, (475, 406), 16, (0, 255, 0), -1) #D
    cv2.circle(kavsak, (310, 474), 16, (0, 0, 255), -1) #G

if(yon == 3): #Güney
    cv2.circle(kavsak, (403, 237), 16, (0, 0, 255), -1) #K
    cv2.circle(kavsak, (475, 406), 16, (0, 0, 255), -1) #D
    cv2.circle(kavsak, (310, 474), 16, (0, 255, 0), -1) #G



cv2.imshow("kavsak",kavsak)

#cv2.imwrite('Output/1.jpg', img1)

cv2.waitKey(0)