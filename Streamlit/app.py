import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
import pandas as pd
import predict_function
from const import CLASSES, COLORS
from settings import DEFAULT_CONFIDENCE_THRESHOLD, DEMO_IMAGE, MODEL, PROTOTXT
import copy
import matplotlib.pyplot as plt
import time

html = """
  <style>
    .reportview-container {
      flex-direction: row-reverse;
    }

    header > .toolbar {
      flex-direction: row-reverse;
      left: 1rem;
      right: auto;
    }

    .sidebar .sidebar-collapse-control,
    .sidebar.--collapsed .sidebar-collapse-control {
      left: auto;
      right: 0.5rem;
    }

    .sidebar .sidebar-content {
      transition: margin-right .3s, box-shadow .3s;
      width: 805px;
    }

    .sidebar.--collapsed .sidebar-content {
      margin-left: auto;
      margin-right: -21rem;
    }

    @media (max-width: 991.98px) {
      .sidebar .sidebar-content {
        margin-left: auto;
      }
    }
  </style>
"""
st.markdown(html, unsafe_allow_html=True)


@st.cache
def process_image(image):
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    net.setInput(blob)
    detections = net.forward()
    return detections


@st.cache
def annotate_image(
    image, detections, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
):
    # loop over the detections
    (h, w) = image.shape[:2]
    labels = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = f"{CLASSES[idx]}: {round(confidence * 100, 2)}%"
            labels.append(label)
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(
                image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2
            )
    return image, labels


st.title("狗狗種類辨識")
st.write("""
 這個app可以辨識120種狗狗種類! \n
 精確度為**65%**\n
 Top-5 精確度為**92%**
""")


sample_image = os.listdir("SampleImages")
sample_image_dir={}
for i in sample_image:
    name=i.split('.')[0]
    name=name.capitalize()
    sample_image_dir[name]="SampleImages/"+i
    
all_breed=pd.read_csv('all_class.csv')
all_breed=all_breed['0'].values
all_breed_for_predict= copy.deepcopy(all_breed)
all_breed.sort()



#st.write("上傳照片:")
# start predict
img_file_buffer = st.file_uploader("", type=["png", "jpg", "jpeg"])
if img_file_buffer is not None:
    image_orignal=Image.open(img_file_buffer)
    image = np.array(image_orignal)

else:
    demo_image = DEMO_IMAGE
    image_orignal=Image.open(demo_image)
    image = np.array(image_orignal)

model=predict_function.openModel()
#from torchsummary import summary
#print(summary(model.cuda(), (3, 299, 299)))

res=predict_function.returnTopN_Predict(image_orignal,model,all_breed_for_predict,n=10)
dataframe=pd.DataFrame(res)
dataframe.columns=['Breed','Probability']
#detections = process_image(image)
#image, labels = annotate_image(image, detections, 0.5)

sample_image2 = np.array(Image.open(sample_image_dir[res[0][0]]))

#st.dataframe(res)

st.sidebar.header('預測結果:')
st.sidebar.write(res[0][0],' , 信心分數:',res[0][1])
st.image(
    image, caption=f"Upload image", use_column_width=True,
)

st.sidebar.image(
    sample_image2, caption=res[0][0], use_column_width=True,
)

st.sidebar.header('120種狗狗種類如下:')
option = st.sidebar.selectbox('',all_breed)
sample_image = np.array(Image.open(sample_image_dir[option]))
st.sidebar.image(
    sample_image, caption=option, use_column_width=True,
)

st.write("## 其他預測結果:")

labels = dataframe['Breed']
sizes = dataframe['Probability']
# print(sizes) # adds up to 1433, which is the total number of participants
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False)
ax1.axis('equal')
st.pyplot(fig1)


