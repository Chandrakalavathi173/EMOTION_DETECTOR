#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install opencv-python


# In[4]:


import cv2


# In[6]:


get_ipython().system('pip install deepface')


# In[7]:


from deepface import DeepFace


# In[8]:


img=cv2.imread("img.jpg")


# In[9]:


import matplotlib.pyplot as plt


# In[10]:


plt.imshow(img)


# In[11]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[12]:


predictions=DeepFace.analyze(img)


# In[13]:


predictions


# In[15]:


d=predictions[0]
d1=d["dominant_emotion"]
d1


# In[16]:


box=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
faces=box.detectMultiScale(gray,1.1,4)
for (x,y,w,h) in faces:
 cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[17]:


text=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,
             d1,
             (0,50),
             text, 1,
             (0,0,0),
             2,
             cv2.LINE_4);


# In[18]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[22]:


import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
box = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(1)
if not cap.isOpened():
 cap = cv2.VideoCapture(0)
if not cap.isOpened():
 raise IOError("Cannot open webcam")
while True:
 ret, frame = cap.read()
 res = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 faces = box.detectMultiScale(gray, 1.1, 4)
 d = res[0]
 d1 = d['dominant_emotion']
 for (x, y, w, h) in faces:
     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
     text = cv2.FONT_HERSHEY_SIMPLEX
     cv2.putText(frame, d1, (0, 50), text, 1, (0, 0, 0), 2, cv2.LINE_4)
     cv2.imshow('original video', frame)
 if cv2.waitKey(1) & 0xFF == ord("q"):
     break
cap.release()
cv2.destroyAllWindows()


# In[ ]:




