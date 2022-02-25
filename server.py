import os
from PIL import Image
import numpy as np
import cv2
import pickle
import pandas as pd

base_dir = os.getcwd()


from flask import Flask, render_template
app = Flask(__name__)

aidi = pd.read_csv(base_dir+"/"+"AIDI1003.csv")
student=aidi.set_index("Names",drop=False)
studentsList=len(aidi)
studentNames=[]
for initialise in range(studentsList):
    studentNames.append(aidi["Names"].iloc[initialise])
    student.loc[studentNames[initialise],"Status"]="Absent"

@app.route('/')
def index():
  return render_template('index.html')

import webbrowser
webbrowser.open('file.html')

@app.route('/my-link/')
def my_link():
  base_dir = os.getcwd()
  image_dir = os.path.join(base_dir, "images")

  x_train = []
  y_labels = []
  current_id = 0
  label_ids = {}



  facecade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
  recognizer = cv2.face.LBPHFaceRecognizer_create()


  for root, dirs, files in os.walk(image_dir):
      for file in files:
          if file.endswith("png") or file.endswith("jpeg"):
              path = os.path.join(root, file)
              label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
             
              if not label in label_ids:
                  label_ids[label] = current_id
                  current_id += 1
              id_ = label_ids[label]
              #x_train.append(path)
              #y_labels.append(label)
              image = Image.open(path).convert("L")   #this opens an image and while converting it into the grayscale
              imagearray = np.array(image, "uint8")   #creating an array using numpy with the images by their numeric values using uint8 type which assigns number to each pixel on the image
            
              faces = facecade.detectMultiScale(imagearray)
            
              for(x,y,w,h) in faces:
                  roi = imagearray[y:y+h, x:x+w]
                  x_train.append(roi)
                  y_labels.append(id_)
                
  with open("labels.pickle", 'wb') as f:
      pickle.dump(label_ids, f)

  recognizer.train(x_train, np.array(y_labels))
  recognizer.save("trainedfile.yml")

            
  facecade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
  recognizer = cv2.face.LBPHFaceRecognizer_create()


  recognizer.read("trainedfile.yml")

  labels = {"person_name": 1}
  with open("labels.pickle", 'rb') as f:
      og_labels = pickle.load(f)
      labels = {v:k for k,v in og_labels.items()}

  cap = cv2.VideoCapture(0)

  while(True):
      # Capture frame-by-frame
      ret, frame = cap.read()
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
      faces = facecade.detectMultiScale(frame, scaleFactor= 1.5, minNeighbors=5)
      for(x,y,w,h) in faces:
        
          roi = gray[y:y+h, x:x+w]
        
          id_, conf = recognizer.predict(roi)
          if conf > 50:
              font = cv2.FONT_HERSHEY_SIMPLEX
              name = labels[id_]
              color = (150, 105,120)
              stroke = 2
              cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
              student.loc[name,"Status"]="Present"
        
          color = (210,135,75)
          stroke=4
          width = x+w
          height = y+h
          cv2.rectangle(frame, (x,y), (width, height), color, stroke)
        
    # Display the resulting frame
      cv2.imshow('frame',frame)
      if cv2.waitKey(20) & 0xFF == ord('q'):
          break
  student.to_csv(base_dir+"/"+"AIDI1003.csv",index=False)
# When everything done, release the capture
  cap.release()
  cv2.destroyAllWindows()
  print ('I got clicked!')
  import smtplib, ssl
  import csv

  port = 465  
  smtp_server = "smtp.gmail.com"
  sender_email = "argusteam0921@gmail.com"
  receiver_ids = {'dheeraj':'dheeraj.savanthy@dcmail.ca','alan':'oscaralan.lozadavilla@dcmail.ca','bharthwaj':'bharathwaj.thirumalaiananthanp@dcmail.ca','ryan':'ryan.shaw@dcmail.ca','mahesh':'mahesh.kamalakannan@dcmail.ca', 'marcos':'marcos.bittencourt@dcfaculty.mycampus.ca'}
  password = "$argusteam_0921"
  message1 = """\
  Subject: Hi there

  You have been marked Present for your attendance in the class today."""

  message2 = """\
  Subject: Hi there

  You have been marked Absent for missing the class today. Please contact me to discuss about your attendance and receive lecture material"""
  context = ssl.create_default_context()
  with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
      server.login(sender_email, password)
      with open("AIDI1003.csv") as file:
          reader = csv.reader(file)
          next(reader)  # Skip header row
          for names, status, Timestamp in reader:
              print(f"Sending email to {names}")
              if status == 'Present':
                  server.sendmail(sender_email, receiver_ids[names], message1)
              if status == 'Absent':
                  server.sendmail(sender_email, receiver_ids[names], message2)

  return 'Click.'

if __name__ == '__main__':
  app.run(debug=True)


