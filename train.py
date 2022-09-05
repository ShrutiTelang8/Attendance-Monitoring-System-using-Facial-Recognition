# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 09:02:18 2019

@author: Shruti Telang
"""
import tkinter as tk
import cv2,os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font

window = tk.Tk()
window.title("Face_Recogniser")
dialog_title = 'QUIT'
dialog_text = 'Are you sure?'
#p=tk.PhotoImage("a.jpg")
window.configure(bg="firebrick4")

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

#GUI
message = tk.Label(window, text="Facial Recognition for Attendance System" ,bg="white"  ,fg="firebrick4"  ,width=45  ,height=3,font=('missy', 30, 'italic bold')) 

message.place(x=200, y=20)

lbl = tk.Label(window, text="Enter Roll No.",width=15  ,height=2  ,fg="firebrick4"  ,bg="white" ,font=('missy', 15, ' italic bold ') ) 
lbl.place(x=400, y=200)

txt = tk.Entry(window,width=15  ,bg="white" ,fg="firebrick4",font=('missy', 15, ' italic bold '))
txt.place(x=700, y=215)

lbl2 = tk.Label(window, text="Enter Name",width=15  ,fg="firebrick4"  ,bg="white"    ,height=2 ,font=('missy', 15, ' italic bold ')) 
lbl2.place(x=400, y=300)

txt2 = tk.Entry(window,width=15  ,bg="white"  ,fg="firebrick4",font=('missy', 15, ' italic bold ')  )
txt2.place(x=700, y=315)

lbl3 = tk.Label(window, text="Notification ",width=15  ,fg="firebrick4"  ,bg="white"  ,height=2 ,font=('missy', 15, ' italic bold ')) 
lbl3.place(x=400, y=400)

message = tk.Label(window, text="" ,bg="white"  ,fg="firebrick4"  ,width=30  ,height=2, activebackground = "yellow" ,font=('missy', 15, ' italic bold ')) 
message.place(x=700, y=400)

lbl3 = tk.Label(window, text="Attendance ",width=15  ,fg="firebrick4"  ,bg="white"  ,height=2 ,font=('missy', 15, ' italic bold ')) 
lbl3.place(x=400, y=650)


message2 = tk.Label(window, text="" ,fg="firebrick4"   ,bg="white",activeforeground = "firebrick4",width=30  ,height=2  ,font=('missy', 15, ' italic bold ')) 
message2.place(x=700, y=650)
 
def clear():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)

def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)    
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

def TakeImages():        
    Id=(txt.get())
    name=(txt2.get())
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                #incrementing sample number 
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame
                cv2.imshow('frame',img)
            #wait for 100 miliseconds 
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum>60:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Images Saved for Roll No. : " + Id +" Name : "+ name
        row = [Id , name]
        with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
    else:
        if(is_number(Id)):
            res="Enter Alphabetical Name"
            message.configure(text= res)
        if(name.isalpha()):
            res="Enter Numeric Id"
            message.configure(text= res)
    
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath="haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("Trainner.yml")
    res="Image Trained"
    message.configure(text= res)

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    faces=[]
    Ids=[]
    
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font=cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Roll No.','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
            ts = time.time()      
            date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            Hour,Minute,Second=timeStamp.split(":")
    
            fileName="Attendance\Attendance_"+date+"_"+Hour+".csv"
            #attendance.to_csv(fileName,index=False)
            with open(fileName, "a", newline='') as fo:
                wr=csv.writer(fo,dialect="excel")
                wr.writerow([Id,aa,date,timeStamp])
            res=attendance
            message2.configure(text= res)
                                
        cam.release()
        cv2.destroyAllWindows()
            #print(attendance)
  
clearButton = tk.Button(window, text="Clear", command=clear  ,fg="firebrick4"  ,bg="white"  ,width=11  ,height=2 ,activebackground = "Red" ,font=('missy', 15, ' italic bold '))
clearButton.place(x=950, y=200)
clearButton2 = tk.Button(window, text="Clear", command=clear2  ,fg="firebrick4"  ,bg="white"  ,width=11  ,height=2, activebackground = "Red" ,font=('missy', 15, ' italic bold '))
clearButton2.place(x=950, y=300)    
takeImg = tk.Button(window, text="Take Images", command=TakeImages  ,fg="firebrick4"  ,bg="white"  ,width=20  ,height=3, activebackground = "Red" ,font=('missy', 15, ' italic bold '))
takeImg.place(x=200, y=500)
trainImg = tk.Button(window, text="Train Images", command=TrainImages  ,fg="firebrick4"  ,bg="white"  ,width=20  ,height=3, activebackground = "Red" ,font=('missy', 15, ' italic bold '))
trainImg.place(x=500, y=500)
trackImg = tk.Button(window, text="Track Images", command=TrackImages  ,fg="firebrick4"  ,bg="white"  ,width=20  ,height=3, activebackground = "Red" ,font=('missy', 15, ' italic bold '))
trackImg.place(x=800, y=500)
quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg="firebrick4"  ,bg="white"  ,width=20  ,height=3, activebackground = "Red" ,font=('missy', 15, ' italic bold '))
quitWindow.place(x=1100, y=500)
copyWrite = tk.Text(window, background=window.cget("background"), borderwidth=0,font=('missy', 30, 'italic bold '))
copyWrite.tag_configure("superscript", offset=10)
copyWrite.configure(state="disabled",fg="black"  )
copyWrite.pack(side="left")
copyWrite.place(x=800, y=750)
 
window.mainloop()