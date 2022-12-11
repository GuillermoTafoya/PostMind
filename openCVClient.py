import requests
import base64
import cv2

data = {}
cam = cv2.VideoCapture(0)
leido, frame = cam.read() 

def sendImage():
    leido, frame = cam.read() 
    if leido == True:
        frame = cv2.flip(frame, 1)
        cv2.imwrite("foto.jpg", frame)
        print("Foto tomada correctamente")
    else:
        print("Error al acceder a la cámara")

    #Codigo q envia imagen
    with open('foto.jpg', mode='rb') as file:
        img = file.read()
        img_64 = base64.b64encode(img)
        data['img'] = img_64.decode('utf-8')

    r = requests.put('http://192.168.100.21:5000/api/image', json = data)
    print(f"Status Code: {r.status_code}, Response: {r.json()}")

def getEmotion():
    r = requests.get('http://192.168.100.21:5000/api/emotion', json = data)
    print(f"Status Code: {r.status_code}, Response: {r.json()}")

if __name__ == "__main__":
    while True:
        text = input("Get emotion or set image? (GEM/SIM) ")
        if text == "SIM":
            sendImage()
        elif text == "GEM":
            getEmotion()
        elif text == "QUIT":
            break