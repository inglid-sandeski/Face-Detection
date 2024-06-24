import numpy as np
import cv2

classificador = cv2.CascadeClassifier('.venv\\Scripts\\haarcascade_frontalface_default.xml')

imagem = cv2.imread('.venv\\beatles.jpg')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

facesDetectadas = classificador.detectMultiScale(imagemCinza)
print(len(facesDetectadas))
print((facesDetectadas))

