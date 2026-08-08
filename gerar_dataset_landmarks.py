import mediapipe as mp
import cv2
import os
import numpy as np

# Input directory. File containing images
input_dir = 'C:\\Users\\lm803\\OneDrive\\Ambiente de Trabalho\\Estagio\\Data FER+\\test\\imgs'

facemesh = mp.solutions.face_mesh # Importing face_mesh module from mediapipe as facemesh.
face_mesh = facemesh.FaceMesh() # Creating an instance of the FaceMesh class from facemesh. 

# Iterate over each image in the input directory
for filename in os.listdir(input_dir):
    
    # Create black image 
    image2 = np.zeros([224,224,3], dtype=np.uint8)
    
    image_path = os.path.join(input_dir, filename) # Create sting with full image path
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    height, width,_ = image.shape # Get height, width and channels

    result = face_mesh.process(image) # Apply the FaceMesh solution to the input image, detecting facial landmarks.

    # Only one face is detected. Face landmarks stores in result.multi_face_landmarks[0]. 
    # Each face_landmarks will have 468 landmarks. Normalized to the height and width of the original image.

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
           for idx in facemesh.FACEMESH_LEFT_EYE:   
               coordinates1 = face_landmarks.landmark[idx[0]] 
               coordinates2 = face_landmarks.landmark[idx[1]]
               true_coord1 = (int(coordinates1.x*width), int(coordinates1.y*height))
               true_coord2 = (int(coordinates2.x*width), int(coordinates2.y*height))
               cv2.circle(image2, (true_coord1[0],true_coord1[1]), 1, (255, 255, 255), -1) # Draws circles in image
               cv2.circle(image2, (true_coord2[0],true_coord2[1]), 1, (255, 255, 255), -1) # Draws circles in image

           for idx in facemesh.FACEMESH_RIGHT_EYE:   
               coordinates1 = face_landmarks.landmark[idx[0]] 
               coordinates2 = face_landmarks.landmark[idx[1]]
               true_coord1 = (int(coordinates1.x*width), int(coordinates1.y*height))
               true_coord2 = (int(coordinates2.x*width), int(coordinates2.y*height))
               cv2.circle(image2, (true_coord1[0],true_coord1[1]), 1, (255, 255, 255), -1) # Draws circles in image
               cv2.circle(image2, (true_coord2[0],true_coord2[1]), 1, (255, 255, 255), -1) # Draws circles in image
               
           for idx in facemesh.FACEMESH_RIGHT_EYEBROW:   
               coordinates1 = face_landmarks.landmark[idx[0]] 
               coordinates2 = face_landmarks.landmark[idx[1]]
               true_coord1 = (int(coordinates1.x*width), int(coordinates1.y*height))
               true_coord2 = (int(coordinates2.x*width), int(coordinates2.y*height))
               cv2.circle(image2, (true_coord1[0],true_coord1[1]), 1, (255, 255, 255), -1) # Draws circles in image
               cv2.circle(image2, (true_coord2[0],true_coord2[1]), 1, (255, 255, 255), -1) # Draws circles in image
           
           for idx in facemesh.FACEMESH_LEFT_EYEBROW:   
               coordinates1 = face_landmarks.landmark[idx[0]] 
               coordinates2 = face_landmarks.landmark[idx[1]]
               true_coord1 = (int(coordinates1.x*width), int(coordinates1.y*height))
               true_coord2 = (int(coordinates2.x*width), int(coordinates2.y*height))
               cv2.circle(image2, (true_coord1[0],true_coord1[1]), 1, (255, 255, 255), -1) # Draws circles in image
               cv2.circle(image2, (true_coord2[0],true_coord2[1]), 1, (255, 255, 255), -1) # Draws circles in image
               
           for idx in facemesh.FACEMESH_LIPS:   
               coordinates1 = face_landmarks.landmark[idx[0]] 
               coordinates2 = face_landmarks.landmark[idx[1]]
               true_coord1 = (int(coordinates1.x*width), int(coordinates1.y*height))
               true_coord2 = (int(coordinates2.x*width), int(coordinates2.y*height))
               cv2.circle(image2, (true_coord1[0],true_coord1[1]), 1, (255, 255, 255), -1) # Draws circles in image
               cv2.circle(image2, (true_coord2[0],true_coord2[1]), 1, (255, 255, 255), -1) # Draws circles in image
                
           for idx in facemesh.FACEMESH_NOSE:   
               coordinates1 = face_landmarks.landmark[idx[0]] 
               coordinates2 = face_landmarks.landmark[idx[1]]
               true_coord1 = (int(coordinates1.x*width), int(coordinates1.y*height))
               true_coord2 = (int(coordinates2.x*width), int(coordinates2.y*height))
               cv2.circle(image2, (true_coord1[0],true_coord1[1]), 1, (255, 255, 255), -1) # Draws circles in image
               cv2.circle(image2, (true_coord2[0],true_coord2[1]), 1, (255, 255, 255), -1) # Draws circles in image
               
    cv2.imwrite(f'C:\\Users\\lm803\\OneDrive\\Ambiente de Trabalho\\Estagio\\Data FER+ Face\\test\\{filename}', image2)
        
print("All images saved successfully!")
