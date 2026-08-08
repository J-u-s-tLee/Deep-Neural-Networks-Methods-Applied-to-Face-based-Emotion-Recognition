from unittest import skip, skipUnless
import mediapipe as mp
import cv2
import os

#image1 = cv2.imread('C:\\Users\\lm803\\OneDrive\\Ambiente de Trabalho\\Data FER+\\train\imgs\\fer0000025.png') # Reads image

# Input directory. File containing images
input_dir = 'C:\\Users\\lm803\\OneDrive\\Ambiente de Trabalho\\Data FER+\\train\imgs'

# Iterate over each image in the input directory
for filename in os.listdir(input_dir):
    
    image_path = os.path.join(input_dir, filename) # Create sting with full image path
    image1 = cv2.imread(image_path)
    image = cv2.resize(image1,(500,500));
    image_copy1 = image.copy();
    image_copy2 = image.copy();
    height, width,_ = image.shape # Get height, width and channels

    facemesh = mp.solutions.face_mesh # Importing face_mesh module from mediapipe as facemesh.
    face_mesh = facemesh.FaceMesh() # Creating an instance of the FaceMesh class from facemesh. 
    # Provides methods to detect facial landmarks and create a 3D representation of the face.

    result = face_mesh.process(image) # Apply the FaceMesh solution to the input image, detecting facial landmarks.

    # Only one face is detected. Face landmarks stores in result.multi_face_landmarks[0]. 
    # Each face_landmarks will have 468 landmarks. Normalized to the height and width of the original image.

    if result.multi_face_landmarks:
        
        for face_landmarks in result.multi_face_landmarks:
            for i in range(0, 468):
             x = int(face_landmarks.landmark[i].x * width)
             y = int(face_landmarks.landmark[i].y * height)
    
             cv2.circle(image_copy1, (x,y), 1, (255, 0, 0), -1) # Draws circles in image

        # source_idx and target_idx are the indexes of two sequencial landmarks
     
        for source_idx, target_idx in facemesh.FACEMESH_LEFT_EYE:
   
            source = face_landmarks.landmark[source_idx] # Gets the first landmark coordinates
            target = face_landmarks.landmark[target_idx] # Gets the second landmark coordinates
    
            # Coordinates relative to image
            relative_source = (int(source.x*image.shape[1]), int(source.y*image.shape[0]))
            relative_target = (int(target.x*image.shape[1]), int(target.y*image.shape[0]))

            # Draw lines betweeen the pair of landmarks until the end
            cv2.line(image_copy2, relative_source, relative_target, (0, 255, 0), 2)
    
        for source_idx, target_idx in facemesh.FACEMESH_RIGHT_EYE:
   
            source = face_landmarks.landmark[source_idx]
            target = face_landmarks.landmark[target_idx]

            relative_source = (int(source.x*image.shape[1]), int(source.y*image.shape[0]))
            relative_target = (int(target.x*image.shape[1]), int(target.y*image.shape[0]))

            cv2.line(image_copy2, relative_source, relative_target, (0, 255, 0), 2)
    
        for source_idx, target_idx in facemesh.FACEMESH_RIGHT_EYEBROW:
   
            source = face_landmarks.landmark[source_idx] 
            target = face_landmarks.landmark[target_idx] 

            relative_source = (int(source.x*image.shape[1]), int(source.y*image.shape[0]))
            relative_target = (int(target.x*image.shape[1]), int(target.y*image.shape[0]))

            cv2.line(image_copy2, relative_source, relative_target, (255, 0, 0), 2)
    
        for source_idx, target_idx in facemesh.FACEMESH_LEFT_EYEBROW:
   
            source = face_landmarks.landmark[source_idx]
            target = face_landmarks.landmark[target_idx]
    
            relative_source = (int(source.x*image.shape[1]), int(source.y*image.shape[0]))
            relative_target = (int(target.x*image.shape[1]), int(target.y*image.shape[0]))

            cv2.line(image_copy2, relative_source, relative_target, (255, 0, 0), 2)
 
        for source_idx, target_idx in facemesh.FACEMESH_LIPS:
   
            source = face_landmarks.landmark[source_idx] 
            target = face_landmarks.landmark[target_idx] 
    
            relative_source = (int(source.x*image.shape[1]), int(source.y*image.shape[0]))
            relative_target = (int(target.x*image.shape[1]), int(target.y*image.shape[0]))

            cv2.line(image_copy2, relative_source, relative_target, (0, 0, 255), 2)
    
        for source_idx, target_idx in facemesh.FACEMESH_FACE_OVAL:
   
            source = face_landmarks.landmark[source_idx]
            target = face_landmarks.landmark[target_idx]

            relative_source = (int(source.x*image.shape[1]), int(source.y*image.shape[0]))
            relative_target = (int(target.x*image.shape[1]), int(target.y*image.shape[0]))

            cv2.line(image_copy2, relative_source, relative_target, (125, 0, 125), 2)
    
        for source_idx, target_idx in facemesh.FACEMESH_NOSE:
   
            source = face_landmarks.landmark[source_idx]
            target = face_landmarks.landmark[target_idx]

            relative_source = (int(source.x*image.shape[1]), int(source.y*image.shape[0]))
            relative_target = (int(target.x*image.shape[1]), int(target.y*image.shape[0]))

            cv2.line(image_copy2, relative_source, relative_target, (10, 10, 0), 2)
    
        cv2.imshow("Image", image_copy1)
        cv2.waitKey(0)
        cv2.imshow("Image", image_copy2)
        cv2.waitKey(0)