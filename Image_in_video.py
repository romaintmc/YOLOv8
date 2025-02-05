#Ce programme permet d'insérer une image entre les mains, dans une vidéo

import cv2
import numpy as np
from ultralytics import YOLO

def place_image_between_hands(image, right_hand, left_hand):
    """ Place une image entre les mains. """
    # Calculer le milieu entre les deux mains
    midpoint_x = (right_hand[0] + left_hand[0]) // 2
    midpoint_y = (right_hand[1] + left_hand[1]) // 2
    midpoint = (midpoint_x, midpoint_y)

    # Redimensionner l'image à une taille proportionnelle à la distance entre les mains
    distance = np.linalg.norm(np.array(right_hand) - np.array(left_hand))
    scale_factor = distance / max(image.shape[:2]) # Facteur d'échelle
    resized_image = cv2.resize(image, (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)))

    # Calculer les coordonnées du coin supérieur gauche pour le placement de l'image
    top_left_x = midpoint[0] - resized_image.shape[1] // 2
    top_left_y = midpoint[1] - resized_image.shape[0] // 2

    return resized_image, (top_left_x, top_left_y)

# Charger le modèle YOLOv8 pour la détection de pose
model = YOLO('yolov8n-pose.pt')

# Charger l'image à insérer dans la vidéo
image = cv2.imread("new/tableau.png", cv2.IMREAD_UNCHANGED)

# Liste pour stocker les keypoints pour chaque image de la vidéo
keypoints_list = []

# Ouvrir le flux vidéo d'entrée
video_capture = cv2.VideoCapture("new/video_deux.mp4")
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))

# Définir le codec et créer un objet VideoWriter pour enregistrer la vidéo
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('new/video_deux_tableau.mp4', fourcc, fps, (frame_width, frame_height))

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    # Effectuer la détection sur l'image
    results = model(frame, conf = 0.8)

    # Parcourir chaque élément dans les résultats
    for item in results:
        keypoints_frame = []  # Liste pour stocker les keypoints de cette frame
        # Parcourir chaque ensemble de points clés pour chaque personne détectée
        for xy, xyn in zip(item.keypoints.xy, item.keypoints.xyn):
            try:
                right_hand = [int(coordinate) for coordinate in xy[10]]  # Convertir le tensor en liste d'entiers
                left_hand = [int(coordinate) for coordinate in xy[9]]  # Convertir le tensor en liste d'entiers

                # Insérer l'image entre les mains
                resized_image, top_left_coord = place_image_between_hands(image, left_hand, right_hand)

                if resized_image is None:
                    raise ValueError("The function 'place_image_between_hands' returned None. Check the function.")

                # Convertir les coordonnées du coin supérieur gauche en entiers
                top_left_x, top_left_y = int(top_left_coord[0]), int(top_left_coord[1])

                # Extraire le canal alpha de l'image
                alpha_channel = resized_image[:, :, 3] / 255.0  # Normaliser les valeurs entre 0 et 1

                # Superposer l'image sur la frame
                overlay_image = frame[top_left_y:top_left_y + resized_image.shape[0], top_left_x:top_left_x + resized_image.shape[1]] * (1 - alpha_channel[:, :, None])
                overlay_image += resized_image[:, :, :3] * alpha_channel[:, :, None]
                frame[top_left_y:top_left_y + resized_image.shape[0],
                top_left_x:top_left_x + resized_image.shape[1]] = overlay_image.astype('uint8')
            except Exception as e:
                    # Afficher l'erreur et continuer avec la prochaine frame
                    print("Error:", e)

    # Écrire la frame modifiée dans la sortie vidéo
    out.write(frame)

# Libérer les ressources
video_capture.release()
out.release()