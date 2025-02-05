#Ce programme permet de détecter les keypoints d'une personne dans une vidéo

import cv2
from ultralytics import YOLO

def tensor_to_tuple(tensor):
    # Convertir le tensor en liste
    list_value = tensor.tolist()

    # Convertir la liste en tuple
    tuple_value = tuple(int(liste_value) for liste_value in list_value)

    return tuple_value

def draw_line_between_keypoints(frame, keypoint1, keypoint2):
    """ Dessine une ligne entre deux keypoints sur le frame. """
    keypoint1 = tensor_to_tuple(keypoint1)
    keypoint2 = tensor_to_tuple(keypoint2)
    cv2.line(frame, keypoint1, keypoint2, color=(0, 255, 0), thickness=2)

# Charger le modèle YOLOv8 pour la détection de pose
model = YOLO('yolov8n-pose.pt')

# Liste pour stocker les keypoints pour chaque image de la vidéo
keypoints_list = []

# Ouvrir le flux vidéo d'entrée
video_capture = cv2.VideoCapture("new/video_detek.mp4")
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))

# Définir le codec et créer un objet VideoWriter pour enregistrer la vidéo
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('new/video_detek_stick.mp4', fourcc, fps, (frame_width, frame_height))

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    # Effectuer la détection sur l'image
    results = model(frame, conf=0.8)

    # Parcourir chaque élément dans les résultats
    for item in results:
        keypoints_frame = []  # Liste pour stocker les keypoints de cette frame

        # Parcourir chaque ensemble de points clés pour chaque personne détectée
        for xy, xyn in zip(item.keypoints.xy, item.keypoints.xyn):
            try:
                # Récupérer les coordonnées des épaules, coudes et mains
                left_shoulder = xy[6]  # Indice 6 correspondant à l'épaule gauche
                left_elbow = xy[8]  # Indice 8 correspondant au coude gauche
                left_hand = xy[10]  # Indice 10 correspondant à la main gauche
                right_shoulder = xy[5]  # Indice 5 correspondant à l'épaule droite
                right_elbow = xy[7]  # Indice 7 correspondant au coude droit
                right_hand = xy[9]  # Indice 9 correspondant à la main droite
                left_eye = xy[1]  # Indice 1 correspondant à l'œil gauche
                right_eye = xy[2]  # Indice 2 correspondant à l'œil droit

                shoulder_to_elbow_left = draw_line_between_keypoints(frame, left_shoulder, left_elbow)
                elbow_to_hand_left = draw_line_between_keypoints(frame, left_elbow, left_hand)
                shoulder_to_elbow_right = draw_line_between_keypoints(frame, right_shoulder, right_elbow)
                elbow_to_hand_right = draw_line_between_keypoints(frame, right_elbow, right_hand)
                eye_to_eye = draw_line_between_keypoints(frame, left_eye, right_eye)

            except Exception as e:
                    # Afficher l'erreur et continuer avec la prochaine frame
                    print("Error:", e)

    # Écrire la frame modifiée dans la sortie vidéo
    out.write(frame)

# Libérer les ressources
video_capture.release()
out.release()