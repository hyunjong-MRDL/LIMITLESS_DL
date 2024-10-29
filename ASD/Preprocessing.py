import cv2
import numpy as np

def get_boundaries(td_img):
    edges = cv2.Canny(td_img, 100, 200)  # Canny Algorithm은 uint8 타입의 이미지에 대해서만 사용 가능 (normalization 필요)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(all_contours)
    return x, y, w, h

def crop_brain(td_img):  # 2D nibabel image를 입력으로 받는다
    edges = cv2.Canny(td_img, 100, 200)  # Canny Algorithm은 uint8 타입의 이미지에 대해서만 사용 가능 (normalization 필요)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(all_contours)
    cropped_img = td_img[y:y+h, x:x+w]
    return cropped_img