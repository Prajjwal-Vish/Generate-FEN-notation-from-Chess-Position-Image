import cv2
import numpy as np
from typing import Tuple, List, Union, Optional

# Configuration (tweak if needed)
MODEL_SQUARE_SIZE = 64
BOARD_SIZE_PX = MODEL_SQUARE_SIZE * 8  # 512

def _square_from_bbox(x:int, y:int, w:int, h:int, img_w:int, img_h:int, pad:int=0) -> Tuple[int,int,int,int]:
    side = max(w, h) + 2*pad
    cx = x + w // 2
    cy = y + h // 2
    half = side // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(img_w, cx + half)
    y2 = min(img_h, cy + half)
    side_w = x2 - x1
    side_h = y2 - y1
    side = min(side_w, side_h)
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    return int(x1), int(y1), int(side), int(side)

def _detect_square_bbox(image: np.ndarray) -> Tuple[int,int,int,int]:
    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    edges = cv2.Canny(gray, 40, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        img_area = float(w_img * h_img)
        quad_candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 0.005 * img_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                x,y,w,h = cv2.boundingRect(approx)
                aspect = w / float(h) if h != 0 else 0
                score = (min(aspect, 1.0/aspect)) * (area / img_area)
                quad_candidates.append((score, area, (x,y,w,h)))
        if quad_candidates:
            quad_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            x,y,w,h = quad_candidates[0][2]
            print(f"Selected quad contour bbox: x={x}, y={y}, w={w}, h={h}")
            return _square_from_bbox(x,y,w,h, w_img, h_img, pad=4)

        # fallback to large-ish near-square bounding rects
        rect_candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 0.01 * img_area:
                continue
            x,y,w,h = cv2.boundingRect(cnt)
            aspect = w / float(h) if h != 0 else 0
            if 0.6 <= aspect <= 1.6:
                rect_candidates.append((area, (x,y,w,h)))
        if rect_candidates:
            rect_candidates.sort(key=lambda x: x[0], reverse=True)
            x,y,w,h = rect_candidates[0][1]
            print(f"Selected rect candidate bbox: x={x}, y={y}, w={w}, h={h}")
            return _square_from_bbox(x,y,w,h, w_img, h_img, pad=6)

        # minAreaRect fallback
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 0.01 * img_area:
                continue
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype(np.int32)
            x_min = int(box[:,0].min())
            y_min = int(box[:,1].min())
            x_max = int(box[:,0].max())
            y_max = int(box[:,1].max())
            w = x_max - x_min
            h = y_max - y_min
            if w > 10 and h > 10:
                print(f"Selected minAreaRect fallback bbox: x={x_min}, y={y_min}, w={w}, h={h}")
                return _square_from_bbox(x_min, y_min, w, h, w_img, h_img, pad=6)

    # final centered fallback
    print("Warning: no suitable contour found -> will fallback to center crop.")
    margin = int(min(h_img, w_img) * 0.04)
    side = min(h_img, w_img) - 2 * margin
    cx, cy = w_img // 2, h_img // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    return (int(x1), int(y1), int(side), int(side))

def _crop_board(image: np.ndarray, bbox: Tuple[int,int,int,int]) -> np.ndarray:
    x, y, w, h = bbox
    h_img, w_img = image.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)
    crop = image[y1:y2, x1:x2].copy()
    if crop.size == 0:
        raise ValueError("Empty crop produced; bbox may be invalid.")
    return crop

def _preprocess_board_to_tiles(board_image: np.ndarray) -> List[np.ndarray]:
    resized = cv2.resize(board_image, (BOARD_SIZE_PX, BOARD_SIZE_PX), interpolation=cv2.INTER_AREA)
    tiles: List[np.ndarray] = []
    for r in range(8):
        for c in range(8):
            y1 = r * MODEL_SQUARE_SIZE
            y2 = (r + 1) * MODEL_SQUARE_SIZE
            x1 = c * MODEL_SQUARE_SIZE
            x2 = (c + 1) * MODEL_SQUARE_SIZE
            tile = resized[y1:y2, x1:x2]
            tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            tile_f = tile_rgb.astype(np.float32)
            tiles.append(tile_f)
    return tiles

ImageInput = Union[np.ndarray, bytes, str]

def process_image(image_input: ImageInput) -> Optional[Tuple[List[np.ndarray], np.ndarray, Tuple[int,int,int,int]]]:

    try:
        # load image depending on type
        if isinstance(image_input, np.ndarray):
            image = image_input
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Could not decode image from given bytes.")
        elif isinstance(image_input, str):
            image = cv2.imread(image_input, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read image from path: {image_input}")
        else:
            raise TypeError("image_input must be numpy array, bytes, or path string.")

        # detect board bbox
        bbox = _detect_square_bbox(image)
        # crop
        board_crop = _crop_board(image, bbox)
        # preprocess tiles
        model_inputs = _preprocess_board_to_tiles(board_crop)
        # Ensure board_image returned is standardized 512x512 BGR uint8
        board_image = cv2.resize(board_crop, (BOARD_SIZE_PX, BOARD_SIZE_PX), interpolation=cv2.INTER_AREA)
        return model_inputs, board_image, bbox
    except Exception as e:
        print(f"Error in process_image: {e}")
        return None
