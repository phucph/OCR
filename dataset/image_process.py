import numpy as np
import torch
# from numpy import random
import cv2


def random_crop_v2(image, boxes=None, tags=None, crop_size=(1280, 1280), max_tries=50):
    crop_h, crop_w = crop_size
    ori_h, ori_w, c = image.shape
    start_x_max = np.clip(ori_w - crop_w, 0, ori_w)
    start_y_max = np.clip(ori_h - crop_h, 0, ori_h)
    new_w = crop_w if start_x_max == 0 else ori_w
    new_h = crop_h if start_y_max == 0 else ori_h
    new_image = np.zeros((new_h, new_w, c))
    new_image[0:ori_h, 0:ori_w, :] = image
    if start_x_max == 0 and start_y_max == 0:
        return new_image, boxes, tags
    # ensure the croped area is not in box
    h_array = np.zeros((new_h), dtype=np.int32)
    w_array = np.zeros((new_w), dtype=np.int32)

    for box in boxes:
        box = np.round(box, decimals=0).astype(np.int32)
        minx = np.min(box[:, 0])
        maxx = np.max(box[:, 0])
        w_array[minx:maxx] = 1
        miny = np.min(box[:, 1])
        maxy = np.max(box[:, 1])
        h_array[miny:maxy] = 1

    # ensure the cropped area not across a text

    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    hb_axis = np.where(h_array == 1)[0]
    wb_axis = np.where(w_array == 1)[0]


    selected_boxes = []
    if len(h_axis) == 0 or len(w_axis) == 0:
        h, w, c = new_image.shape
        new_image = cv2.resize(new_image, (crop_w, crop_h))
        boxes[:, :, 0] *= crop_w * 1.0 / w
        boxes[:, :, 1] *= crop_h * 1.0 / h
        return new_image, boxes, tags
    for i in range(max_tries):
        rand_start_x = torch.randint(0, start_x_max + 1, (1,))[0]
        rand_end_x = rand_start_x + crop_w
        rand_start_y = torch.randint(0, start_y_max + 1, (1,))[0]
        rand_end_y = rand_start_y + crop_h
        x_valid = rand_start_x in w_axis and rand_end_x in w_axis
        y_valid = rand_start_y in h_axis and rand_end_y in h_axis
        if not (x_valid and y_valid):
            continue
        if boxes.shape[0] != 0:
            box_axis_in_area = (boxes[:, :, 0] >= rand_start_x) & (boxes[:, :, 0] <= rand_end_x) \
                               & (boxes[:, :, 1] >= rand_start_y) & (boxes[:, :, 1] <= rand_end_y)
            selected_boxes = np.where(np.sum(box_axis_in_area, axis=1) == 4)[0]
            if len(selected_boxes) > 0:
                break

    cropped_image = new_image[rand_start_y:rand_end_y,
                    rand_start_x:rand_end_x, :]
    boxes = boxes[selected_boxes]
    tags = tags[selected_boxes]
    print()
    # boxes[:, :, 0] -= rand_start_x
    # boxes[:, :, 1] -= rand_start_y
    return cropped_image, boxes, tags


def random_crop(image, boxes=None, tags=None, crop_size=(2176, 2176), max_tries=50):
    crop_h, crop_w = crop_size
    ori_h, ori_w, c = image.shape

    start_x_max = np.clip(ori_w - crop_w, 0, ori_w)
    start_y_max = np.clip(ori_h - crop_h, 0, ori_h)

    new_w = crop_w if start_x_max == 0 else ori_w
    new_h = crop_h if start_y_max == 0 else ori_h

    new_image = np.zeros((new_h, new_w, c))
    new_image[0:ori_h, 0:ori_w, :] = image

    # =====================================================
    # def resize_img(image, crop_size,  )
    # ======================================================
    if start_x_max == 0 or start_y_max == 0:
        new_image = cv2.resize(new_image, (crop_w, crop_h))
        boxes[:, :, 0] *= crop_w / new_w
        boxes[:, :, 1] *= crop_h / new_h
        return new_image, boxes, tags
    tries = 0

    if torch.rand(1).item() > 0.2 and np.sum(tags > 0) > 0:
        boxes_o, tags_o = boxes.copy(), tags.copy()
        while tries < max_tries:
            boxes, tags = boxes_o.copy(), tags_o.copy()
            rand_start_x = torch.randint(0, start_x_max + 1, (1,)).item()
            rand_end_x = rand_start_x + crop_w
            rand_start_y = torch.randint(0, start_y_max + 1, (1,)).item()
            rand_end_y = rand_start_y + crop_h


            cropped_image = image[rand_start_y:rand_end_y,
                            rand_start_x:rand_end_x, :]

            boxes[:, :, 0] = (boxes[:, :, 0] - rand_start_x)
            boxes[:, :, 1] = (boxes[:, :, 1] - rand_start_y)

            # boxes[:, :, 0]=np.clip(boxes[:,:,0],0,crop_w)
            # boxes[:, :, 1]=np.clip(boxes[:,:,1],0,crop_h)

            # filter empty box
            # for out side box, the x is all 0 or 1, the y is 0 or 1
            keep = ~(((boxes[:, :, 0] <= 0.).sum(1) == 4) | ((boxes[:, :, 0] >= crop_w).sum(1) == 4)
                     | ((boxes[:, :, 1] <= 0.).sum(1) == 4) | ((boxes[:, :, 1] >= crop_h).sum(1) == 4))

            boxes = boxes[keep]
            tags = tags[keep]
            if np.sum(tags > 0) > 0:
                break
            tries = tries + 1
    else:
        tags_o = tags.copy()
        rand_start_x = torch.randint(0, start_x_max + 1, (1,)).item()
        rand_end_x = rand_start_x + crop_w
        rand_start_y = torch.randint(0, start_y_max + 1, (1,)).item()
        rand_end_y = rand_start_y + crop_h


        cropped_image = image[rand_start_y:rand_end_y,
                        rand_start_x:rand_end_x, :]

        boxes[:, :, 0] = (boxes[:, :, 0] - rand_start_x)
        boxes[:, :, 1] = (boxes[:, :, 1] - rand_start_y)

        # boxes[:, :, 0]=np.clip(boxes[:,:,0],0,crop_w)
        # boxes[:, :, 1]=np.clip(boxes[:,:,1],0,crop_h)

        # filter empty box
        # for out side box, the x is all 0 or 1, the y is 0 or 1
        keep = ~(((boxes[:, :, 0] <= 0.).sum(1) == 4) | ((boxes[:, :, 0] >= crop_w).sum(1) == 4)
                 | ((boxes[:, :, 1] <= 0.).sum(1) == 4) | ((boxes[:, :, 1] >= crop_h).sum(1) == 4))

        boxes = boxes[keep]
        tags = tags[keep]
        if np.sum(tags > 0) < 0:
          scale = crop_h * 1.0 / max(ori_h, ori_w)
          cropped_image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
          if boxes is not None:
              boxes[:, :, 0] *= scale
              boxes[:, :, 1] *= scale
              boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, crop_w)
              boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, crop_h)
          tags = tags_o

    return cropped_image, boxes, tags


def random_ratio_scale(image, boxes=None, tags=None, ratios=np.arange(0.8, 1.3, 0.1)):
    ori_h, ori_w, c = image.shape
    if min(ori_h, ori_w) >= (2480 - 32):
        ratios = np.arange(0.9, 1.15, 0.01)
    rand_index = torch.randint(0, len(ratios) - 1, (1,)).item()
    ratio = ratios[rand_index]
    ratio = np.sqrt(ratio)
    new_h = int(ori_h * ratio)
    new_w = int(ori_w / ratio)
    new_image = cv2.resize(image, (new_w, new_h))
    boxes[:, :, 1] *= ratio
    boxes[:, :, 0] /= ratio
    return new_image, boxes, tags


def random_resize(image, boxes=None, tags=None, longer_sides=np.arange(1840, 2480, 32)):
    ori_h, ori_w, c = image.shape
    if max(ori_h, ori_w <= 2480):
      return image, boxes, tags
    rand_index = torch.randint(0, len(longer_sides) - 1, (1,)).item()
    longer_side = longer_sides[rand_index]
    ratio = longer_side * 1.0 / ori_h
    boxes[:, :, 1] *= ratio
    boxes[:, :, 0] *= ratio
    new_image = cv2.resize(image, (longer_side, longer_side))
    return new_image, boxes, tags


def random_rotate(image, boxes=None, tags=None, rotate_angles=np.arange(-5, 5, 1)):
    rand_index = torch.randint(0, len(rotate_angles) - 1, (1,)).item()
    angle = rotate_angles[rand_index]
    ori_h, ori_w, _ = image.shape
    cX, cY = ori_w // 2, ori_h // 2
    matrix = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos, sin = matrix[0, 0], -matrix[0, 1]
    abs_cos, abs_sin = np.abs(cos), np.abs(sin)
    nW = int((ori_h * abs_sin) + (ori_w * abs_cos))
    nH = int((ori_h * abs_cos) + (ori_w * abs_sin))
    matrix[0, 2] += (nW / 2) - cX
    matrix[1, 2] += (nH / 2) - cY
    new_image = cv2.warpAffine(image, matrix, (nW, nH), borderValue=(128, 128, 128))
    new_boxes = np.zeros_like(boxes)
    boxes -= np.array([cX, cY])
    new_boxes[:, :, 0] = (boxes[:, :, 0] * cos - boxes[:, :, 1] * sin) + nW / 2
    new_boxes[:, :, 1] = (boxes[:, :, 1] * cos + boxes[:, :, 0] * sin) + nH / 2
    return new_image, new_boxes, tags
