import cv2
import numpy as np

import albumentations as A

class AlbumentationsAug:
    def __init__(self):
        # 定义transforms列表
        self.transforms = A.Compose([
            A.Blur(p=0.1),
            A.MedianBlur(p=0.1),
            A.CoarseDropout(
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0,
            ),
        ])

    def __call__(self, image):
        # 对输入图像应用transforms
        return self.transforms(image=image)['image']

class YOLOXHSVRandomAug:
    """Apply HSV augmentation to image sequentially. It is referenced from
    https://github.com/Megvii-
    BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py#L21.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        hue_delta (int): delta of hue. Defaults to 5.
        saturation_delta (int): delta of saturation. Defaults to 30.
        value_delta (int): delat of value. Defaults to 30.
    """

    def __init__(self, hue_delta=5, saturation_delta=30, value_delta=30):
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta

    def _get_hsv_gains(self):
        hsv_gains = np.random.uniform(-1, 1, 3) * [
            self.hue_delta, self.saturation_delta, self.value_delta
        ]
        # random selection of h, s, v
        hsv_gains *= np.random.randint(0, 2, 3)
        # prevent overflow
        hsv_gains = hsv_gains.astype(np.int16)
        return hsv_gains

    def __call__(self, img):
        # 假设img是cv2.RGB图像
        # img = results['img']
        hsv_gains = self._get_hsv_gains()
        # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)

        img_hsv[..., 0] = (img_hsv[..., 0] + hsv_gains[0]) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_gains[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_gains[2], 0, 255)
        # cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)
        cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2RGB, dst=img)

        # results['img'] = img
        # return results
        return img

def vertical_flip(image, prob=0.5, keypoints=None):
    """
    Perform vertical flip on the given images and corresponding keypoints.
    Args:
        image (ndarray): images to perform vertical flip, the dimension is
             `height` x `width` x `channel`.
        prob (float): probility to flip the images.
        keypoints (ndarray or None): optional. Corresponding 3D keypoints to images.
            Dimension is `num joints` x 3.
    Returns:
        images (ndarray): flipped images with dimension of
            `height` x `width` x `channel`.
        flipped_keypoints (ndarray or None): the flipped keypoints with dimension of
            `num keypoints` x 3.
    """
    if keypoints is None:
        flipped_keypoints = None
    else:
        flipped_keypoints = keypoints.copy()

    if np.random.uniform() < prob:
        # images = images.flip((-1))
        image = np.flip(image, axis=0)

        # if len(images.shape) == 3:
        #     width = images.shape[2]
        # elif len(images.shape) == 4:
        #     width = images.shape[3]
        # else:
        #     raise NotImplementedError("Dimension does not supported")
        if keypoints is not None:
            # flipped_boxes[:, [0, 2]] = width - boxes[:, [2, 0]] - 1
            flipped_keypoints[:, 1] = - flipped_keypoints[:, 1]

    return image, flipped_keypoints