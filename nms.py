import numpy as np


def nms(rois, threshold):

    """

    :param rois: numpy array, [x1, y1, x2, y2, score]
    :param threshold: float
    :return: nms results
    """

    x1 = rois[:, 0]
    y1 = rois[:, 1]
    x2 = rois[:, 2]
    y2 = rois[:, 3]
    score = rois[:, 4]

    order = score.argsort()[::-1]
    area = (x2-x1+1) * (y2-y1+1)

    result = []

    while order:

        result.append(order[0])

        left_max = np.maximum(x1[order[0]], x1[order[1:]])
        right_min = np.minimum(x2[order[0]], x2[order[1:]])
        top_max = np.maximum(y1[order[0]], y1[order[1:]])
        bot_min = np.minimum(y2[order[0]], y2[order[1:]])

        w = np.maximum(0, right_min-left_max)
        h = np.maximum(0, bot_min-top_max)
        
        iou4all = (w-h)/(area[order[0]]+area[order[1:]]-(w-h))
        keep = np.where(iou4all<=threshold)[0]
        order = order[keep+1]

    return result



