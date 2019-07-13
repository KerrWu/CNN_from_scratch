import numpy as np


def img2col(img, k_h, k_w, s=1):
    """
    high performance convolution operation, convert a 4-d tensor to a 2-d matrix so that
    the convolution operation could achieve just by computing a mat mul

    :param img: batch img - 4d, [N, H, W, C]
    :param k_h: kernel height
    :param k_w: kernel width
    :param s: stride
    :return: matrix - 2d, [N*out_H*out_W, k_h*k_w*C]
    """

    # 将输入划分成若干个与卷积核相同大小的不同子集
    N, H, W, C = img.shape
    out_H = (H - k_h) // s + 1
    out_W = (W - k_w) // s + 1

    # 一次矩阵乘法完成所有卷积运算
    # 对应也要将卷积核变为 [k_h*k_w*C, out_channel]
    result = np.zeros((N * out_H * out_W, k_h * k_w * C))
    outsize = out_H * out_W

    for y in range(out_H):

        ymin = y * s
        ymax = ymin * s + k_h
        ystart = y * out_W

        for x in range(out_W):
            xmin = x * s
            xmax = xmin * s + k_w

            # 从img上取出一次卷积对应的区域
            result[ystart + x::outsize, :] = img[:, ymin:ymax, xmin:xmax, :].reshape(N, -1)

    return result, out_H, out_W


def conv2d(x, w, s=1, padding="SAME"):
    out_c, k_h, k_w, in_c = w.shape

    if padding == "SAME":
        p_h = k_h // 2
        p_w = k_w // 2

        x = np.pad(x, ((0, 0), (p_h, p_h), (p_w, p_w), (0, 0)), "constant", constant_values=0)

    N, H, W, C = x.shape
    col_img, out_H, out_W = img2col(x, k_h, k_w, s=s)
    z = np.dot(col_img, w.reshape(w.shape[0], -1).transpose())

    return z.reshape(N, out_H, out_W, -1)


x = np.zeros([32, 64, 64, 8])
w = np.ones([16, 3, 3, 8])

result = conv2d(x, w)
print(result.shape)