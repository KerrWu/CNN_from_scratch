import numpy as np
default_type = np.float32


# 简易版本，不考虑pad, kernel size为正方形，实现一个简易的valid的前向传播,全程用numpy实现
# 格式为[N, H, W, C], 卷积核的初始化在卷积操作内部实现，直接用标准正态分布初始化，只返回卷积结果
def conv_2d(inputs, out_c, k_size=3, stride=1, use_bias=True):

    assert len(inputs.shape) == 4, "inputs must be a 4-d tensor"
    batch_size, in_h, in_w, in_c = inputs.shape

    # step1: 初始化卷积核,shape=[out_c, k, k, in_c]
    kernel = np.random.randn(out_c, k_size, k_size, in_c).astype(default_type)
    if use_bias:
        bias = np.random.uniform(0,1, size=(out_c))


    # step2：用滑动窗口法实现卷积操作，注意到kernel和inputs的后三维是一样的，因此直接用滑动矩形

    out_h, out_w = (in_h-(k_size-1))//stride, (in_w-(k_size-1))//stride
    results = np.zeros(shape=(batch_size, out_h, out_w, out_c), dtype=default_type)

    for sample in range(batch_size):

        for h in range(out_h):
            for w in range(out_w):
                cur_value = inputs[sample,h*stride:h*stride+k_size,w*stride:w*stride+k_size,:] * kernel[sample]
                cur_value = np.sum(cur_value,axis=(0,1), dtype=default_type)
                if use_bias:
                    cur_value+=bias
                results[sample][h][w] = cur_value

    return results

if __name__ == "__main__":

    import skimage.data
    from skimage import io
    import matplotlib.pyplot as plt

    img = skimage.data.chelsea()
    # img = skimage.color.rgb2gray(img)
    # io.imshow(img)
    # plt.show()

    img = np.reshape(img, [1, 300, 451, 3])/255.0

    print("shape before conv",img.shape)
    img = conv_2d(img, 3, 10, stride=5)*255.0
    io.imshow(img[0, :, :, 2])
    plt.show()











