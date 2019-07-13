'''
手写CNN前向传播forward部分

forward需要实现的包括：
1。 conv操作
2。 pad操作
3。 pool操作

conv操作流程：
    args:
        当前层输入 [b, h, w, cin] （必须输入，无默认值）
        当前层输出feature map个数 cout （必须输入，无默认值）
        卷积核尺寸 [size, size] （必须输入，无默认值）
        卷积步长 s, s>=1, default=1
        pad方式 SAME, VALID, default="SAME"

    return:
        输出 [b, hout, wout, cout]


    step1: 初始化kernel
    step2: 对输入进行pad
    step3: 卷积
    step4: 输出

pad操作流程：
    args:
        当前层输入 [b, h, w, cin] （必须输入，无默认值）
        pad尺寸 kh kw, 前者为h需要pad的总数，后者为w需要pad的总数 （必须输入，无默认值）
        pad填充值, default=0

    return:
        输出 [b, h+kh, w+kw, cin]


max/avg pool操作流程：
    args:
        当前层输入 [b, h, w, cin] （必须输入，无默认值）
        pool尺寸 [k_h, k_w] （必须输入，无默认值）
        pool的stride [s_h, s_w]（必须输入，无默认值）
        pad方式 SAME, VALID, default="SAME"

    return:
        输出 [b, h_out, w_out, cin]
'''


import numpy as np

def pad2d(x :"np.array, shape=[b, h, w, c]",
          size :"np.array, [pad_h,pad_w]",
          pad_value:"int"=0)->np.array:

    # numpy中有pad函数，ndarray = numpy.pad(array, pad_width, mode, **kwargs)
    # array为要填补的数组
    # pad_width是在各维度的各个方向上想要填补的长度,如（（1，2），（2，2）），
    # 表示在第一个维度上现有元素之前padding=1,之后padding=2，比如从[1,2,3]变成[0,1,2,3,0,0],
    # 在第二个维度上现有元素之前padding=2,之后padding=2。
    # 如果直接输入一个整数，则说明各个维度和各个方向所填补的长度都一样。
    # mode为填补类型，即怎样去填补，有“constant”，“edge”等模式，如果为constant模式，就得指定填补的值，如果不指定，则默认填充0。
    # 剩下的都是一些可选参数，具体可查看 https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html

    # 通过numpy的pad函数进行pad，这边需要讨论的地方是pad是在现有元素之前还是之后
    # 这里采用的策略是，如果size是偶数，则将其平分，之前之后pad相同数目
    # 如果是奇数，则之后pad比之前多1

    assert size[0] >= 0, print("pad size[0] must > 0 but get {0}".format(size[0]))
    assert size[1] >= 0, print("pad size[1] must > 0 but get {0}".format(size[1]))

    pad_h_top = int(np.floor(size[0]/2))
    pad_h_bottom = int(np.ceil(size[0]/2))

    pad_w_left = int(np.floor(size[1]/2))
    pad_w_right = int(np.ceil(size[1]/2))

    new_x = np.pad(x, ((0,0), (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right), (0,0)), mode="constant", constant_values=(pad_value,pad_value))

    return new_x


def conv2d(x :"np.array, [b,h,w,c]]",
           out_num :"int, 16",
           kernel_size :"np.array, [3,3]",
           strides :"np.array, [1,1]]" = [1,1],
           pad :"string, SAME/VALID"="SAME")->np.array:

    assert kernel_size[0] % 2 != 0, print("kernel size must be odd but get {0}".format(kernel_size[0]))
    assert kernel_size[1] % 2 != 0, print("kernel size must be odd but get {0}".format(kernel_size[1]))

    batch_size, input_h, input_w, input_c = np.shape(x)
    k_h, k_w = kernel_size
    s_h, s_w = strides

    # step1: 初始化kernel
    # 这里是用随机正态分布，实际需要更复杂的实现
    kernel = np.random.randn(out_num, kernel_size[0], kernel_size[1], input_c)


    # step2: 提前对输入图像进行padding，这样在conv时只需关注conv操作
    if pad == "SAME":
        # 对于SAME pad， 输出feature map的new_size为 ceil(w/s)，与filter size无关，
        # 所以此时需要将input的shape变化为new_size*s+(k-1)，这样在conv时可以直接得到new_size的输出

        # new_h, new_w分别为输出same padding下输出的尺寸
        # pad_h, pad_w分别为h和w方向需要pad的元素大小
        new_h = int(np.ceil(input_h/s_h))
        new_w = int(np.ceil(input_w/s_w))


        pad_h = np.ceil(input_h/s_h)*s_h+(k_h-1) - input_h
        pad_w = np.ceil(input_w/s_w)*s_w+(k_w-1) - input_w

        # 调用pad2d接口对x进行pad，default=zero pad
        x = pad2d(x, [pad_h, pad_w], pad_value=0)

    elif pad == "VALID":
        # 对于VALID pad， 输出feature map的new_size为 ceil((w-f+1)/s),
        # 具体实现时，可以只保留原本x的0 - new_size*s+(k-1)部分的元素，
        # 这样在conv时可以直接得到new_size的输出
        new_h = int(np.ceil( (input_h-k_h+1)/s_h ))
        new_w = int(np.ceil( (input_w-k_w+1)/s_w ))

        # 保留new次conv所需的元素以及最后一次conv需要的元素构成新的x，其余的抛弃
        x = x[:, 0:new_h*s_h+k_h-1, 0:new_w*s_w+k_w-1, :]

    else:
        # 暂未实现其他pad方式
        raise ValueError


    # 创建一个容器用于保存conv结果
    new_x = np.zeros([batch_size, new_h, new_w, out_num])


    # step3: conv
    # 输入为c个channel，输出为k个channel，
    # conv有两种实现方式，

    #   方式1：
    #   1. 对batch size = b 个图像
    #   2. 先用c个2-d kernel对c个channnel中的每个2-d channel做卷积，
    #   这里需要c次2-d卷积，得到c个feature map，再通过element-wise相加合并为1个feature map
    #   3. 再用不同的kernel重复上一步k次，得到k个feature map
    #   故一共需要b * k * c次2-d卷积

    #   方式2：
    #   1. 对batch size = b 个图像
    #   2. 将c个kernel看作一个3-d kernel， 直接对c个channel卷积，得到1个输出feature map
    #   3. 用不同的kernel重复上一步k次，得到k个feature map
    #   故一共需要b*k次3-d卷积

    # 这里采用第二种实现
    # 其中每次卷积都是用一个[k_h,k_w,c]的3维核对这c个channel组成的3维tensor同时处理得到一个[h,w]的2维map，
    # 最后合成一个[h,w,k]的3维tensor


    for i in range(batch_size):
        # 每个cur_kernel[k]都是一个[k_h,k_w,c]的3d kernel
        # 每个cur_img都是一个[h, w, c]的3d map
        # 每个cur_img_feature_map就是这个kernel和这个map卷积得到的[h,w,1]的map
        # 步骤为：
        # 先取出一个cur_img,分别用k个kernel对其卷积得到k个[h,w,1]的map，
        # 再将得到的k个map在最后一个维度进行concat，得到该img的[h,w,k]的输出
        # 一共batch_size个图片，
        # 所以最后得到的就是[batch_size, h, w, k]的输出
        cur_img = x[i]

        cur_img_feature_map = []
        for k in range(out_num):

            cur_kernel = kernel[k]
            conv_map = conv_(cur_img, cur_kernel, strides)
            cur_img_feature_map.append(conv_map)

        cur_img_feature_map = np.concatenate(cur_img_feature_map, axis=-1)
        new_x[i] = cur_img_feature_map

    return new_x

def conv_(img :"np.array, shape=[h,w,c]]",
          kernel :"np.array, shape=[k_h, k_w, c]]",
          strides :"np.array, [s_h,s_w]")->np.array:

    # 输入的img是经过pad操作后的结果
    img_h, img_w, img_c = np.shape(img)
    k_h, k_w, k_c = np.shape(kernel)
    s_h, s_w = strides

    # w_time,h_time分别是该img要进行conv的次数，
    # 同时也是生成map的两个维度上的元素个数(1次conv对应map上一个元素)
    w_time = int(np.ceil( (img_w-(k_w-1))/s_w ))
    h_time = int(np.ceil( (img_h-(k_h-1))/s_h ))

    # 创建一个容器保存conv结果map
    new_feature_map = np.zeros([h_time, w_time, 1])

    for c in range(0, h_time):
        for r in range(0, w_time):

            # 每次conv操作为2个[h,w,c]矩阵点乘，输出一个标量
            cur_region = img[c*s_h:c*s_h+k_h, r*s_w:r*s_w+k_w, :]
            conv_result = np.sum(kernel*cur_region)
            new_feature_map[c][r][0] = conv_result

    return new_feature_map

def max_pool_2d(x :"np.array, shape=[batch, h, w, c]",
                kernel :"np.array, [k_h, k_w]",
                strides :"np.array, [s_h, s_w]]",
                pad :"string, SAME/VALID"="SAME")->np.array:

    # step1: 将图片pad,
    # 如果是SAME pad，输出的feature map尺寸为ceil(input / s)，需要pad的尺寸就是前后之差；
    # 如果是VALID pad，输出feature map的new_size为 ceil((w-f+1)/s),因此只保留0 - new_size+(k-1)

    batch_size, input_h, input_w, input_c = np.shape(x)
    k_h, k_w = kernel
    s_h,s_w = strides

    if pad=="SAME":
        new_h = int(np.ceil(input_h/s_h))
        new_w = int(np.ceil(input_w/s_w))
        pad_h = np.ceil(input_h/s_h)*s_h+(k_h-1) - input_h
        pad_w = np.ceil(input_w/s_w)*s_w+(k_w-1) - input_w
        x = pad2d(x, [pad_h, pad_w], pad_value=0)

    elif pad=="VALID":
        new_h = int(np.ceil( (input_h-k_h+1)/s_h ))
        new_w = int(np.ceil( (input_w-k_w+1)/s_w ))
        x = x[:, 0:new_h*s_h+k_h-1, 0:new_w*s_w+k_w-1, :]
    else:
        raise ValueError

    batch_size, padded_h, padded_w, input_c = np.shape(x)

    # step2: 将输入分解为batch size张图像，每张图像shape=[h,w,c]，分解为c个map分别求max pool
    # 故总计要对b*c个map求max pool， 先创建一个new_x用于保存结果

    # w_time = int( np.floor((padded_w - (k_w - 1)) / s_w))
    # h_time = int( np.floor((padded_h - (k_h - 1)) / s_h))
    w_time = new_w
    h_time = new_h

    new_x = np.zeros([batch_size, new_h, new_w, input_c])

    for i in range(batch_size):
        cur_img = x[i]

        for m in range(input_c):
            cur_map = cur_img[:,:,m]

            for h in range(0, h_time):
                for w in range(0, w_time):
                    cur_region = cur_map[h*s_h:h*s_h+k_h, w*s_w:w*s_w+k_w]
                    new_x[i,h,w,m] = np.max(cur_region)

    return new_x

def avg_pool_2d(x :"np.array, shape=[batch, h, w, c]",
                kernel :"np.array, [k_h, k_w]",
                strides :"np.array, [s_h, s_w]]",
                pad :"string, SAME/VALID"="SAME")->np.array:

    # step1: 将图片pad,
    # 如果是SAME pad，输出的feature map尺寸为ceil(input / s)，需要pad的尺寸就是前后之差；
    # 如果是VALID pad，输出feature map的new_size为 ceil((w-f+1)/s),因此只保留0 - new_size+(k-1)

    batch_size, input_h, input_w, input_c = np.shape(x)
    k_h, k_w = kernel
    s_h,s_w = strides

    if pad=="SAME":
        new_h = int(np.ceil(input_h/s_h))
        new_w = int(np.ceil(input_w/s_w))
        pad_h = np.ceil(input_h/s_h)*s_h+(k_h-1) - input_h
        pad_w = np.ceil(input_w/s_w)*s_w+(k_w-1) - input_w
        x = pad2d(x, [pad_h, pad_w], pad_value=0)

    elif pad=="VALID":
        new_h = int(np.ceil( (input_h-k_h+1)/s_h ))
        new_w = int(np.ceil( (input_w-k_w+1)/s_w ))
        x = x[:, 0:new_h*s_h+k_h-1, 0:new_w*s_w+k_w-1, :]
    else:
        raise ValueError

    batch_size, padded_h, padded_w, input_c = np.shape(x)

    # step2: 将输入分解为batch size张图像，每张图像shape=[h,w,c]，分解为c个map分别求max pool
    # 故总计要对b*c个map求max pool， 先创建一个new_x用于保存结果

    # w_time = int( np.floor((padded_w - (k_w - 1)) / s_w))
    # h_time = int( np.floor((padded_h - (k_h - 1)) / s_h))
    w_time = new_w
    h_time = new_h

    new_x = np.zeros([batch_size, new_h, new_w, input_c])

    for i in range(batch_size):
        cur_img = x[i]

        for m in range(input_c):
            cur_map = cur_img[:,:,m]

            for h in range(0, h_time):
                for w in range(0, w_time):
                    cur_region = cur_map[h*s_h:h*s_h+k_h, w*s_w:w*s_w+k_w]
                    new_x[i,h,w,m] = np.mean(cur_region)

    return new_x


def relu(x):

    x[x<0] = 0

    return x


def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):

    return np.exp(x) / np.sum(np.exp(x))






if __name__ == "__main__":

    import skimage.data
    from skimage import io
    import matplotlib.pyplot as plt

    img = skimage.data.chelsea()
    # img = skimage.color.rgb2gray(img)
    # io.imshow(img)
    # plt.show()

    img = np.reshape(img, [1, 300, 451, 3])

    print("shape before conv",img.shape)
    img = conv2d(img, 3, [5, 5], strides=[2, 2], pad="VALID")
    print("shape after conv", img.shape)

    img = avg_pool_2d(img, [2, 2],[2, 2], pad="SAME")
    print("shape after max pool", img.shape)

    #img = np.reshape(img, [149, 224])
    io.imshow(img[0,:,:,2])
    plt.show()




