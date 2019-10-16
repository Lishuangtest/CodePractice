#参考网址：https://blog.csdn.net/qq_41661809/article/details/98180623
import os
import tensorflow as tf
num_classes=10


#设定用于训练和评估的样本总数
num_examples_pre_epoch_for_train=50000
num_examples_pre_epoch_for_eval=10000

#一个空类，该类用于返回读取的caif10数据集中的数据
class CIFAR10Record(object):
    pass

#接着 ,在文件中定义一个 read_cifar10() 函数, 用于读取文件队列中的数据.
#定义读取 Cifar-10 数据的函数
def read_cifar10(file_queue):
    # 首先创建一个CIFAR10Record类的实例对象, 属性height, weight, depth分别存储了一幅图像的高度, 宽度, 深度 。
    result=CIFAR10Record()

    label_bytes=1 #如果是 Cifar-100 数据集 ,此处为 2
    result.height=32
    result.width=32
    result.depth=3
    image_bytes=result.height*result.width*result.depth

    #每一个样本都包含一个 label 数据 和 image 数据
    record_bytes=label_bytes+image_bytes

    # FixedLengthRecordReader 类用于读取固定长度字节数信息
    reader=tf.FixedLengthRecordReader(record_bytes=record_bytes)
    # 调用该类的reader() 函数并向函数传入 文件队列就可以通过read来进行读取了。
    result.key,value=reader.read(file_queue)
    # record_bytes 可以将文件中的字符串解析成图像对应的像素数组
    record_bytes=tf.decode_raw(value,tf.uint8)
    # 将得到的 record_bytes 数组中的第一个元素类型转换成 int32 类型
    result.label=tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes]),tf.int32)
    # 剪切label 之后剩下的就是图片数据
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),[result.depth, result.height, result.width])
    # 通过 transpose() 将  [ depth ,height , width ]  转化成 [height ,width,depth]
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result

#然后是 inputs() 函数 ,这个函数传入的 data_dir 参数 就是存放原始CiFar-10 数据的目录。
def inputs(data_dir,batch_size,distorted):
    #通过 join() 函数拼接文件完整的路径,作为文件队列创建函数 train.string_input_producder() 的参数
    filenames=[os.path.join(data_dir,"data_batch_%d.bin"%i)for i in range(1,6)]
    # 创建一个文件队列,并调用 read_cifar10() 函数读取队列中的文件
    file_queue=tf.train.string_input_producer(filenames)
    #将队列传入上面写好的read_cifar10()函数中，得到result.其中result.uint8.image属性存储了原始的图像数据,而label则存储了对应的标签。
    read_input=read_cifar10(file_queue)
    # 为了方便图像数据处理对读取到的原始图像进行处理, 转换成float32格式 。
    reshaped_image=tf.cast(read_input.uint8image,tf.float32)
    num_examples_pre_epoch=num_examples_pre_epoch_for_train
    # distorted 参数来确定是否要对图像数据进行翻转,随机剪切,制造更多的样本
    # 对图像数据进行数据增强处理
    if distorted != None:
        # 将 [32,32,3] 大小的图片随机剪成 [24,24,3]
        cropped_image = tf.random_crop(reshaped_image,[24,24,3])
        # 随机左右翻转图片
        flipped_image = tf.image.random_flip_left_right(cropped_image)
        # 调整亮度
        adjusted_brightness = tf.image.random_brightness(flipped_image, max_delta=0.8)
        # 调整对比度
        adjusted_contrcast = tf.image.random_contrast(adjusted_brightness, lower=0.2, upper=1.8)
        # 标准化图片
        float_image = tf.image.per_image_standardization(adjusted_contrcast)
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])
        min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)
        print('Filling queue with %d CIFAR images before starting to train .' % min_queue_examples)
        print('This will take a few minutes.')

        # 使用 shuffle_batch() 函数随机产生一个 batch 的image 和label，即随机打乱图片的顺序
        images_train, labels_train = tf.train.shuffle_batch([float_image, read_input.label], batch_size=batch_size,
                                                            num_threads=16,
                                                            capacity=min_queue_examples + 3 * batch_size,
                                                            min_after_dequeue=min_queue_examples)
        return images_train, tf.reshape(labels_train, [batch_size])
    else:
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,24,24)
        # 直接标准化
        float_image = tf.image.per_image_standardization(resized_image)
        # 设置图片数据以及 label 的形状
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_pre_epoch * 0.4)
        images_test, labels_test = tf.train.batch([float_image, read_input.label],
                                                  batch_size=batch_size, num_threads=16,
                                                  capacity=min_queue_examples + 3 * batch_size)
        return images_test, tf.reshape(labels_test, [batch_size])