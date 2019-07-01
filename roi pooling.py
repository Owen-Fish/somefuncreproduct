import numpy as np
import time 

def roipooling(x):
    # x[0]==>1,64,480,480 x[1]===>[200, 4]
    bottom_data = x[0]
    rois = x[1]

    num_batch = bottom_data.shape[0]
    num_channel = bottom_data.shape[0]
    num_rois = rois.shape[0]
    _height = bottom_data.shape[2]
    _width = bottom_data.shape[2]

    output =np.zeros((num_rois, num_channel, Pooled_height, Pooled_width))
    max_idx = np.zeros_like(output)

    start = time.time()
    for n in range(num_rois):
        roi_batch = rois[n]
        # 将原图坐标映射到feature map上
        roi_start_w = round(roi_batch[0] * Scale_ratio)
        roi_start_h = round(roi_batch[1] * Scale_ratio)
        roi_end_w = round(roi_batch[2] * Scale_ratio)
        roi_end_h = round(roi_batch[3] * Scale_ratio)

        roi_height = np.maximum(roi_end_h - roi_start_h + 1, 1)
        roi_width = np.maximum(roi_end_w - roi_start_w + 1, 1)

        bin_size_h = roi_height / Pooled_height
        bin_size_w = roi_width / Pooled_width

        for c in range(num_channel):  # calculate channel by channel
            for ph in range(Pooled_height):
                for pw in range(Pooled_width):
                    hstart = np.floor(ph * bin_size_h)
                    hend = np.ceil((ph + 1) * bin_size_h)
                    wstart = np.floor(pw * bin_size_w)
                    wend = np.ceil((pw + 1) * bin_size_w)

                    # 匹配到真正的像素位置坐标
                    hstart = np.minimum(np.maximum(hstart + roi_start_h, 0), _height)
                    hend = np.minimum(np.maximum(hend + roi_start_h, 0), _height)
                    wstart = np.minimum(np.maximum(wstart + roi_start_w, 0), _width)
                    wend = np.minimum(np.maximum(wend + roi_start_w, 0), _width)

                    # 如果roi 经过scale_ratio之后得到的ROI pool map 小于原先设定的7*7， 那么就直接忽略掉这个ROI ，这样会导致很多小的物体检测不到
                    is_empty = (hend <= hstart) or (wend <= wstart)
                    pool_idx = ph * Pooled_width + pw
                    if is_empty:
                        output[n][c].T.reshape(-1)[pool_idx] = 0
                        max_idx[n][c].T.reshape(-1)[pool_idx] = -1
                        continue

                    # 获取最大值的索引, 这一步是为了backward时候计算梯度。
                    max_cont = - np.inf
                    for h in range(np.int(hstart), np.int(hend)):
                        for w in range(np.int(wstart), np.int(wend)):
                            idx = h * _width + w
                            if bottom_data.T.reshape(-1)[idx] > max_cont:
                                max_cont = bottom_data.T.reshape(-1)[idx]
                    # import pdb
                    # pdb.set_trace()
                    max_idx[n][c][ph][pw] = idx
                    output[n][c][ph][pw] = max_cont
    end = time.time() - start
    return output, end


if __name__ == '__main__':
    np.random.seed(3)
    input = [np.random.rand(1, 64, 480, 480), np.array([0, 0, 224, 224]).reshape(1, 4)]
    output, time_used = roipooling(input)
    print(output.shape, time_used)
