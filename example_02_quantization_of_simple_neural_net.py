import numpy as np
import random

random.seed(42)
np.random.seed(42)


def init_weights1():
    w1 = np.random.uniform(-5, 5, (3, 3, 2, 2))
    bias1 = np.random.uniform(-10, 10, 2)
    return w1, bias1


def init_weights2():
    w2 = np.random.uniform(-6, 6, (3, 3, 2, 1))
    bias2 = np.random.uniform(-20, 20, 1)
    return w2, bias2


def print_weights(w1, bias1):
    for i in range(w1.shape[3]):
        for j in range(w1.shape[2]):
            print('Kernel output: {} input: {}'.format(i, j))
            for k in range(w1.shape[1]):
                print("{:.4f} {:.4f} {:.4f}".format(w1[k, 0, j, i], w1[k, 1, j, i], w1[k, 2, j, i]))
            print('')

    for i in range(bias1.shape[0]):
        print('Bias {}: {:.4f}'.format(i, bias1[i]))
    print('')


def get_calibration_dataset(min_value_input, max_value_input, num_images):
    arr = np.random.uniform(min_value_input, max_value_input, (num_images, 5, 5, 2))
    arr[arr > 0.999] = 1
    arr[arr < -0.999] = -1
    return arr


def run_neural_net_with_stat(input_arr, w0, bias0, w1, bias1):
    layer_stat = dict()
    layer_stat['layer0'] = input_arr.copy()
    layer_stat['layer1'] = []
    layer_stat['layer2'] = []
    layer_stat['layer3'] = []

    layer1 = np.zeros((input_arr.shape[0], 3, 3, 2))
    # layer1 - main convolution
    # Cycle by all batch images
    for i in range(input_arr.shape[0]):
        # Vertical cycle
        for j in range(1, input_arr.shape[1] - 1, 1):
            # Horizontal cycle
            for k in range(1, input_arr.shape[2] - 1, 1):
                # Input activations cycle
                for l in range(0, input_arr.shape[3], 1):
                    # Output filters cycle
                    for m in range(0, w0.shape[3], 1):
                        submatrix = input_arr[i, j-1:j+2, k-1:k+2, l]
                        weight = w0[:, :, l, m]
                        out_sh0 = j - 1
                        out_sh1 = k - 1
                        # print(submatrix.shape, weight.shape)
                        res = (submatrix * weight).sum()
                        layer1[i, out_sh0, out_sh1, m] += res

    # Add bias
    for m in range(0, bias0.shape[0], 1):
        layer1[:, :, :, m] += bias0[m]

    layer_stat['layer1'] = layer1.copy()

    # Layer 2 (RELU6)
    layer2 = layer1.copy()
    layer2[layer2 < 0] = 0
    layer2[layer2 > 6] = 6

    layer_stat['layer2'] = layer2.copy()

    # Last Conv2D layer
    input_images = layer2.copy()
    layer3 = np.zeros((input_images.shape[0], 1, 1, 1))
    # Cycle by all batch images
    for i in range(input_images.shape[0]):
        # Vertical cycle
        for j in range(1, input_images.shape[1] - 1, 1):
            # Horizontal cycle
            for k in range(1, input_images.shape[2] - 1, 1):
                # Input activations cycle
                for l in range(0, input_images.shape[3], 1):
                    # Output activations cycle
                    for m in range(0, w1.shape[3], 1):
                        submatrix = input_images[i, j - 1:j + 2, k - 1:k + 2, l]
                        weight = w1[:, :, l, m]
                        out_sh0 = j - 1
                        out_sh1 = k - 1
                        # print(submatrix.shape, weight.shape)
                        res = (submatrix * weight).sum()
                        layer3[i, out_sh0, out_sh1, m] += res

    # Add bias
    for m in range(0, bias1.shape[0], 1):
        layer3[:, :, :, m] += bias1[m]

    layer_stat['layer3'] = layer3.copy()
    return layer_stat


def quant_matrix(m, scale, bk):
    # Scale
    m_quant = np.round(m * scale).astype(np.int64)
    # Clip
    if bk is not None:
        m_quant[m_quant < -bk] = -bk
        m_quant[m_quant > bk] = bk
    return m_quant


def dequant_matrix(m_quant, scale):
    m = m_quant.astype(np.float32) / scale
    return m


def requant_matrix(m, scale_old, scale_new, bk):
    # Scale
    m_quant = np.round((m.astype(np.float32) * scale_new / scale_old)).astype(np.int64)
    # Clip
    if bk is not None:
        m_quant[m_quant < -bk] = -bk
        m_quant[m_quant > bk] = bk
    return m_quant


def compare_matrices(m_float, m_dequant):
    err = (m_float - m_dequant)
    err[m_float != 0] /= m_float[m_float != 0]
    err = np.abs(err)
    print('Matrix shape: {} Maximum error: {:.4f} % Avg error: {:.4f} %'.format(m_float.shape, err.max() * 100, err.mean() * 100))


if __name__ == '__main__':
    bit_precision = 8
    bk = 2 ** (bit_precision - 1) - 1
    print('Math bit precision: {}'.format(bit_precision))
    print('Use symmetric quantization. Range: [{}; {}]'.format(-bk, bk))

    w0, bias0 = init_weights1()
    w1, bias1 = init_weights2()

    print_weights(w0, bias0)
    print_weights(w1, bias1)

    # For this example we assume that all input images have values in the range from -1 to 1.
    # And for some networks like MobileNet, it actually like this.
    min_value_input = -1
    max_value_input = 1

    print('Weights 0. Min: {:.6f} Max: {:.6f}'.format(w0.min(), w0.max()))
    print('Weights 1. Min: {:.6f} Max: {:.6f}'.format(w1.min(), w1.max()))
    w0_max = np.abs(w0).max()
    w1_max = np.abs(w1).max()
    print('Max: {:.6f} {:.6f}'.format(w0_max, w1_max))
    weight_scale0 = bk / w0_max
    weight_scale1 = bk / w1_max
    print('Weight 0 scale: {:.6f}'.format(weight_scale0))
    print('Weight 1 scale: {:.6f}'.format(weight_scale1))
    print('')

    # Generate Calibration dataset randomly (1000 "images" in total).
    image_set = get_calibration_dataset(min_value_input, max_value_input, 1000)
    print('Calibration set shape: {}'.format(image_set.shape))

    # Run the Calibration dataset to collect statistics on the activations of the intermediate layers
    layer_stat = run_neural_net_with_stat(image_set, w0, bias0, w1, bias1)
    print('Activations 0. Min: {:.6f} Max: {:.6f}'.format(layer_stat['layer0'].min(), layer_stat['layer0'].max()))
    print('Activations 1. Min: {:.6f} Max: {:.6f}'.format(layer_stat['layer1'].min(), layer_stat['layer1'].max()))
    print('Activations 2. Min: {:.6f} Max: {:.6f}'.format(layer_stat['layer2'].min(), layer_stat['layer2'].max()))
    print('Activations 3. Min: {:.6f} Max: {:.6f}'.format(layer_stat['layer3'].min(), layer_stat['layer3'].max()))
    a0_max = np.abs(layer_stat['layer0']).max()
    a1_max = np.abs(layer_stat['layer1']).max()
    a2_max = np.abs(layer_stat['layer2']).max()
    a3_max = np.abs(layer_stat['layer3']).max()
    print('Max: {:.6f} {:.6f} {:.6f} {:.6f}'.format(a0_max, a1_max, a2_max, a3_max))
    act_scale0 = bk / a0_max
    act_scale1 = bk / a1_max
    act_scale2 = bk / a2_max
    act_scale3 = bk / a3_max
    print('Activations 0 scale: {:.6f}'.format(act_scale0))
    print('Activations 1 scale: {:.6f}'.format(act_scale1))
    print('Activations 2 scale: {:.6f}'.format(act_scale2))
    print('Activations 3 scale: {:.6f}'.format(act_scale3))

    # Get random single image
    input_image = get_calibration_dataset(min_value_input, max_value_input, 1)

    # Check first cell for correсtness
    # r1 = (input_image[0, :3, :3, 0] * w0[:, :, 0, 0]).sum() + (input_image[0, :3, :3, 1] * w0[:, :, 1, 0]).sum() + bias0[0]

    nn_results_float = run_neural_net_with_stat(input_image, w0, bias0, w1, bias1)
    print(nn_results_float['layer1'][0, :, :, 0])
    print(nn_results_float['layer1'][0, :, :, 1])

    # Scale weights (use int32 in example to avoid overflow)
    w0_quant = quant_matrix(w0, weight_scale0, bk)
    w1_quant = quant_matrix(w1, weight_scale1, bk)

    # Scale initial image
    input_image_quant = quant_matrix(input_image, act_scale0, bk)

    # print(input_image_quant[0, :, :, 0])
    # print(input_image_quant[0, :, :, 1])

    # We add bias to the product of weights and activations, so its Scale is equal to the product of scale for weights and activations!
    scale_bias0 = weight_scale0 * act_scale0
    scale_bias1 = weight_scale1 * act_scale2
    print('Bias 0 scale: {:.6f}'.format(scale_bias0))
    print('Bias 1 scale: {:.6f}'.format(scale_bias1))
    bias0_quant = quant_matrix(bias0, scale_bias0, None)
    bias1_quant = quant_matrix(bias1, scale_bias1, None)
    print(bias0_quant)
    print(bias1_quant)
    print('')

    print_weights(w0_quant, bias0_quant)
    print_weights(w1_quant, bias1_quant)

    # Calculation of the first layer in quantized form
    print('Go first CONV2D layer')
    layer1_quant = np.zeros((input_image_quant.shape[0], 3, 3, 2), dtype=np.int64)
    # layer1 - main convolution
    # Cycle by all batch images
    for i in range(input_image_quant.shape[0]):
        # Vertical cycle
        for j in range(1, input_image_quant.shape[1] - 1, 1):
            # Horizontal cycle
            for k in range(1, input_image_quant.shape[2] - 1, 1):
                # Input activations cycle
                for l in range(0, input_image_quant.shape[3], 1):
                    # Output activations cycle
                    for m in range(0, w0_quant.shape[3], 1):
                        submatrix = input_image_quant[i, j - 1:j + 2, k - 1:k + 2, l]
                        weight = w0_quant[:, :, l, m]
                        out_sh0 = j - 1
                        out_sh1 = k - 1
                        # print(submatrix.shape, weight.shape)
                        res = (submatrix * weight).sum()
                        layer1_quant[i, out_sh0, out_sh1, m] += res

    # Add bias
    for m in range(0, bias0_quant.shape[0], 1):
        layer1_quant[:, :, :, m] += bias0_quant[m]

    print(layer1_quant[0, :, :, 0])
    print(layer1_quant[0, :, :, 1])

    layer1_float = dequant_matrix(layer1_quant, weight_scale0 * act_scale0)
    compare_matrices(nn_results_float['layer1'], layer1_float)

    print('Current scale: {:.6f} Needed scale: {:.6f} Rescale coeff: {:.10f}'.format(weight_scale0 * act_scale0, act_scale1, act_scale1 / (weight_scale0 * act_scale0)))
    layer1_quant = requant_matrix(layer1_quant, weight_scale0 * act_scale0, act_scale1, bk)

    print(layer1_quant[0, :, :, 0])
    print(layer1_quant[0, :, :, 1])

    layer1_float = dequant_matrix(layer1_quant, act_scale1)
    compare_matrices(nn_results_float['layer1'], layer1_float)
    print('')

    # RELU layer
    print('Go RELU layer')
    print('Current scale: {:.6f} Needed scale: {:.6f} Rescale coeff: {:.10f}'.format(act_scale1, act_scale2, act_scale2 / act_scale1))
    layer2_quant = requant_matrix(layer1_quant, act_scale1, act_scale2, bk)
    # Now use RELU function
    layer2_quant[layer2_quant < 0] = 0
    print(layer2_quant[0, :, :, 0])
    print(layer2_quant[0, :, :, 1])

    # Checking
    layer2_float = dequant_matrix(layer2_quant, act_scale2)
    compare_matrices(nn_results_float['layer2'], layer2_float)
    print('')

    # Last CONV2D layer
    print('Go second CONV2D layer')
    input_images = layer2_quant.copy()
    layer3_quant = np.zeros((input_images.shape[0], 1, 1, 1))
    # Cycle by all batch images
    for i in range(input_images.shape[0]):
        # Vertical cycle
        for j in range(1, input_images.shape[1] - 1, 1):
            # Horizontal cycle
            for k in range(1, input_images.shape[2] - 1, 1):
                # Input activations cycle
                for l in range(0, input_images.shape[3], 1):
                    # Output activations cycle
                    for m in range(0, w1_quant.shape[3], 1):
                        submatrix = input_images[i, j - 1:j + 2, k - 1:k + 2, l]
                        weight = w1_quant[:, :, l, m]
                        out_sh0 = j - 1
                        out_sh1 = k - 1
                        # print(submatrix.shape, weight.shape)
                        res = (submatrix * weight).sum()
                        layer3_quant[i, out_sh0, out_sh1, m] += res

    # Add bias
    for m in range(0, bias1_quant.shape[0], 1):
        layer3_quant[:, :, :, m] += bias1_quant[m]

    # print(layer3_quant)

    # Dequantize as is (we don't need additional rescale here)
    cur_scale = weight_scale1 * act_scale2
    layer3_float = dequant_matrix(layer3_quant, cur_scale)
    compare_matrices(nn_results_float['layer3'], layer3_float)
    print('Real output: {:.6f} Returned output: {:.6f}'.format(nn_results_float['layer3'].flatten()[0], layer3_float.flatten()[0]))
