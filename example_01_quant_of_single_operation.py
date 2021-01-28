# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), IPPM RAS'

import numpy as np


if __name__ == '__main__':
    bit_size = 8

    # Initial floating point data. We need to calculate value of W*A + B
    # W - weights, A - activations, B - bias

    weights = np.array([-23.543, 8.533, 1.012, -12.4756, 34.53])
    activations = np.array([0.32432, 1.123213, 0.234324, 5.234234, 4.234323])
    bias = np.array(33.1)
    res = (weights*activations).sum() + bias
    print('Floating point result: {}'.format(res))
    # Floating point result: 116.19701015660002

    # We can find minimum and maximum value for weights directly because it fixed in neural net after training
    # We use symmetric quantization, so choose scale on which we multiply weights
    # Zero is not moving in symmetric quantization
    # Yq = round(scale * clip(Y, -a, a)) - 'scale' - scaling factor, 'a' - clip threshold
    # s = (2 ** (k-1) - 1) / a - k - количество бит для представления числа
    k = bit_size
    a = np.abs(weights).max()
    scale_weights = (2 ** (k-1) - 1) / a
    print('Weights data: k = {} a = {:.6f} s = {:.6f}'.format(k, a, scale_weights))

    # To find maximum/minimum values for activation we need to run set of training data through neural net to find possible activation values
    # Let's assume we already run some data and find minimum is 0 and maximum is 6.
    k = bit_size
    a = np.abs([0, 6]).max()
    scale_activations = (2 ** (k-1) - 1) / a
    print('Activations data: k = {} a = {:.6f} s = {:.6f}'.format(k, a, scale_activations))
    print('Maximum value in quant: {}'.format((2 ** (k-1)) - 1))

    # For bias we need to use scale equal to (scale weights) * (scale activations) because we sum it
    # with result of multiplication.
    scale_bias = scale_weights * scale_activations

    # Now we ready to get quantized values. We can't set them int8 because operations like multiplication will overflow
    # so we use 32-bit accumulators
    weights_quant = np.round(weights*scale_weights).astype(np.int32)
    bias_quant = np.round(bias * scale_bias).astype(np.int32)
    activations_quant = np.round(activations * scale_activations).astype(np.int32)
    # In general case we can't gurantee that acivation not overflow. sometimes it can happens,
    # so we need to clip them to [-127, 127] interval
    activations_quant = np.clip(activations_quant, -127, 127)

    # Let's print them now
    print('Quant weights: {}'.format(weights_quant))
    print('Quant bias: {}'.format(bias_quant))
    print('Quant activations: {}'.format(activations_quant))

    # After perform the multiplication the new_scale = scale1 * scale2
    new_scale = scale_weights * scale_activations
    print('New scale after multiplication: {:.6f}'.format(new_scale))

    # Now we can calculate the final result
    res_quant = (weights_quant * activations_quant).sum() + bias_quant
    print('Quantized result: {}'.format(res_quant))

    # Next step shows how can we make a requantization. In current example it's not really needed
    # But it's needed if you plan to use result of previous operation on next layer

    # We know maximum values for activations on next operation because we run calibration dataset.
    # Let's assume we have coefficient a = 150. Then we need to requntize "res_quant" variable using it
    a = 150.0
    scale_layer2 = (2 ** (k-1) - 1) / a
    print('Result scale data a = {:.6f} scale = {:.6f}'.format(a, scale_layer2))
    requant_coeff = scale_layer2 / new_scale
    print('Requantization coefficient: {:.6f}'.format(requant_coeff))
    res_quant_rescaled = requant_coeff * res_quant.astype(np.float32)
    # Again we can't qurantee that activations are fit in -127, 127 interval, so we clip them
    res_quant_rescaled = np.clip(res_quant_rescaled, -127, 127)
    res_quant_rescaled = res_quant_rescaled.astype(np.int8)
    print('Result rescaled for layer2: {:.6f}'.format(res_quant_rescaled))

    # Now we get final quantized result at the end of calculations.
    # We can dequantize it back to floating point using current scale
    res_dequant = res_quant_rescaled.astype(np.float32) / scale_layer2
    print('Dequant result: {:.6f} Real result: {:.6f} Error: {:.6f}%'.format(res_dequant, res, 100 * np.abs(res - res_dequant) / res))