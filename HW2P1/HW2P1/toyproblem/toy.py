import sys
import numpy as np

sys.path.append('../mytorch')
from activation import *
from loss import *
from linear import *
from batchnorm import *
from conv import *
from value import *


"""
Forward:
Input -> Layer1 -> ReLU -> Layer2 -> Criterion(L2) -> Target (derivative loss is given)
Backward:
Loss -> Layer2 -> ReLU -> Layer1 -> Input
"""

def conv_test(conv, input, w1, b1, y1lin, y1, w2, b2, y2, deriv_loss,
              grady1, dw2, db2, grady1lin, gradinput, dw1, db1):
    """
    [GIVEN] -> values setted in value.py      [TEST] -> values need tests by 'assert'
    conv: the convolution class you will test (Conv1D(), Conv2D())
    input[GIVEN]: input for the network (INPUT_1D, INPUT_1D_B, INPUT_2D_B)
    w1[GIVEN]: the weights of layer1 (W1_1D, W1_1D_B, W1_2D_B) 
    b1[GIVEN]: the bias of layer1 (B1_1D, B1_1D_B, B1_2D_B)
    y1lin[TEST]: the output of layer1 (OUTPUT_L1lin_1D, OUTPUT_L1lin_1D_B, OUTPUT_L1lin_2D_B)  
    y1[TEST]: the activated output of y1lin
    w2[GIVEN]: the weights of layer2 (W2_1D, W2_1D_B, W2_2D_B) 
    b2[GIVEN]: the bias of layer2 (B2_1D, B2_1D_B, B2_2D_B)
    y2[TEST]: the output of layer2 (OUTPUT_L2lin_1D, OUTPUT_L2lin_1D_B, OUTPUT_L12in_2D_B)
    deriv_loss[TEST]: the derivative of loss over y2 (DERIV_LOSS_1D, DERIV_LOSS_1D_B, DERIV_LOSS_2D_B)
    grady1[TEST]: the derivative of loss over y1 (GRAD_Y1_1D, GRAD_Y1_1D_B, GRAD_Y1_2D_B)
    dw2[TEST]: the derivative of loss over w2 (dW2_1D, dW2_1D_B, dW2_2D_B)
    db2[TEST]: the derivative of loss over b2 (dB2_1D, dB2_1D_B, dB2_2D_B)
    grady1lin[TEST]: the derivative of loss over y1lin (GRAD_Y1lin_2D, GRAD_Y1lin_1D_B, GRAD_Y1lin_2D_B)
    gradyinput[TEST]: the derivative of loss over input (GRAD_INPUT_2D, GRAD_INPUT_1D_B, GRAD_INPUT_2D_B)
    dw1[TEST]: the derivative of loss over w1 (dW1_1D, dW1_1D_B, dW1_2D_B)
    db1[TEST]: the derivative of loss over b1 (dB1_1D, dB1_1D_B, dB1_2D_B)
    """
    # Layer1
    conv_l1 = conv(in_channel=3, out_channel=2, kernel_size=2, stride=1)
    conv_l1.W = w1
    conv_l1.b = b1
    Y1lin = conv_l1.forward(input)
    assert((Y1lin == y1lin).all())

    # Actiation
    relu1 = ReLU()
    Y1 = relu1.forward(Y1lin)
    assert((Y1 == y1).all())

    # Layer2
    conv_l2 = conv(in_channel=2, out_channel=1, kernel_size=2, stride=2)
    conv_l2.W = w2
    conv_l2.b = b2
    Y2 = conv_l2.forward(Y1)
    assert((Y2 == y2).all())

    # BACKWARD
    # derivative loss is given
    grad_y2 = deriv_loss
    assert((grad_y2 == deriv_loss).all())

    grad_Y1 = conv_l2.backward(grad_y2)
    dW2 = conv_l2.dW
    dB2 = conv_l2.db
    assert((grad_Y1 == grady1).all())
    assert((dW2 == dw2).all())
    assert((dB2 == db2).all())

    grad_Y1lin = grad_Y1 * relu1.derivative()
    assert((grad_Y1lin == grady1lin).all())

    grad_input = conv_l1.backward(grad_Y1lin)
    dW1 = conv_l1.dW
    dB1 = conv_l1.db
    assert((grad_input == gradinput).all())
    assert((dW1 == dw1).all())
    assert((dB1 == db1).all())


# TODO: TEST CONV1D WITHOUT BATCH
conv_test(Conv1D, INPUT_1D, W1_1D, B1_1D, OUTPUT_L1lin_1D, OUTPUT_L1_1D, W2_1D, B2_1D, OUTPUT_L2lin_1D,
          DERIV_LOSS_1D, GRAD_Y1_1D, dW2_1D, dB2_1D, GRAD_Y1lin_1D, GRAD_INPUT_1D, dW1_1D, dB1_1D)
print("Congratulations! You pass the conv1d (without batch) test!")

# TODO: TEST CONV1D WITH BATCH
conv_test(Conv1D, INPUT_1D_B, W1_1D_B, B1_1D_B, OUTPUT_L1lin_1D_B, OUTPUT_L1_1D_B, W2_1D_B, B2_1D_B, OUTPUT_L2lin_1D_B,
          DERIV_LOSS_1D_B, GRAD_Y1_1D_B, dW2_1D_B, dB2_1D_B, GRAD_Y1lin_1D_B, GRAD_INPUT_1D_B, dW1_1D_B, dB1_1D_B)
print("Congratulations! You pass the conv1d (with batch) test!")

# TODO: TEST CONV2D WITH BATCH
conv_test(Conv2D, INPUT_2D_B, W1_2D_B, B1_2D_B, OUTPUT_L1lin_2D_B, OUTPUT_L1_2D_B, W2_2D_B, B2_2D_B, OUTPUT_L2lin_2D_B,
          DERIV_LOSS_2D_B, GRAD_Y1_2D_B, dW2_2D_B, dB2_2D_B, GRAD_Y1lin_2D_B, GRAD_INPUT_2D_B, dW1_2D_B, dB1_2D_B)
print("Congratulations! You pass the conv2d (with batch) test!")

"""
Test dilation and Padding, for convenience, feed forward through only one layers. Take a fake loss for backpropagation
Forward:
Input -> Layer1 -> Target
Backward:
Loss -> Layer1 -> Input
"""

# TODO: TEST CONV2D WITH DILATION
conv = Conv2D_dilation(in_channel=3, out_channel=1, kernel_size=2, stride=2, padding=1, dilation=2)
conv.W = W_DILATION
conv.b = B_DILATION
Y1 = conv.forward(INPUT_DILATION)
assert((Y1.round() == Y1_DILATION.round()).all())

grad_input = conv.backward(DERIV_DILATION)
dw = conv.dW
db = conv.db
assert((grad_input.round() == GRAD_INPUT.round()).all())
assert((dw.round() == dW_DILATION.round()).all())
assert((db.round() == dB_DILATION.round()).all())
print("Congratulations! You pass the conv2d (with dilation) test!")

