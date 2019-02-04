import pickle
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import cv2
import matplotlib.image as mpimg
import numpy as np
from tqdm import tqdm

input_images = []

for index in range(1, 5):
    file_name = "image_" + str(index) + ".jpg"
    input_images.append(mpimg.imread(file_name))

X = []
Y = []
for idx, image in enumerate(input_images):
    print("image:", idx)

    input_img = cv2.resize(image, (500, 500), interpolation=cv2.INTER_CUBIC)
    for i in range(198, 502, 2):
        Y.append(cv2.resize(input_img, (i, i), interpolation=cv2.INTER_CUBIC))
print(Y[0].shape, Y[-1].shape)

for j in range(len(Y)):
    h, w, _ = Y[j].shape
    X.append(cv2.resize(Y[j], (h + 2, w + 2)))
print(X[0].shape, X[-1].shape)

Y.pop(0)
X.pop()
Y.pop(151)
X.pop(151)
Y.pop(302)
X.pop(302)
Y.pop(453)
X.pop(453)


print("After pop")
print(X[0].shape, X[-1].shape,len(X))
print(Y[0].shape, Y[-1].shape,len(Y))

for i in range(len(X)):
    print(i,X[i].shape,Y[i].shape)


# for i in range(302):
#     h, w, n_c = input_img_1.shape
#     X.append(cv2.resize(input_img_1, (h + i, w + i), interpolation=cv2.INTER_CUBIC))
#
# for i in range(301):
#     he, we, _ = X[i].shape
#     Y.append(cv2.resize(X[i], (he + 1, we + 1), interpolation=cv2.INTER_CUBIC))

# X.pop(0)  # popping image with dim 43 coz itss only used to get blurred 44
# X.append(input_image)
# print(X[-1].shape)
diff_y=[Y[i]-X[i] for i in range(len(X))]


X = np.asarray(X)

print("shape of train input:", X.shape,X[0].shape)

Y = np.asarray(Y)
print("shape of train input:", Y.shape,Y[0].shape)

diff_y=np.asarray(diff_y)
print("shape of train input:", diff_y.shape,diff_y[0].shape)

num_epochs = 10
learning_rate = 0.0001

np.random.seed(1)




train_data = [(X[i], diff_y[i]) for i in range(len(X))]

shuffled_train_data = shuffle(train_data)


def init_weights(filter_size, ip_layer, op_layer):
    filter_height, filter_width = filter_size
    # a=np.random.randn(filter_height, filter_width, ip_layer, op_layer)
    # print("std:",np.std(a),np.sqrt(
    #     2 / (ip_layer*filter_width*filter_height)))
    # w = np.random.randn(filter_height, filter_width, ip_layer, op_layer) * np.sqrt(
    #     2 / (ip_layer*filter_width*filter_height))
    w = np.random.normal(loc=0.0, scale=0.1, size=(filter_height, filter_width, ip_layer, op_layer)) * np.sqrt(1 / 1000)
    return w


def init_bias(op_layer):
    return np.zeros((1, 1, 1, op_layer))


def MSE(A, B):
    mse = (np.square(np.subtract(A, B))).mean()
    return mse


def zero_pad(X, pad):
    X_pad = np.pad(X, ((pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
    return X_pad


def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev, W) + b
    Z = np.sum(s)
    return Z


def Relu(A):
    A[A <= 0] = 0
    return A


def conv_forward(A_prev, W, b, hparameters):
    (n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters['stride']
    pad = hparameters['pad']
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    Z = np.zeros((n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)
    for h in range(n_H):  # loop over vertical axis of the output volume
        for w in range(n_W):  # loop over horizontal axis of the output volume
            for c in range(n_C):  # loop over channels (= #filters) of the output volume
                # Find the corners of the current "slice" (≈4 lines)
                vert_start = h * stride
                vert_end = vert_start + f
                horiz_start = w * stride
                horiz_end = horiz_start + f
                # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                a_slice_prev = A_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                Z[h, w, c] = conv_single_step(a_slice_prev, W[..., c], b[..., c])

    assert (Z.shape == (n_H, n_W, n_C))

    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)

    return Z, cache


def conv_backward(dZ, cache):
    (A_prev, W, b, hparameters) = cache

    (n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    (f, f, n_C_prev, n_C) = W.shape

    stride = hparameters["stride"]
    pad = hparameters["pad"]

    (n_H, n_W, n_C) = dZ.shape

    dA_prev = np.zeros((n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for h in range(n_H):  # loop over vertical axis of the output volume
        for w in range(n_W):  # loop over horizontal axis of the output volume
            for c in range(n_C):  # loop over the channels of the output volume

                # Find the corners of the current "slice"
                vert_start = h * stride

                vert_end = vert_start + f
                horiz_start = w * stride

                horiz_end = horiz_start + f

                # Use the corners to define the slice from a_prev_pad
                a_slice = A_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                # Update gradients for the window and the filter's parameters using the code formulas given above
                dA_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[h, w, c]
                dW[:, :, :, c] += a_slice * dZ[h, w, c]
                db[:, :, :, c] += dZ[h, w, c]

    dA_prev[:, :, :] = dA_prev_pad[pad:-pad, pad:-pad, :]

    # Making sure your output shape is correct
    assert (dA_prev.shape == (n_H_prev, n_W_prev, n_C_prev))

    return dA_prev, dW, db


def CNN(image, y, params):
    w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6, w7, b7, w8, b8, w9, b9 = params
    hparam = {"pad": 1, "stride": 1}
    """Forward propagation"""
    print("forward propagation")
    conv1, cache1 = conv_forward(image, w1, b1, hparam)
    # conv1[conv1 <= 0] = 0  # pass through ReLU non-linearity
    rconv1 = Relu(conv1[:])

    conv2, cache2 = conv_forward(rconv1, w2, b2, hparam)
    # conv2[conv2 <= 0] = 0
    rconv2 = Relu(conv2[:])

    conv3, cache3 = conv_forward(rconv2, w3, b3, hparam)
    rconv3 = Relu(conv3[:])

    conv4, cache4 = conv_forward(rconv3, w4, b4, hparam)
    rconv4 = Relu(conv4[:])

    conv5, cache5 = conv_forward(rconv4, w5, b5, hparam)
    rconv5 = Relu(conv5[:])

    conv6, cache6 = conv_forward(rconv5, w6, b6, hparam)
    rconv6 = Relu(conv6[:])

    conv7, cache7 = conv_forward(rconv6, w7, b7, hparam)
    rconv7 = Relu(conv7[:])

    conv8, cache8 = conv_forward(rconv7, w8, b8, hparam)
    rconv8 = Relu(conv8[:])

    conv9, cache9 = conv_forward(rconv8, w9, b9, hparam)
    loss = MSE(conv9, y)
    """Backward propagation """
    print("back propogation")
    dconv9 = conv9 - y
    # dconv8, dw9, db9 = conv_backward(dconv9, (rconv8, w9, b9, hparam))
    dconv8, dw9, db9 = conv_backward(dconv9, cache9)
    dconv8[conv8 <= 0] = 0

    # dconv7, dw8, db8 = conv_backward(dconv8, (rconv7, w8, b8, hparam))
    dconv7, dw8, db8 = conv_backward(dconv8, cache8)
    dconv7[conv7 <= 0] = 0

    # dconv6, dw7, db7 = conv_backward(dconv7, (rconv6, w7, b7, hparam))
    dconv6, dw7, db7 = conv_backward(dconv7, cache7)
    dconv6[conv6 <= 0] = 0

    # dconv5, dw6, db6 = conv_backward(dconv6, (rconv5, w6, b6, hparam))
    dconv5, dw6, db6 = conv_backward(dconv6, cache6)
    dconv5[conv5 <= 0] = 0

    # dconv4, dw5, db5 = conv_backward(dconv5, (rconv4, w5, b5, hparam))
    dconv4, dw5, db5 = conv_backward(dconv5, cache5)
    dconv4[conv4 <= 0] = 0

    # dconv3, dw4, db4 = conv_backward(dconv4, (rconv3, w4, b4, hparam))
    dconv3, dw4, db4 = conv_backward(dconv4, cache4)
    dconv3[conv3 <= 0] = 0

    # dconv2, dw3, db3 = conv_backward(dconv3, (rconv2, w3, b3, hparam))
    dconv2, dw3, db3 = conv_backward(dconv3, cache3)
    dconv2[conv2 <= 0] = 0

    # dconv1, dw2, db2 = conv_backward(dconv2, (rconv1, w2, b2, hparam))
    dconv1, dw2, db2 = conv_backward(dconv2, cache2)
    dconv1[conv1 <= 0] = 0

    # dimage, dw1, db1 = conv_backward(dconv1, (image, w1, b1, hparam))
    dimage, dw1, db1 = conv_backward(dconv1, cache1)
    grads = [dw1, db1, dw2, db2, dw3, db3, dw4, db4, dw5, db5, dw6, db6, dw7, db7, dw8, db8, dw9, db9]
    return grads, loss


def train_and_update_cnn(inp, out, cost, params):
    w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6, w7, b7, w8, b8, w9, b9 = params
    # batch_size = len(batch)
    cost_ = 0

    dw1 = np.zeros(w1.shape)
    db1 = np.zeros(b1.shape)
    dw2 = np.zeros(w2.shape)
    db2 = np.zeros(b2.shape)
    dw3 = np.zeros(w3.shape)
    db3 = np.zeros(b3.shape)
    dw4 = np.zeros(w4.shape)
    db4 = np.zeros(b4.shape)
    dw5 = np.zeros(w5.shape)
    db5 = np.zeros(b5.shape)
    dw6 = np.zeros(w6.shape)
    db6 = np.zeros(b6.shape)
    dw7 = np.zeros(w7.shape)
    db7 = np.zeros(b7.shape)
    dw8 = np.zeros(w8.shape)
    db8 = np.zeros(b8.shape)
    dw9 = np.zeros(w9.shape)
    db9 = np.zeros(b9.shape)

    # for i in range(batch_size):
    x = inp
    y = out
    print(x.shape, y.shape)
    print("training image:",)
    grads, loss = CNN(x, y, params)
    print("loss for image:", loss)
    dw1, db1, dw2, db2, dw3, db3, dw4, db4, dw5, db5, dw6, db6, dw7, db7, dw8, db8, dw9, db9 = grads
    # dw1_, db1_, dw2_, db2_, dw3_, db3_, dw4_, db4_, dw5_, db5_, dw6_, db6_, dw7_, db7_, dw8_, db8_, dw9_, db9_ = grads
    # dw1 = dw1_
    # db1 = db1_
    # dw2 += dw2_
    # db2 += db2_
    # dw3 += dw3_
    # db3 += db3_
    # dw4 += dw4_
    # db4 += db4_
    # dw5 += dw5_
    # db5 += db5_
    # dw6 += dw6_
    # db6 += db6_
    # dw7 += dw7_
    # db7 += db7_
    # dw8 += dw8_
    # db8 += db8_
    # dw9 += dw9_
    # db9 += db9_
    cost_ = loss

    """update params"""
    w1 = w1 - (learning_rate * (dw1))
    b1 = b1 - (learning_rate * (db1))
    w2 = w2 - (learning_rate * (dw2))
    b2 = b2 - (learning_rate * (db2))
    w3 = w3 - (learning_rate * (dw3))
    b3 = b3 - (learning_rate * (db3))
    w4 = w4 - (learning_rate * (dw4))
    b4 = b4 - (learning_rate * (db4))
    w5 = w5 - (learning_rate * (dw5))
    b5 = b5 - (learning_rate * (db5))
    w6 = w6 - (learning_rate * (dw6))
    b6 = b6 - (learning_rate * (db6))
    w7 = w7 - (learning_rate * (dw7))
    b7 = b7 - (learning_rate * (db7))
    w8 = w8 - (learning_rate * (dw8))
    b8 = b8 - (learning_rate * (db8))
    w9 = w9 - (learning_rate * (dw9))
    b9 = b9 - (learning_rate * (db9))

    cost.append(cost_)
    params = [w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6, w7, b7, w8, b8, w9, b9]

    return params, cost


def train_cnn():
    w1 = init_weights((3, 3), 3, 64)
    b1 = init_bias(64)
    w2 = init_weights((3, 3), 64, 64)
    b2 = init_bias(64)
    w3 = init_weights((3, 3), 64, 64)
    b3 = init_bias(64)
    w4 = init_weights((3, 3), 64, 64)
    b4 = init_bias(64)
    w5 = init_weights((3, 3), 64, 64)
    b5 = init_bias(64)
    w6 = init_weights((3, 3), 64, 64)
    b6 = init_bias(64)
    w7 = init_weights((3, 3), 64, 64)
    b7 = init_bias(64)
    w8 = init_weights((3, 3), 64, 64)
    b8 = init_bias(64)
    w9 = init_weights((3, 3), 64, 3)
    b9 = init_bias(3)

    params = [w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6, w7, b7, w8, b8, w9, b9]
    cost = []

    for epoch in range(num_epochs):
        print("epoch:", epoch)
        # batches = [shuffled_train_data[m:m + 1] for m in range(0, len(shuffled_train_data), 1)]
        # batches = [X]
        # train_iter = tqdm(batches)
        for i, batch in enumerate(shuffled_train_data):
            params, cost = train_and_update_cnn(batch[0], batch[1], cost, params)
            print("loss at i:", i, cost[-1])
    with open('params.pkl', 'wb') as file:
        pickle.dump(params, file)

    return cost




print(train_cnn())

