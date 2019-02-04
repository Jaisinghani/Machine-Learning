from layers import Convolution
from relu import ReLU
import matplotlib.image as mpimg
import cv2
import numpy as np
from cnn import CNN
import pickle
from sklearn.utils import shuffle

downscale_const = 500


class train:
    def train_cnn(self, image_name_list, epochs=1):
        self.train_data = []

        self.train_data.extend(self.get_train_data(image_name_list))
        shuffled_train_data = shuffle(self.train_data)
        _, _, ip_channels = shuffled_train_data[0][0].shape
        out_channels = 64
        filter_size = 3
        padding = 1
        lr = 0.0001

        layer_1 = Convolution(ip_channels, out_channels, filter_size, padding)
        relu_1 = ReLU()
        layer_2 = Convolution(out_channels, out_channels, filter_size, padding)
        relu_2 = ReLU()
        layer_3 = Convolution(out_channels, out_channels, filter_size, padding)
        relu_3 = ReLU()
        layer_4 = Convolution(out_channels, out_channels, filter_size, padding)
        relu_4 = ReLU()
        layer_5 = Convolution(out_channels, out_channels, filter_size, padding)
        relu_5 = ReLU()
        layer_6 = Convolution(out_channels, out_channels, filter_size, padding)
        relu_6 = ReLU()
        layer_7 = Convolution(out_channels, out_channels, filter_size, padding)
        relu_7 = ReLU()
        layer_8 = Convolution(out_channels, out_channels, filter_size, padding)
        relu_8 = ReLU()
        layer_9 = Convolution(out_channels, ip_channels, filter_size, padding)

        layers = [(layer_1, relu_1), (layer_2, relu_2), (layer_3, relu_3), (layer_4, relu_4), (layer_5, relu_5),
                  (layer_6, relu_6),
                  (layer_7, relu_7), (layer_8, relu_8), (layer_9, None)]
        cnn_obj = CNN(layers)
        for epoch in range(epochs):
            for x, y in self.train_data:
                cnn_obj = CNN(layers)
                loss, grads = cnn_obj.train_step(x, y)
                print("loss:", loss)
                params1, params2, params3, params4, params5, params6, params7, params8, params9 = grads
                dw1, db1 = params1
                dw2, db2 = params2
                dw3, db3 = params3
                dw4, db4 = params4
                dw5, db5 = params5
                dw6, db6 = params6
                dw7, db7 = params7
                dw8, db8 = params8
                dw9, db9 = params9
                layer_1.weights = layer_1.weights - (lr * dw1)
                layer_1.bias = layer_1.bias - (lr * db1)
                layer_2.weights = layer_2.weights - (lr * dw2)
                layer_2.bias = layer_2.bias - (lr * db2)
                layer_3.weights = layer_3.weights - (lr * dw3)
                layer_3.bias = layer_3.bias - (lr * db3)
                layer_4.weights = layer_4.weights - (lr * dw4)
                layer_4.bias = layer_4.bias - (lr * db4)
                layer_5.weights = layer_5.weights - (lr * dw5)
                layer_5.bias = layer_5.bias - (lr * db5)
                layer_6.weights = layer_6.weights - (lr * dw6)
                layer_6.bias = layer_6.bias - (lr * db6)
                layer_7.weights = layer_7.weights - (lr * dw7)
                layer_7.bias = layer_7.bias - (lr * db7)
                layer_8.weights = layer_8.weights - (lr * dw8)
                layer_8.bias = layer_8.bias - (lr * db8)
                layer_9.weights = layer_9.weights - (lr * dw9)
                layer_9.bias = layer_9.bias - (lr * db9)
                layers = [(layer_1, relu_1), (layer_2, relu_2), (layer_3, relu_3), (layer_4, relu_4), (layer_5, relu_5),
                          (layer_6, relu_6),
                          (layer_7, relu_7), (layer_8, relu_8), (layer_9, None)]

        return cnn_obj.params, layer_9.A_curr

    def get_train_data(self, train_images):
        input_images = []
        X = []
        Y = []
        if len(train_images) == 1:
            input_img = mpimg.imread(train_images[0])
            h, w, n_c = input_img.shape
            input_image = cv2.resize(input_img, (h, w), interpolation=cv2.INTER_CUBIC)
            for i in range(8):
                Y.append(cv2.resize(input_image, (h - i, w - i), interpolation=cv2.INTER_CUBIC))
            print(Y[0].shape, Y[-1].shape)

            for j in range(1, len(Y)):
                h, w, _ = Y[j].shape
                X.append(cv2.resize(Y[j], (h + 1, w + 1)))
            print(X[0].shape, X[-1].shape)

            Y.pop()

            print("After pop")
            print(X[0].shape, X[-1].shape, len(X))
            print(Y[0].shape, Y[-1].shape, len(Y))
        else:
            for index in range(len(train_images)):
                input_images.append(mpimg.imread(train_images[index]))

            for idx, image in enumerate(input_images):
                print("image:", idx)

                input_img = cv2.resize(image, (downscale_const, downscale_const), interpolation=cv2.INTER_CUBIC)
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
            print(X[0].shape, X[-1].shape, len(X))
            print(Y[0].shape, Y[-1].shape, len(Y))

            # for i in range(len(X)):
            #     print(i, X[i].shape, Y[i].shape)

        diff_y = [Y[i] - X[i] for i in range(len(X))]

        X = np.asarray(X)

        print("shape of train input:", X.shape, X[0].shape)

        Y = np.asarray(Y)
        print("shape of train input:", Y.shape, Y[0].shape)

        diff_y = np.asarray(diff_y)
        print("shape of train input:", diff_y.shape, diff_y[0].shape)
        train_data = [(X[i], diff_y[i]) for i in range(len(X))]

        shuffled_train_data = shuffle(train_data)
        return shuffled_train_data


def main(train_image, output_resolution):
    train_obj = train()
    type = train_image.rsplit(".")[1]
    if train_image is None or output_resolution is None:
        image_names_list = ['image_1.jpg', 'image_2.jpg', 'image_3.jpg', 'image_4.jpg']
        params, img = train_obj.train_cnn(image_names_list)
    else:
        params, img = train_obj.train_cnn([train_image])
        h, w, n_c = img.shape
        while h < output_resolution:
            scaled_image = cv2.resize(img, (h + 1, w + 1), interpolation=cv2.INTER_CUBIC)
            params, img = train_obj.train_cnn(scaled_image)
            h, w, n_c = img.shape
    output_file_name = train_image + "_sr" + str(output_resolution) + "." + type
    cv2.imwrite(output_file_name, img)
    with open('params.pkl', 'wb') as file:
        pickle.dump(params, file)


def test_cnn(test_image_name):
    test_img = mpimg.imread(test_image_name)
    pkl_file = open('params.pkl', 'rb')
    type = test_image_name.rsplit(".")[1]
    params = pickle.load(pkl_file)
    params1, params2, params3, params4, params5, params6, params7, params8, params9 = params
    h, w, ip_channels = test_img.shape
    out_channels = 64
    filter_size = 3
    padding = 1
    layer_1 = Convolution(ip_channels, out_channels, filter_size, padding)
    layer_1.weights = params1[0]
    layer_1.bias=params1[1]
    relu_1 = ReLU()
    layer_2 = Convolution(out_channels, out_channels, filter_size, padding)
    layer_2.weights = params2[0]
    layer_2.bias = params2[1]
    relu_2 = ReLU()
    layer_3 = Convolution(out_channels, out_channels, filter_size, padding)
    layer_3.weights = params3[0]
    layer_3.bias = params3[1]
    relu_3 = ReLU()
    layer_4 = Convolution(out_channels, out_channels, filter_size, padding)
    layer_4.weights = params4[0]
    layer_4.bias = params4[1]
    relu_4 = ReLU()
    layer_5 = Convolution(out_channels, out_channels, filter_size, padding)
    layer_5.weights = params5[0]
    layer_5.bias = params5[1]
    relu_5 = ReLU()
    layer_6 = Convolution(out_channels, out_channels, filter_size, padding)
    layer_6.weights = params6[0]
    layer_6.bias = params6[1]
    relu_6 = ReLU()
    layer_7 = Convolution(out_channels, out_channels, filter_size, padding)
    layer_7.weights = params7[0]
    layer_7.bias = params7[1]
    relu_7 = ReLU()
    layer_8 = Convolution(out_channels, out_channels, filter_size, padding)
    layer_8.weights = params8[0]
    layer_8.bias = params8[1]
    relu_8 = ReLU()
    layer_9 = Convolution(out_channels, ip_channels, filter_size, padding)
    layer_9.weights = params9[0]
    layer_9.bias = params9[1]
    layers = [(layer_1, relu_1), (layer_2, relu_2), (layer_3, relu_3), (layer_4, relu_4), (layer_5, relu_5),
              (layer_6, relu_6),
              (layer_7, relu_7), (layer_8, relu_8), (layer_9, None)]
    cnn_obj = CNN(layers)
    out_img = cnn_obj.forward(test_img)
    output_file_name = test_image_name + "_sr" + str(h) + "." + type
    cv2.imwrite(output_file_name, out_img)


# main(sys.argv[1], sys.argv[2])
main('smallcat.jpg', 80)
#print(test_cnn('smallcat.jpg'))
