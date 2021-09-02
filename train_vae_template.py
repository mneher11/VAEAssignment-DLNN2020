import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import sys
import pickle
import scipy.io
import tqdm


debug = False
img_rows = 28
img_cols = 20
ff = scipy.io.loadmat('data/frey_rawface.mat')
ff = ff["ff"].T.reshape((-1, 1, img_rows, img_cols))
ff = ff.astype('float32') / 255.
print(ff.shape)

n_samples = ff.shape[0]


# Number of parameters
input_size = 560
hidden_size = 128
latent_size = 16
std = 0.02
learning_rate = 0.02
loss_function = 'bce'  # mse or bce
beta1=0.9
beta2=0.999


def get_minibatch(batch_size, idx=0, indices=None):
    start_idx = batch_size * idx
    end_idx = min(start_idx + batch_size, n_samples)

    if indices is None:
        sample_b = ff[start_idx:end_idx]
    else:
        idx = indices[start_idx:end_idx]
        sample_b = ff[idx]

    sample_b = np.resize(sample_b, (batch_size, 560))

    sample_b = np.transpose(sample_b, (1, 0))

    return sample_b


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# The derivative of the sigmoid function
def dsigmoid(y, x=None):
    return y * (1 - y)

# The tanh function
def tanh(x):
    return np.tanh(x)


# The derivative of the tanh function
def dtanh(y, x=None):
    return 1 - y * y


def softplus(x):
    return np.log(1 + np.exp(x))


def dsoftplus(y, x=None):
    assert x is not None
    return sigmoid(x)


def sample_unit_gaussian(latent_size):
    return np.random.standard_normal(size=(latent_size))


# (Inplace) relu function
def relu(x):
    x[x < 0] = 0

    return x


# Gradient of Relu
def drelu(y, x=None):
    return 1. * (y > 0)


# Initialization was done not exactly according to Kingma et al. 2014 (he used Gaussian).
# input to hidden weight
Wi = np.random.uniform(-std, std, size=(hidden_size, input_size))
Bi = np.random.uniform(-std, std, size=(hidden_size, 1))

# encoding weight (hidden to code mean)
Wm = np.random.uniform(-std, std, size=(latent_size, hidden_size))  # hidden to mean
Bm = np.random.uniform(-std, std, size=(latent_size, 1))  # hidden to mean\

Wv = np.random.uniform(-std, std, size=(latent_size, hidden_size))  # hidden to logvar
Bv = np.random.uniform(-std, std, size=(latent_size, 1))  # hidden to logvar

# weight mapping code to hidden
Wd = np.random.uniform(-std, std, size=(hidden_size, latent_size))
Bd = np.random.uniform(-std, std, size=(hidden_size, 1))

# decoded hidden to output
Wo = np.random.uniform(-std, std, size=(input_size, hidden_size))
Bo = np.random.uniform(-std, std, size=(input_size, 1))


def forward(input, epsilon=None):
    # YOUR FORWARD PASS FROM HERE
    batch_size = input.shape[-1]

    H_e = np.zeros([hidden_size, batch_size])
    mean = np.zeros([latent_size, batch_size])
    logvar = np.zeros([latent_size, batch_size])
    z = np.zeros([latent_size, batch_size])
    H_d = np.zeros([hidden_size, batch_size])
    output = np.zeros([input_size, batch_size])
    p = np.zeros([input_size, batch_size])

    if input.ndim == 1:
        input = np.expand_dims(input, axis=1)

    # (1) linear
    # H = W_i \times input + Bi
    # np.dot(Wi, input, out=H_e)
    # H_e += Bi
    H_e = np.dot(Wi, input) + Bi

    # (2) ReLU
    # H = ReLU(H)
    H_e = relu(H_e)

    # (3) h > mu
    # Estimate the means of the latent distributions
    # mean = Wm \times H + Bm
    # np.dot(Wm, H_e, out=mean)
    # mean += Bm
    mean = np.dot(Wm, H_e) + Bm

    # (4) h > log var
    # Estimate the (diagonal) variances of the latent distributions
    # logvar = Wv \times H + Bv
    # np.dot(Wv, H_e, out=logvar)
    # logvar += Bv
    logvar = np.dot(Wv, H_e) + Bv

    # (5) sample the random variable z from means and variances (refer to the "reparameterization trick" to do this)
    if epsilon is None:
        epsilon = sample_unit_gaussian(latent_size=[latent_size, batch_size])
    # np.multiply(epsilon, np.exp(logvar/2), out=z)
    # z += mean
    z = np.multiply(epsilon, np.exp(logvar/2)) + mean

    # (6) decode z
    # D = Wd \times z + Bd
    # np.dot(Wd, z, out=H_d)
    # H_d += Bd
    H_d = np.dot(Wd, z) + Bd

    # (7) relu
    # D = ReLU(D)
    H_d = relu(H_d)

    # (8) dec to output
    # output = Wo \times D + Bo
    # np.dot(Wo, H_d, out=output)
    # output += Bo
    output = np.dot(Wo, H_d) + Bo

    # # (9) dec to p(x)

    # and (10) reconstruction loss function (same as the
    if loss_function == 'bce':
        p = sigmoid(output)
        loss = -np.sum(np.multiply(input, np.log(p)) + np.multiply(1 - input, np.log(1 - p)))
        # BCE Loss

    elif loss_function == 'mse':
        p = output
        loss = np.sum(0.5 * (p - input)**2)
        # MSE Loss

    # variational loss with KL Divergence between P(z|x) and U(0, 1)
    kl_div_loss = -1/2 * np.sum(1 + logvar - mean**2 - np.exp(logvar))
    # kl_div_loss = 0

    #kl_div_loss = - 0.5 * (1 + logvar - mean^2 - e^logvar)

    # your loss is the combination of
    #loss = rec_loss + kl_div_loss

    # Store the activations for the backward pass
    # activations = ( ... )
    activations = (epsilon, H_e, mean, logvar, z, H_d, output, p)
    return loss+kl_div_loss, kl_div_loss, activations


def decode(z):

    # basically the decoding part in the forward pass: maaping z to p

    # o = W_d \times z + B_d
    H_d = np.dot(Wd, z) + Bd
    H_d = relu(H_d)
    output = np.dot(Wo, H_d) + Bo

    # p = sigmoid(o) if bce or o if mse
    if loss_function == 'bce':
        p = sigmoid(output)
    elif loss_function == 'mse':
        p = output

    return p


def backward(input, activations, scale=True, alpha=1.0):
    # allocating the gradients for the weight matrice
    dWi = np.zeros_like(Wi)
    dWm = np.zeros_like(Wm)
    dWv = np.zeros_like(Wv)
    dWd = np.zeros_like(Wd)
    dWo = np.zeros_like(Wo)
    dBi = np.zeros_like(Bi)
    dBm = np.zeros_like(Bm)
    dBv = np.zeros_like(Bv)
    dBd = np.zeros_like(Bd)
    dBo = np.zeros_like(Bo)

    batch_size = input.shape[-1]
    scaler = batch_size if scale else 1

    # activations = (epsilon, H_e, mean, logvar, z, H_d, output, p)
    eps, H_e, mean, logvar, z, H_d, output, p = activations

    # backprop from (8) and (9) (if there is an additional activation function)
    if loss_function == 'mse':
        dp = p - input
        dp = dp / scaler
        do = dp

    elif loss_function == 'bce':
        dp = -1 * (input / p - (1 - input) / (1 - p))
        dp = dp / scaler
        do = np.multiply(dp, dsigmoid(p))

    # backprop from (7) through fully-connected
    dH_d = np.dot(Wo.T, do)
    dWo += np.dot(do, H_d.T)
    dBo += np.sum(do, axis=-1, keepdims=True)

    # backprop from (6) through ReLU
    dH_d = np.multiply(drelu(H_d), dH_d)

    # backprop from (5) through fully-connected
    dz = np.dot(Wd.T, dH_d)
    dWd += np.dot(dH_d, z.T)
    dBd += np.sum(dH_d, axis=-1, keepdims=True)

    # FROM PIXEL LOSS
    # backprop to mean
    dMean = dz
    dWm += np.dot(dMean, H_e.T)
    dBm += np.sum(dMean, axis=-1, keepdims=True)
    # backprop to mean
    dVar = np.multiply(dz, 0.5 * eps * np.exp(logvar/2))
    dWv += np.dot(dVar, H_e.T)
    dBv += np.sum(dVar, axis=-1, keepdims=True)

    # backprop to hidden state (encoder)
    dH_e = np.dot(Wm.T, dMean) + np.dot(Wv.T, dVar)
    dH_e = np.multiply(drelu(H_e), dH_e)

    dWi += np.dot(dH_e, input.T)
    dBi += np.sum(dH_e, axis=-1, keepdims=True)

    # FROM KL_LOSS
    dMean_KL = mean
    dVar_KL = 0.5 * (np.exp(logvar) - 1)

    # backprop to mean
    dWm += np.dot(dMean_KL, H_e.T)
    dBm += np.sum(dMean_KL, axis=-1, keepdims=True)
    # backprop to mean
    dWv += np.dot(dVar_KL, H_e.T)
    dBv += np.sum(dVar_KL, axis=-1, keepdims=True)

    # backprop to hidden state (encoder)
    dH_e_KL = np.dot(Wm.T, dMean_KL) + np.dot(Wv.T, dVar_KL)
    dH_e_KL = np.multiply(drelu(H_e), dH_e_KL)

    dWi += np.dot(dH_e_KL, input.T)
    dBi += np.sum(dH_e_KL, axis=-1, keepdims=True)

    gradients = (dWi, dWm, dWv, dWd, dWo, dBi, dBm, dBv, dBd, dBo)

    return gradients


def train():
    # Momentums for adagrad
    mWi, mWm, mWv, mWd, mWo = np.zeros_like(Wi), np.zeros_like(Wm), np.zeros_like(Wv),\
                              np.zeros_like(Wd), np.zeros_like(Wo)

    mBi, mBm, mBv, mBd, mBo = np.zeros_like(Bi), np.zeros_like(Bm), np.zeros_like(Bv), \
                              np.zeros_like(Bd), np.zeros_like(Bo)

    # Velocities for Adam
    vWi, vWm, vWv, vWd, vWo = np.zeros_like(Wi), np.zeros_like(Wm), np.zeros_like(Wv), \
                              np.zeros_like(Wd), np.zeros_like(Wo)

    vBi, vBm, vBv, vBd, vBo = np.zeros_like(Bi), np.zeros_like(Bm), np.zeros_like(Bv), \
                              np.zeros_like(Bd), np.zeros_like(Bo)

    def save_weights():

        print("Saving weights to %s and moments to %s" % ('weights.vae.pkl', 'momentums.vae.pkl'))

        weights = (Wi, Wm, Wv, Wd, Wo, Bi, Bm, Bv, Bd, Bo)
        with open('models/weights.vae.pkl', 'wb') as output:
            pickle.dump(weights, output, pickle.HIGHEST_PROTOCOL)

        momentums = (mWi, mWm, mWv, mWd, mWo, mBi, mBm, mBv, mBd, mBo)
        with open('models/momentums.vae.pkl', 'wb') as output:
            pickle.dump(momentums, output, pickle.HIGHEST_PROTOCOL)

        return

    batch_size = 128
    n_epoch = 100000

    save_every = 2000

    # first we have to shuffle the data
    n_samples = ff.shape[0]
    indices = np.arange(n_samples)
    total_loss = 0
    total_kl_loss = 0
    total_pixels = 0
    total_samples = 0
    count = 0
    alpha = 0.0

    n_minibatch = math.ceil(n_samples / batch_size)
    for epoch in range(n_epoch):

        rand_indices = np.random.permutation(indices)

        for i in range(n_minibatch):

            x_i = get_minibatch(batch_size, i, rand_indices)
            bsz = x_i.shape[-1]

            loss, kl_loss, acts = forward(x_i)
            _, _, _, _, z, _, _, _ = acts
            # lol I computed kl_div again here

            total_loss += loss
            total_kl_loss += kl_loss
            total_pixels += bsz * 560

            gradients = backward(x_i, acts, alpha=alpha)

            dWi, dWm, dWv, dWd, dWo, dBi, dBm, dBv, dBd, dBo= gradients

            count += 1

            # perform parameter update with Adagrad
            # perform parameter update with Adam
            for param, dparam, mem, velo in zip([Wi, Wm, Wv, Wd, Wo, Bi, Bm, Bv, Bd, Bo],
                                            [dWi, dWm, dWv, dWd, dWo, dBi, dBm, dBv, dBd, dBo],
                                            [mWi, mWm, mWv, mWd, mWo, mBi, mBm, mBv, mBd, mBo],
                                            [vWi, vWm, vWv, vWd, vWo, vBi, vBm, vBv, vBd, vBo]):
                mem += dparam * dparam
                param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

                # Adam update
                # bias_correction1 = 1 - beta1 ** count
                # bias_correction2 = 1 - beta2 ** count
                #
                # mem = mem * beta1 + (1 - beta1) * dparam
                # velo = velo * beta2 + (1 - beta2) * dparam * dparam
                # denom = np.sqrt(velo) / math.sqrt(bias_correction2) + 1e-9
                # step_size = learning_rate / bias_correction1
                #
                # param += -step_size * mem / denom

            total_samples += bsz  # lol it can be total_pixels / 560

            if count % 50 == 0:
                avg_loss = total_loss / total_pixels
                avg_kl = total_kl_loss / total_samples
                print("Epoch %d Iteration %d Updates %d Loss per pixel %0.6f avg KLDIV %0.6f " %
                      (epoch, i, count, avg_loss, avg_kl))

            # save weights to file every 500 updates so we can load to visualize later
            if count % 500 == 0:
                save_weights()

    return


def grad_check():
    batch_size = 8
    delta = 0.0001

    x = get_minibatch(batch_size)

    _, _, acts = forward(x)
    epsilon = acts[0]

    gradients = backward(x, acts, scale=False)

    dWi, dWm, dWv, dWd, dWo, dBi, dBm, dBv, dBd, dBo = gradients

    for weight, grad, name in zip([Wi, Wm, Wv, Wd, Wo, Bi, Bm, Bv, Bd, Bo],
                                  [dWi, dWm, dWv, dWd, dWo, dBi, dBm, dBv, dBd, dBo],
                                  ['Wi', 'Wm', 'Wv', 'Wd', 'Wo', 'Bi', 'Bm', 'Bv', 'Bd', 'Bo']):

        if name != 'Wv':
            continue

        str_ = ("Dimensions dont match between weight and gradient %s and %s." % (weight.shape, grad.shape))
        assert (weight.shape == grad.shape), str_

        print("Checking grads for weights %s ..." % name)
        n_warnings = 0
        for i in tqdm.tqdm(range(weight.size)):

            w = weight.flat[i]

            weight.flat[i] = w + delta
            loss_positive, _, _ = forward(x, epsilon)

            weight.flat[i] = w - delta
            loss_negative, _, _ = forward(x, epsilon)

            weight.flat[i] = w  # reset old value for this parameter

            grad_analytic = grad.flat[i]
            grad_numerical = (loss_positive - loss_negative) / (2 * delta)
            # print('Grad numerical: %f, grad analytical: %f' % (grad_numerical, grad_analytic))

            rel_error = abs(grad_analytic - grad_numerical) / (abs(grad_numerical + grad_analytic) + 1e-9)

            if rel_error > 0.001:
                n_warnings += 1
                print('WARNING %f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
        print("%d gradient mismatch warnings found. " % n_warnings)

    return


def eval():
    while True:

        # read weights from file
        cmd = input("Enter an image number:  ")

        img_idx = int(cmd)

        if img_idx < 0:
            exit()

        fig = plt.figure(figsize=(2, 2))
        n_samples = 1

        sample_ = ff[img_idx]
        org_img = sample_ * 255
        sample_ = np.resize(sample_, (1, 560)).T

        # Here the sample_ is processed by the network to produce the reconstruction

        img = np.sum(p, axis=-1)
        img = img / n_samples

        fig.add_subplot(1, 2, 1)
        plt.imshow(org_img.reshape(28, 20), cmap='gray')

        fig.add_subplot(1, 2, 2)
        plt.imshow(img.reshape(28, 20), cmap='gray')
        plt.show(block=True)

        print("Done")


def sample():
    while True:
        cmd = input("Press anything to continue, press q to quit:  ")
        if cmd == 'q':
            break

        z = np.random.randn(latent_size) * 10
        z = np.expand_dims(z, 1)

        # The decode function should be implemented before this
        p = decode(z)
        img = p

        fig = plt.figure(figsize=(2, 2))
        # gs = gridspec.GridSpec(4, 4)
        # gs.update(wspace=0.05, hspace=0.05)

        plt.imshow(img.reshape(28, 20), cmap='gray')
        # plt.title('reconstructed face %d' % 0)
        plt.show(block=True)


def forward_test():
    batch_size = 10
    indices = np.arange(batch_size)
    rand_indices = np.random.permutation(indices)
    x_i = get_minibatch(batch_size, 0, rand_indices)
    return forward(x_i)

if len(sys.argv) != 2:
    print("Need an argument train or gradcheck or reconstruct")
    exit()

option = sys.argv[1]

if option == 'train':
    train()
elif option in ['grad_check', 'gradcheck']:
    grad_check()
elif option in ['eval', 'sample', 'forward']:

    # read trained weights from file
    try:
        with open('models/weights.vae.pkl', "rb") as f:
            weights = pickle.load(f)

        Wi, Wm, Wv, Wd, Wo, Bi, Bm, Bv, Bd, Bo = weights
    except:
        print("No model weights found, inizializing random weights.")

    if option == 'eval':
        eval()
    elif option == 'forward':
        forward_test()
    else:
        sample()
else:
    raise NotImplementedError
