import matplotlib.pyplot as plt


def zadanie1():
    img = plt.imread('data\pic1.png')
    print(img.dtype)
    print(img.shape)

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    Y1 = 0.299 * R + 0.587 * G + 0.0114 * B
    Y2 = 0.2126 * R + 0.7152 * G + 0.0722 * B

    plt.subplot(3, 3, 1)
    plt.imshow(img)

    plt.subplot(3, 3, 2)
    plt.imshow(Y1, cmap=plt.cm.gray)

    plt.subplot(3, 3, 3)
    plt.imshow(Y2, cmap=plt.cm.gray)

    plt.subplot(3, 3, 4)
    plt.imshow(R, cmap=plt.cm.gray)

    plt.subplot(3, 3, 5)
    plt.imshow(G, cmap=plt.cm.gray)

    plt.subplot(3, 3, 6)
    plt.imshow(B, cmap=plt.cm.gray)

    R = img.copy()
    R[:, :, 1] = 0
    R[:, :, 2] = 0

    plt.subplot(3, 3, 7)
    plt.imshow(R)

    G = img.copy()
    G[:, :, 0] = 0
    G[:, :, 2] = 0

    plt.subplot(3, 3, 8)
    plt.imshow(G)

    B = img.copy()
    B[:, :, 0] = 0
    B[:, :, 1] = 0

    plt.subplot(3, 3, 9)
    plt.imshow(B)

    plt.show()
