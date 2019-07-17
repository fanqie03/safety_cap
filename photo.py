import keras
import argparse
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/home/mf/datasets/helmet_class')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--epochs', default=100)
    parser.add_argument('--init_lr', default=0.1)

    return parser.parse_args()

args = get_args()

data_gen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1 / 255,
    horizontal_flip=True,
    validation_split=0.1,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    # brightness_range=[-0.1, 0.1],
)
train_generator = data_gen.flow_from_directory(
    directory=args.data_dir,
    target_size=(112, 112),
    batch_size=args.batch_size,
    subset='training',
)

c = 0
for x_batch, y_batch in train_generator:

    for i in range(16):
        # print(i // 4)
        plt.subplot(4, 4, i + 1)
        # plt.axis('off')
        plt.imshow(x_batch[i].reshape(112, 112, 3))
    plt.show()
    c += 1
    if c == 10:
        break