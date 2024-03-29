import json

from model import *
from tools import *

keras = tf.keras
layers = keras.layers
K = tf.keras.backend


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/home/mf/datasets/helmet_class')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--init_lr', default=0.001, type=float)
    parser.add_argument('--model', default='simple_net')

    return parser.parse_args()


def _main(args):
    data_gen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1 / 255,
        horizontal_flip=True,
        validation_split=0.1,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
    )
    train_generator = data_gen.flow_from_directory(
        directory=args.data_dir,
        target_size=(112, 112),
        batch_size=args.batch_size,
        subset='training',
    )
    validation_generator = data_gen.flow_from_directory(
        directory=args.data_dir,
        target_size=(112, 112),
        batch_size=args.batch_size,
        subset='validation'
    )

    csv_file = os.path.join(args.output_dir, 'log.csv')
    log_dir = os.path.join(args.output_dir, 'logs')
    ckpt_dir = os.path.join(args.output_dir, 'ckpt')
    ckpt_file = os.path.join(ckpt_dir,
                             args.model + '-val_loss-{val_loss:.4f}-val_acc-{val_acc:.4f}.h5')
    auto_save_file = os.path.join(args.output_dir, 'autosave_{}.h5'.format(args.model))
    plot_file = os.path.join(args.output_dir, 'model.png')
    class_file = os.path.join(args.output_dir, 'classes.json')

    with open(class_file, 'w') as f:
        json.dump(train_generator.class_indices, f)

    check_dir(log_dir)
    check_dir(ckpt_dir)

    model = eval(args.model)()

    model.compile(
        keras.optimizers.Adam(lr=args.init_lr),
        keras.losses.categorical_crossentropy,
        metrics=['acc']
    )

    keras.utils.plot_model(model, plot_file, show_shapes=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_file, save_best_only=True),
        keras.callbacks.CSVLogger(csv_file),
        keras.callbacks.ReduceLROnPlateau(patience=5, verbose=1, min_lr=1e-5),
        keras.callbacks.TensorBoard(log_dir),
        # keras.callbacks.EarlyStopping(patience=10),
    ]

    try:
        h = model.fit_generator(
            generator=train_generator,
            callbacks=callbacks,
            epochs=args.epochs,
            validation_data=validation_generator,
        )
    except KeyboardInterrupt as k:
        model.save(auto_save_file)


if __name__ == '__main__':
    args = get_args()
    _main(args)
