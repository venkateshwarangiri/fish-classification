from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data(img_size=(224, 224), batch_size=32):
    train_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
    val_gen = ImageDataGenerator(rescale=1./255)
    test_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        'data/train',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_data = val_gen.flow_from_directory(
        'data/val',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_data = test_gen.flow_from_directory(
        'data/test',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_data, val_data, test_data
