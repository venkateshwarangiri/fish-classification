from src.data_loader import get_data
from src.cnn_model import create_cnn
from src.transfer_models import build_transfer_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0

input_shape = (224, 224, 3)
train_data, val_data = get_data(img_size=(224, 224))

num_classes = train_data.num_classes

# Train CNN
cnn = create_cnn(input_shape, num_classes)
cnn_checkpoint = ModelCheckpoint('models/cnn_model.h5', save_best_only=True)
cnn.fit(train_data, validation_data=val_data, epochs=10, callbacks=[cnn_checkpoint])

# Pretrained models
model_list = {
    "vgg16": VGG16,
    "resnet50": ResNet50,
    "mobilenet": MobileNet,
    "inceptionv3": InceptionV3,
    "efficientnetb0": EfficientNetB0
}

for name, fn in model_list.items():
    model = build_transfer_model(fn, input_shape, num_classes)
    checkpoint = ModelCheckpoint(f'models/{name}.h5', save_best_only=True)
    model.fit(train_data, validation_data=val_data, epochs=5, callbacks=[checkpoint])
