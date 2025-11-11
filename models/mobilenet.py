from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

def build_feature_extractor():
    """Load MobileNetV2 as a pretrained feature extractor."""
    return MobileNetV2(include_top=False, pooling='avg', input_shape=(224, 224, 3))
