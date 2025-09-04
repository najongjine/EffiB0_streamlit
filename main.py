from keras import models
m = models.load_model('EfficientNetB0_wierd_animals.keras', safe_mode=False)
print('input_shape:', m.input_shape)  # 반드시 (None, 224, 224, 3) 여야 함