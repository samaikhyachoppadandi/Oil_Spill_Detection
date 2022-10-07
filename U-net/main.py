import numpy as np
from model import *
from data import *
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

data_gen_args = dict(rotation_range=0.2,
                     horizontal_flip=True,
                     fill_mode='nearest')
myGene = trainGenerator(2, 'data/oilspill/train', 'image', 'label', data_gen_args, save_to_dir=None)
model = None
model = unet()
model_checkpoint = ModelCheckpoint('unet_oilspill.hdf5', monitor='loss', verbose=1, save_best_only=True)
result = model.fit(myGene, steps_per_epoch=10, epochs=30, callbacks=[model_checkpoint])


testGene = testGenerator("data/oilspill/test/*")
results = model.predict_generator(testGene, verbose=1)
saveResult("data/oilspill/test", results)


plt.figure(1)
plt.plot(result.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('steps_per_epochs')
plt.legend(['training_acc'], loc="upper left")
plt.show()

plt.figure(2)
plt.plot(result.history['mean_io_u'])
plt.title('model iou obtained')
plt.ylabel('iou')
plt.xlabel('epochs')
plt.legend(['training_iou'], loc="upper left")
plt.show()

plt.figure(3)
plt.plot(result.history['loss1'])

plt.title('loss during training')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['training_loss'], loc="upper left")
plt.show()


