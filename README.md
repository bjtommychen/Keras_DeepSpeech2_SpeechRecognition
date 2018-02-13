### Info
Do SpeechRecognition using DeepSpeech2,  backend using Keras.

https://www.kaggle.com/c/tensorflow-speech-recognition-challenge

#### Dataset
TRAIN DATA: total 64722. spilit to 8:1:1

http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz


#### Logs
```python
# Epoch 15/20
# - 269s - loss: 0.1738 - val_loss: 0.8504
# param: 1367k
# acc. ~94%

Starting evaluate...
diff/total 921 1024 0.899
diff/total 1827 2048 0.892
diff/total 2742 3072 0.893
diff/total 3648 4096 0.891
diff/total 4558 5120 0.89
diff/total 5474 6144 0.891
diff/total 6395 7168 0.892
```

#### Refer
1. Mozilla [DeepSpeech](https://github.com/mozilla/DeepSpeech)
2. Baidu [DS1](https://arxiv.org/abs/1412.5567) & [DS2](https://arxiv.org/abs/1512.02595) papers

