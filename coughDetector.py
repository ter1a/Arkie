from cProfile import label
from numpy import append
import pyaudio
import numpy
import warnings
from datetime import datetime
import tensorflow as tf
from pandas import DataFrame

def my_publish_callback(envelope, status):
    if status.is_error():
        ... #handle error here
        print(str(status.error_data.exception))
    else:
        print(envelope)


def get_noice_data():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        FORMAT = pyaudio.paFloat32
        SAMPLEFREQ = 44100
        FRAMESIZE = 1376
        NOFFRAMES = 32
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,channels=1,rate=SAMPLEFREQ,input=True,frames_per_buffer=FRAMESIZE)
        data = stream.read(NOFFRAMES*FRAMESIZE)
        decoded = numpy.fromstring(data, 'float32')

        return [decoded]

def append_to_excel(data):
    df = DataFrame([data])
    df.to_csv('coughData.csv',mode='a', header=False)
    print("recorded")

def predict():
    print("loading model") 
    interpreter = tf.lite.Interpreter(model_path="/Users/panwenxiao/OneDrive - McMaster University/Mac Hack/soundclassifier_with_metadata.tflite")
    interpreter.allocate_tensors()
    print("loaded model")

    input_details = interpreter.get_input_details()
    # print(str(input_details))
    output_details = interpreter.get_output_details()
    # print(str(output_details))


    while True:
        input_data = get_noice_data()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = numpy.squeeze(interpreter.get_tensor(output_details[0]['index']))
        result = output_data.tolist()
        labels = {0:'background Noice',1:'dry_cough'}
        print('result : {}, possibility : {}'.format(labels[result.index(max(result))], max(result)))
        if result.index(max(result))!=0 and max(result) > 0.82:
            append_to_excel([datetime.now().isoformat(),labels[result.index(max(result))],round(max(result),2)])

predict()