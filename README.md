# Increasing the robustness of deep neural networks for text classification by examining adversarial examples
This repository containes code used for my [master thesis 'Increasing the robustness of deep neural networks for text classification by examining adversarial examples'](http://edoc.sub.uni-hamburg.de/informatik/volltexte/2018/238/).

If you use this code please cite:
```
@mastersthesis{Soll18,
    author = {Marcus Soll},
    title = {Increasing the robustness of deep neural networks for text
classification by examining adversarial examples},
    school = {Universität Hamburg},
    year = {2018},
    address = {},
    language = {eng},
}
```

## Required python modules
* keras (tested version: 2.1.5)
* tensorflow (tested version: 1.4.0)
* h5py
* numpy
* matplotlib
* gensim (tested version: 2.1.0)
* nltk (tested version: 3.2.5)
* yaml

## Enable TensorFlow in Keras

Edit *~/.keras/keras.json* set:

```
"backend": "tensorflow"
```

Alternatively use enviroment variables:

```
KERAS_BACKEND=tensorflow
```

## Memory usage
Some of the operations take quite a lot of memory. Usage on 32-bit machines / python or low memory machines is not recomended.

## License
GPL-3+
