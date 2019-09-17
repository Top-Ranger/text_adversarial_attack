# Increasing the robustness of deep neural networks for text classification by examining adversarial examples
This repository containes code used for my [master thesis 'Increasing the robustness of deep neural networks for text classification by examining adversarial examples'](http://edoc.sub.uni-hamburg.de/informatik/volltexte/2018/238/). Some of the results were also published at the [International Conference on Artificial Neural Networks 2019 (ICANN2019)](https://link.springer.com/chapter/10.1007%2F978-3-030-30508-6_54).

## Citation
If you want to refer to the effectiveness of defensive distillation, please cite:
```
@InProceedings{10.1007/978-3-030-30508-6_54,
author="Soll, Marcus
and Hinz, Tobias
and Magg, Sven
and Wermter, Stefan",
editor="Tetko, Igor V.
and K{\r{u}}rkov{\'a}, V{\v{e}}ra
and Karpov, Pavel
and Theis, Fabian",
title="Evaluating Defensive Distillation for Defending Text Processing Neural Networks Against Adversarial Examples",
booktitle="Artificial Neural Networks and Machine Learning -- ICANN 2019: Image Processing",
year="2019",
publisher="Springer International Publishing",
address="Cham",
pages="685--696",
isbn="978-3-030-30508-6"
}
```

Otherwise, If you use this code please cite:
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
