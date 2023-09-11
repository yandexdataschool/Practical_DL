
## Materials
* Russian lectures:
  * Lecture on basic neural networks (russian) - [video](https://yadi.sk/i/yyHZub6R3Ej5dV)
  * Backprop one formula at a time (russian) - [video](https://yadi.sk/i/0AuHgNsv3EHZhN)
* English lectures:
  * A lecture on backprop (karpathy) - [video](https://www.youtube.com/watch?v=59Hbtz7XgjM)
  * [alternative] A more classical lecture on neural networks (english) - [video](https://www.youtube.com/watch?v=uXt8qF2Zzfo)


## More materials
  - Interactive [neural network playground](http://playground.tensorflow.org/) in your browser
  - [Backprop by cs231](http://cs231n.github.io/optimization-2/)
  - A wonderful [blog post](http://karpathy.github.io/2019/04/25/recipe/) on what to expect from deep learning
  - [Notation](http://cs231n.github.io/neural-networks-1/#nn)
  - pretty much all the module 1 of http://cs231n.github.io/
  - [Cool interactive demo of momentum](http://distill.pub/2017/momentum/)
  - [wikipedia on SGD :)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent), expecially the "extensions and variants" section
  - [RMSPROP video](https://www.youtube.com/watch?v=defQQqkXEfE)


## Practice

Seminar: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yandexdataschool/Practical_DL/blob/fall23/week01_backprop/backprop.ipynb)


As usual, go to the only notebook in this folder (`adaptive_sgd.ipynb`) and follow instructions from there. Alternatively, [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yandexdataschool/Practical_DL/blob/fall23/week01_backprop/adaptive_sgd.ipynb)


Homework:
- (5 pts) implement backpropagation in backprop.ipynb such that it works for any number of layers,
- (5 pts) try several SGD modifications in adaptive_sgd.ipynb .

If you want to practice more, here's a few things you could try for bonus points:
- add better SGD or try extra "layers" in backprop.ipynb
- implement Adam in adaptive_sgd , based on [this paper](https://arxiv.org/abs/1412.6980)

However, don't feel obligated to do so unless you want to - the course is calibrated to allow you to get any grade without bonus assignments.


__Note__: Starting from the next seminar, assignments will require you to install a deep learning framework. Click [here](https://github.com/yandexdataschool/Practical_DL/issues/6) for details. You can also run everything in colab of course - it comes with pytorch pre-installed.

