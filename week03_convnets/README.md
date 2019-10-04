Lecture slides - __[click here](https://yadi.sk/i/M_5pn6-EYxjf3Q)__

__Note__: Seminars assume that you remember batch normalization and dropout from last lecture. If you don't, go recap week2.


## Materials
- [russian] Convolutional networks - [video](https://yadi.sk/i/hDIkaR4H3EtnXM)
- [english] Convolutional networks (karpathy) - [video](https://www.youtube.com/watch?v=AQirPKrAyDg)

- Reading
  - http://cs231n.github.io/convolutional-networks/
  - http://cs231n.github.io/understanding-cnn/
  - [a deep learning neophite cheat sheet](http://www.kdnuggets.com/2016/03/must-know-tips-deep-learning-part-1.html)
  - [more stuff for vision](https://bavm2013.splashthat.com/img/events/46439/assets/34a7.ranzato.pdf)
  - a [CNN trainer in a browser](https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html)
  
- Bonus reading:
  - Interpreting neural network predictions: [distill.pub post](https://distill.pub/2018/building-blocks/)


## Practice

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yandexdataschool/Practical_DL/blob/fall19/week03_convnets/seminar_pytorch.ipynb)


As usual, go to seminar_pytorch.ipynb and folow instructons from there.
There's also an alternative version for other frameworks (tensorflow+keras) available.

__Grading is DOUBLED__
starting at zero points
 * +4 for describing your iteration path in a report below.
 * +4 for building a network that gets above 20% accuracy
 * +2 for beating each of these milestones on TEST dataset:
    * 50% (10 total)
    * 60% (12 total)
    * 65% (14 total)
    * 70% (16 total)
    * 75% (18 total)
    * 80% (20 total)

Bonus points

Common ways to get bonus points are:
*    Get higher score, obviously.
*    Anything special about your NN. For example "A super-small/fast NN that gets 80%" gets a bonus.
*    Any detailed analysis of the results. (saliency maps, whatever)

Restrictions

*    Please do NOT use pre-trained networks for this assignment until you reach 80%.
        In other words, base milestones must be beaten without pre-trained nets (and such net must be present in the e-mail). After that, you can use whatever you want.
    you can use validation data for training, but you can't' do anything with test data apart from running the evaluation procedure.
