## Materials
* Adaptive optimization methods (russian) - [video](https://yadi.sk/i/SAGl44PS3EHZeK)
* Deep learning frameworks (russian) - [video](https://www.youtube.com/watch?v=ghZyptkanB0) 

* Stochastic gradient descent modiffications (english) - [video](https://www.youtube.com/watch?v=nhqo0u1a6fw)
* A blog post overview of gradient descent methods - [url](http://ruder.io/optimizing-gradient-descent/)
* Deep learning frameworks (english) - [video](https://www.youtube.com/watch?v=Vf_-OkqbwPo)


## More on adaptive optimization
* [Cool interactive demo of momentum](http://distill.pub/2017/momentum/)
* [wikipedia on SGD :)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent), expecially the "extensions and variants" section
* [RMSPROP video](https://www.youtube.com/watch?v=defQQqkXEfE)


## More on DL frameworks
  - A lecture on nonlinearities, intializations and other tricks in deep learning (karpathy) - [video](https://www.youtube.com/watch?v=GUtlrDbHhJM)
  - A lecture on activations, recap of adaptive SGD and dropout (karpathy) - [video](https://www.youtube.com/watch?v=KaR4lIdI1MQ)
  - [a deep learning neophite cheat sheet](http://www.kdnuggets.com/2016/03/must-know-tips-deep-learning-part-1.html)
  - [bonus video] Deep learning philosophy: [our humble take](https://www.youtube.com/watch?v=9qyE1Ev1Xdw) (english)
  - [reading] on weight initialization: [blog post](http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization)
  - [reading] pretty much all the module 1 of http://cs231n.github.io/


## Practice

As usual, go to `seminar_pytorch.ipynb` and follow instructions from there.

This seminar doesn't count towards final grade, yet it teaches you the framework that you're gonna use throughout all subsequent assignments.

_There's also a theano & tensorflow assignments available in case you want to see deep learning from a viewpoint of static graphs. However, we recommend that you begin with pytorch version unless you're already proficient with it._

Choosing deep learning framework is not like choosing your Hogwarts house: if you pick one, you can always learn others later. At the time this file was last edited, pytorch was more popular among ML reasearchers while tensorflow is easier to deploy into production written in other languages. 

Theano's an odd one: it's similar to tensorflow, but much older. In fact, it's slowly dying. You can, however, still use it to optimize graphs to consume much less time and peak memory than in tensorflow. But otherwise, the only reason it's here is that some of the course staff harbour an irrational love for theano and lasagne against all odds. Proceed at your own risk.


