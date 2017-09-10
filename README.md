# Deep learning course @ fall'17
Fork of Lempitsky DL for HSE master students.

Lecture and seminar materials for each week are in ./week* folders

__Attention!__ This is a new iteration of on-campus deeplearning course. For full course materials '2016, go to [this branch](https://github.com/yandexdataschool/practical_deeplearning/tree/last_iteration)


# General info
* Create cloud jupyter with repo https://beta.mybinder.org/v2/gh/yandexdataschool/Practical_RL/fall17
* Lecture slides are stored in [this folder](https://yadi.sk/d/ExaKWAFN3MjPsd) (odp, pdf). You can also view slides from each week's page.
* Any technical issues, ideas, bugs in course materials, contribution ideas - add an [issue](https://github.com/yandexdataschool/HSE_deeplearning/issues)

# Enrollment guide
1. Bookmark repo https://github.com/yandexdataschool/practical_deeplearning
2. Join telegram chat https://t.me/dl_hse_fall17
3. Enroll to http://anytask.org with invite code __7pp6jP3__
4. Join piazza https://piazza.com/cs_hse/fall2017/dl101/home with access code __dl101__

# Announcements
* 06.09 - Course started

# Syllabus
- __week0__ Recap
  - [ ] Lecture: Linear models, stochastic optimization, basic neural networks and backprop
  - [ ] Seminar: Neural networks in numpy, adaptive SGD
     - [ ] HW due: 17.09.16, 23.59.
  - [ ] Please get bleeding edge theano+lasagne installed for the next seminar. 
    - [Issue](https://github.com/yandexdataschool/HSE_deeplearning/issues/1)
    - [Linux Guidelines](http://agentnet.readthedocs.io/en/latest/user/install.html)
    - You may choose tensorflow/pytorch version if you prefer 'em
- __week1__ Symbolic graphs
  - [ ] Lecture: Backprop recap. Deep learning frameworks. Some philosophy. DL tricks: dropout, normalization
  - [ ] Seminar: Symbolic graphs and basic neural networks
  
- __week2__ Deep learning for computer vision
  - [ ] Lecture: Convolutional neural networks, data augmentation & hacks.
  - [ ] Seminar: Convnets for CIFAR
  
- __week3__ Advanced computer vision
  - [ ] Lecture: Computer vision beyond image classification. Segmentation, object detection, identification. Model zoo & fine-tuning
  - [ ] Seminar: Model zoo. Siamese nets for identification.
  
- __week4__ Unsupervised & generative methods
  - [ ] Lecture: Autoencoders, Generative Adversarial Networks
  - [ ] Seminar: Generative Adversarial Networks. [hopefully] Art Style Transfer by Dmitry Ulyanov

- __week5__ Deep learning for natural language processing 101
  - [ ] Lecture: NLP problems and applications, bag of words, word embeddings, word2vec, text convolution.
  - [ ] Seminar: Word embeddings. Text convolutions for salary prediction.
  
- __week6__ Recurrent neural networks
  - [ ] Lecture: Simple RNN. Why BPTT isn't worth 4 letters. GRU/LSTM. Language modelling. Optimized softmax. Time series applications.
  - [ ] Seminar: Generating laws for pitiful humans with mighty RNNs.

- __week7__ Recurrent neural networks II
  - [ ] Lecture: Sequence labeling & applications. Seq2seq & applications. Attention. Batchnorm and dropout for RNN.
  - [ ] Seminar: Image Captioning

- __week8__: Deep reinforcement learning
  - [ ] Lecture: Reinforcement learning applications. Policy gradient. REINFORCE.
  - [ ] Seminar: REINFORCE agent with deep neural net policy for RL problems
  
- __week9__: Bayesian deep learning
  - [ ] Lecture: Bayesian vs Frequentist idea of probability. Bayesian methods around you. Variational Autoencoder. Bayesian Neural Network.
  - [ ] Seminar: Bayesian Neural Nets; Variational autoencoders [Hopefully by Mikhail Khalman]
 


# Stuff
* [One rule to rule them all](https://github.com/yandexdataschool/HSE_deeplearning/wiki/Core:)
* [Project rules](https://github.com/yandexdataschool/HSE_deeplearning/wiki/Course-projects)
* [Project examples](https://github.com/yandexdataschool/HSE_deeplearning/wiki/Project-examples)
* [Reducing lateness penalty](https://github.com/yandexdataschool/HSE_deeplearning/wiki/Back-to-the-future)
* [Feedback form (anonymous)](https://docs.google.com/forms/u/0/d/1HaODcG3vW7PAiQOUexZAwaZzrcGtVIYbJjymhLhgLYA/edit)

# Contributors & course staff
Course materials and teaching performed by
- [Fedor Ratnikov](https://github.com/justheuristic/) - lectures, seminars, hw checkups
- [Oleg Vasilev](https://github.com/Omrigan) - seminars, hw checkups, technical issue resolution
- [Arseniy Ashukha](https://github.com/ars-ashuha) - image captioning, sound processing, week7&9 lectures
- [Dmitry Ulyanov](https://github.com/DmitryUlyanov) - generative models, week8 lecture, week12 homework assignment
- [Mikhail Khalman](https://github.com/mihaha?tab=activity) - variational autoencoders, lecture 12
- [Vadim Lebedev](https://github.com/vadim-v-lebedev) - week0 & week6 homeworks
