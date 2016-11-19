# HSE_deeplearning
[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/yandexdataschool/hse_deeplearning)
Fork of Lempitsky DL for HSE master students
Lecture and seminar materials for each week is in ./week* folders

# Coordinates
* The classes are currently happening on __fridays at 18.00, room 400 (CS.HSE)__. There is also a backup thread for those who __cannot__ appear in the main thread(please e-mail or PM@slack us for details).
* Course mail - deeplearninghse16@gmail.com - for homework assignments, project stuff, questions, suggestions, intimidations, etc.
* Slack group - write a.manokhina@gmail.com for slack invitation [instructors appear ~once in two days]
* Any technical issues, ideas, bugs in course materials, contribution ideas - add an [issue](https://github.com/yandexdataschool/HSE_deeplearning/issues)


# Announcements
* 15.11 - HW checkup wave. If you sent us your homework solution before 05:00 15.11.2016 but we never replied yet - go get us @slack or post an issue here - we'll help
* 13.11 - we fix__d__ week7 assignment and uploaded all the PDFs. Homework checkup wave is underway :)
* x.11 - another wave of homework checkups happened
* 3.11 - week7 will happen on 11.11.2016 the regular way (friday, 18-00, room 400)
* 3.11 - we checked up all homework assigments sent to us before 4:00 am 3.11.16. If you were for some reason left behind - write us (mail, slack, github issue, irl, ...) - we'll fix that.
* 3.11 - week6 homework announced
* 1.11 - Week6 lecture will occur on Wednesday, 2.11 at 18-00 at approximately the same room 400.
* 26.10 - Preliminary poll on next lecture date: http://doodle.com/poll/cvew2rkn9u9x5ept . Please participate ASAP.
* 24.10 - Checked up homeworks "so far". If you sent us any mail@homework before 4.20 24.10.16 and we haven't replied yet - PM us in slack or create github issue or just contact us IRL - we'll fix that.
* 21.10 - There will be no classes on the 28.10;
* 21.10 - week5 notebook is NOT homework assignment notebook. Homework and deadlines TBA
* 20.10 - we now have a [![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/yandexdataschool/hse_deeplearning) for you if you got to the seminar without a notebook (with a phone) or have some temporary technical issues. Binder only lasts 1 hour so please do not use it for homeworks.
* 20.10 - current status of course staff is "Адъ и израиль", so the new wave of homeworks will be checked with a short delay (hope to finish by the weekend. The lectures will proceed as planned.
* 17.10 - week4 partially completed seminar uploaded
* 17.10 - a way for students to reduce lateness penalty on their homework assignments [announced](https://github.com/yandexdataschool/HSE_deeplearning/wiki/Back-to-the-future)
* 14.10 - week4 lecture, seminar and homework uploaded
* 12.10 - If you sent us anything before 6-00 AM (Moscow) 12.10.16 and still got no reply / no score [here](https://docs.google.com/spreadsheets/d/1z3v1CG3Y7qnvSUNNzGR6maZq0QzveWQNhRH2VcMm1fM/edit?usp=sharing) - contact us (slack, issue on github, anything) - there may be a problem with e-mail delivery.
* 12.10 - for those rare specimen who read official curriculum - we'll have have to reorder the curriculum. Advanced vision goes 2 weeks forward, advanced text gets 2 weeks sooner. The rest stays as planned.
* 7.10 - If you are considering project ideas, please contact us (slack >= mail) asap to know you exist. You don't have to know actual project topic - just tell us that you're there and you may be up to something.
* 7.10 - We've now got [da feedback form](https://docs.google.com/forms/u/0/d/1HaODcG3vW7PAiQOUexZAwaZzrcGtVIYbJjymhLhgLYA/edit) - it's fully anonymous and you can send there whatever you won't send us via slack/mail.
* 7.10 - Week3 lecture notes and homework uploaded.
* 30.09 - Week2 homework (lasagne/cifar): Please do not forget to add deterministic=True for your neural network when computing accuracy (not when training)
* 30.09 - added week2 lecture and homework.
* 30.09 - published [scoreboard](https://docs.google.com/spreadsheets/d/1z3v1CG3Y7qnvSUNNzGR6maZq0QzveWQNhRH2VcMm1fM/edit#gid=0)
* 23.09 - added week1 materials and homework
* 20.09 - by default we meet on __Friday at 18-00, room 400__. _if you cannot make it, please send us an e-mail (course mail) or PM me in slack ASAP - we have a second parallel available and a few other options_.
* 20.09 - HW0 deadline was shifted 1 wek into the future. Rejoice!
* 20.09 - grading info added
* 15.09 - Doodle on when do we meet - [link](http://doodle.com/poll/cygymdnkf7q5vqmm)
* 15.09 - Please get the frameworks installed by the next class - [issue](https://github.com/yandexdataschool/HSE_deeplearning/issues/1)
* 15.09 - added project rules and examples (see "course stuff" below)
* 15.09 - week0 assignment published (see "syllabus")


# Syllabus
- __week0__ Recap
  - [ ] Lecture: Linear models, stochastic optimization, regularization
  - [ ] Seminar: Linear classification, sgd, modifications
     - [ ] HW due: 28.09.16, 23.59.
  - [ ] Please get bleeding edge theano+lasagne installed for the next seminar. 
    - [Issue](https://github.com/yandexdataschool/HSE_deeplearning/issues/1)
    - [Linux Guidelines](http://agentnet.readthedocs.io/en/latest/user/install.html)
- __week1__ Getting deeper
  - [ ] Lecture: Neural networks 101
  - [ ] Seminar: theano, symbolic graphs and basic neural networks
    - [ ] HW due: 3.10.16 23.59 
- __week2__ Deep learning for computer vision 101
  - [ ] Lecture: Convolutional neural networks
  - [ ] Seminar: lasagne and CIFAR
    - [ ] HW due: 9.10.16 23.59 on first submission.
- __week3__ Deep learning for natural language processing 101
  - [ ] Lecture: NLP problems and applications, bag of words, word embeddings, word2vec, text convolution.
  - [ ] Seminar: Text convolutions for Avito content filtering task
    - [ ] HW due: 16.10.16 23.59 on first submission.
- __week4__ Recurrent neural networks for sequences
  - [ ] Lecture: Simple RNN. Why BPTT isn't worth 4 letters. GRU/LSTM. Language modelling. Optimized softmax. Time series applications.
  - [ ] Seminar: Generating laws for pitiful humans with mighty RNNs.
    - [ ] HW due: 28.10.16 23.59 on first submission.
- __week5__ Recurrent neural networks II
  - [ ] Lecture: Batchnorm and dropout for RNN; Seq2seq: machine translation, conversation models, speech recognition and more. Attention. Long term memory architectures.
  - [ ] Seminar: a toy machine translation task
    - to be anounced
- [Skip week]
- __week6__ Fine-tuning with neural networks
  - [ ] Lecture: Large CV datasets, model zoo, reusing pre-trained networks, fine-tuning, "knowledge transfer", soft-targets
  - [ ] Seminar: Cats Vs Dogs Vs Very Deep Networks
    - [ ] HW due 17.11.16 23.59
- __week7__ Advanced computer vision
  - [ ] Lecture: Representations within convnets, fully-convolutional networks, bounding box regression, maxout, etc.
  - [ ] Seminar: Image captioning by Arseniy Ashukha
    - [ ] HW due 24.11.16 23.59
- __week8__: Generative models for computer vision (around 18.11)
  - [ ] Lecture: Autoencoders, Generative Adversarial Networks
  - [ ] Seminar: Art Style Transfer with deep learning (Dmitry Ulyanov)

(future lectures in random order)
- __week[++i]__: Deep Reinforcement Learning I (Basic RL, Approximate RL, DQN, decorrelating)
- __week[++1]__: Deep Reinforcement Learning II (POMDP, continuous action space, hierarchical rl)
- __week[++i]__: Deep learning for sound processing (Ars)
- __week[++i]__: Bayesian methods in deep learning (Khalman)

# Stuff
* [One rule to rule them all](https://github.com/yandexdataschool/HSE_deeplearning/wiki/Core:)
* [Project rules](https://github.com/yandexdataschool/HSE_deeplearning/wiki/Course-projects)
* [Project examples](https://github.com/yandexdataschool/HSE_deeplearning/wiki/Project-examples)
* [Grading system](https://github.com/yandexdataschool/HSE_deeplearning/wiki/Grading)
* [Grading table](https://docs.google.com/spreadsheets/d/1z3v1CG3Y7qnvSUNNzGR6maZq0QzveWQNhRH2VcMm1fM/edit?usp=sharing)
* [Reducing lateness penalty](https://github.com/yandexdataschool/HSE_deeplearning/wiki/Back-to-the-future)
* [Feedback form (anonymous)](https://docs.google.com/forms/u/0/d/1HaODcG3vW7PAiQOUexZAwaZzrcGtVIYbJjymhLhgLYA/edit)
* [![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/yandexdataschool/hse_deeplearning)
