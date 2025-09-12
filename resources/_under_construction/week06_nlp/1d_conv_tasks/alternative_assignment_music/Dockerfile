FROM andrewosh/binder-base

MAINTAINER Alexander Panin <justheuristic@gmail.com>

USER root

RUN apt-get update
RUN apt-get install -y htop
RUN apt-get install -y unzip
RUN apt-get install -y cmake
RUN apt-get install -y zlib1g-dev
RUN apt-get install -y libjpeg-dev 
RUN apt-get install -y xvfb libav-tools xorg-dev python-opengl
RUN apt-get install -y libav-tools
RUN apt-get -y install swig

USER main

RUN pip install librosa
RUN pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
RUN pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
RUN pip install --upgrade https://github.com/yandexdataschool/AgentNet/archive/master.zip
RUN mkdir ~/gym2 && cd ~/gym2 && git clone https://github.com/openai/gym.git && cd gym && pip install -e .[atari]

RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade https://github.com/yandexdataschool/AgentNet/archive/master.zip
RUN mkdir ~/gym3 && cd ~/gym3 && git clone https://github.com/openai/gym.git && cd gym && /home/main/anaconda/envs/python3/bin/pip install -e .[atari]
