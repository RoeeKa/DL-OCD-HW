FROM nvidia/cuda:11.6.0-cudnn8-runtime-ubuntu18.04
WORKDIR /app
COPY . .
RUN apt-get -y update
RUN apt-get -y install python3.8
RUN apt -y install python3.8-distutils
RUN apt -y install python3.8-tk
RUN apt-get -y install libglu1-mesa
RUN apt-get -y install libglib2.0-0

RUN apt -y install curl
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py

RUN pip3.8 install -r requirements.txt
RUN python3.8 -c "import torchvision; vgg = torchvision.models.vgg11(pretrained=True)"
