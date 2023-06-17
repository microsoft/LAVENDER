FROM nvcr.io/nvidia/pytorch:21.10-py3

# install ffmpeg, may prompt for user input
RUN DEBIAN_FRONTEND="noninteractive"  apt-get -y update
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y ffmpeg

# install required packages through pip
# for docker image, use opencv-python-headless instead of opencv-python
COPY requirements.txt ./
RUN pip install -r requirements.txt && rm ./requirements.txt

# additonal packages through git repo
RUN python -c "import nltk; nltk.download('wordnet')"
RUN pip install git+https://github.com/epfml/sent2vec.git

RUN pip install git+https://github.com/openai/CLIP.git

# captioning

# captioning eval tool (java for PTBtokenizer and METEOR)
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y update
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends openjdk-8-jdk && apt-get clean

# Clone the caption evaluation API and download some files needed for evaluation
RUN mkdir /evalcap
RUN git clone https://github.com/LuoweiZhou/coco-caption.git /evalcap/coco_caption
RUN cd /evalcap/coco_caption && ./get_stanford_models.sh

# Clone the caption evaluation CIDEr API and download some files needed for evaluation
RUN git clone https://github.com/xiaoweihu/cider.git /evalcap/cider

# use the faster pillow-simd instead of the original pillow
# https://github.com/uploadcare/pillow-simd
RUN pip uninstall -y pillow && \
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

RUN pip --no-cache-dir install --force-reinstall -I pyyaml

RUN pip install --upgrade --force-reinstall git+https://github.com/arogozhnikov/einops.git
RUN pip install deepspeed

# new
Run pip install torch==1.10.2 torchvision==0.11.3
Run pip install toolz packaging
Run pip install scikit-image

WORKDIR /src