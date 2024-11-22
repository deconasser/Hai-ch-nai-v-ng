FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
ENV CUDA_HOME=/usr/local/cuda-11.3/
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

WORKDIR /base

RUN ln -sf /usr/share/zoneinfo/Asia/Ho_Chi_Minh /etc/localtime

RUN apt-get update && apt-get install -y python3 python3-pip cmake wget llvm
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx

RUN pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install terminaltables==3.1.10 pycocotools==2.0.7 fairscale==0.4.13 timm==0.9.16 yapf==0.40.2
RUN pip3 install -U openmim
RUN apt install -y wget
RUN wget https://bootstrap.pypa.io/get-pip.py -O ./get-pip.py
RUN python3 ./get-pip.py
RUN pip3 install yapf==0.40.1

RUN pip3 install --no-cache-dir --upgrade pip wheel setuptools
RUN mim install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
RUN pip3 install scipy==1.10.1
RUN apt-get install -y python3-magic
RUN pip3 install python-magic
RUN pip3 install ensemble_boxes

CMD bash train.sh && tail -f /dev/null
#CMD bash infer.sh && tail -f /dev/null