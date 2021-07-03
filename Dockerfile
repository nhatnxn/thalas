FROM continuumio/miniconda3:4.8.2

ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         manpages-dev \
         ffmpeg \
         libsm6 \
         libxext6 && \
    apt-get -y install supervisor && \
    rm -rf /var/lib/apt/lists/* && \
    conda install -y -c conda-forge uwsgi && \
    pip install --upgrade pip wheel && \
    pip install torch==1.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install torchvision==0.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install Pillow==8.0.1 &&\
    rm -rf /opt/miniconda/pkgs

RUN mkdir -p /usr/src/app

WORKDIR /usr/src/app

COPY . /usr/src/app

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8080

ENTRYPOINT ["uwsgi"]

CMD ["uwsgi.ini"]