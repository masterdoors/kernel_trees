FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-get clean && apt-get update    
RUN apt-get install -yqq curl

RUN apt-get install -yqq cmake build-essential \
        libboost-dev libboost-thread-dev libboost-program-options-dev \
        libboost-date-time-dev libboost-filesystem-dev libboost-iostreams-dev \
        libboost-system-dev libboost-chrono-dev libboost-timer-dev git zlib1g-dev python3-pip 

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN pip  install -U pip
RUN pip install regex
RUN pip install  sympy
RUN pip install  pyyaml
RUN pip install  scipy
RUN python -m pip install -U cython
RUN python -m pip install -U numpy 
RUN python -m pip install -U Flask

RUN python -m pip install scipy pandas joblib

COPY sklearn/scikit-learn /sklearn/scikit-learn
RUN cd sklearn/scikit-learn && \
            python setup.py build && \
            python setup.py install

RUN apt-get install -yqq apache2 apache2-utils ssl-cert libapache2-mod-wsgi-py3

COPY client.conf /etc/apache2/sites-enabled
RUN chown -R www-data:www-data /var/www

ENV APACHE_RUN_USER  www-data
ENV APACHE_RUN_GROUP www-data
ENV APACHE_LOG_DIR   /var/log/apache2
ENV APACHE_PID_FILE  /var/run/apache2/apache2.pid
ENV APACHE_RUN_DIR   /var/run/apache2
ENV APACHE_LOCK_DIR  /var/lock/apache2
ENV APACHE_LOG_DIR   /var/log/apache2

RUN mkdir -p $APACHE_RUN_DIR
RUN mkdir -p $APACHE_LOCK_DIR
RUN mkdir -p $APACHE_LOG_DIR

EXPOSE 80
VOLUME data

RUN rm -rf /etc/apache2/sites-enabled/000-default.conf

COPY client.py /var/www/client.py

CMD ["apache2", "-D", "FOREGROUND"]
