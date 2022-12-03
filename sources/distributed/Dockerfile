FROM ubuntu:20.04

RUN apt-get clean && apt-get update    
RUN apt-get install -yqq curl

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -yqq cmake build-essential git python3-pip 

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN pip  install -U pip
RUN pip install regex
RUN pip install  sympy
RUN pip install  pyyaml
RUN pip install  scipy

RUN pip install redis
RUN pip install types-redis
RUN pip install tox
RUN pip install mypy
RUN pip install black
RUN pip install flake8

RUN python -m pip install -U cython
RUN python -m pip install -U numpy 
RUN python -m pip install Flask==2.2.2

RUN python -m pip install scipy pandas joblib

ADD scikit-learn /tmp/sklearn

ADD https://api.github.com/repos/masterdoors/kernel_trees/git/refs/heads/master version.json

RUN mkdir /tmp/kernel_trees && \
    git clone https://github.com/masterdoors/kernel_trees /tmp/kernel_trees && \
#    mkdir /tmp/sklearn && \
#    git clone --branch 0.22.X  https://github.com/scikit-learn/scikit-learn.git /tmp/sklearn && \
#    cp /tmp/kernel_trees/thirdparty/base.py /tmp/sklearn/sklearn/svm/base.py && \
#    cp /tmp/kernel_trees/thirdparty/classes.py	/tmp/sklearn/sklearn/svm/classes.py && \
    #cp /tmp/kernel_trees/thirdparty/linear.cpp	/tmp/sklearn/sklearn/svm/src/liblinear/linear.cpp  && \
    #cp /tmp/kernel_trees/thirdparty/linear.h  /tmp/sklearn/sklearn/svm/src/liblinear/linear.h  && \
    #cp /tmp/kernel_trees/thirdparty/liblinear_helper.c  /tmp/sklearn/sklearn/svm/src/liblinear/liblinear_helper.c  && \
    cd /tmp/sklearn/ && \
            python setup.py build && \
            python setup.py install

#RUN apt-get install -yqq apache2 apache2-utils ssl-cert  libapache2-mod-wsgi-py3

#RUN cp /tmp/kernel_trees/sources/distributed/client.conf /etc/apache2/sites-enabled
#RUN chown -R www-data:www-data /var/www

#ENV APACHE_RUN_USER  www-data
#ENV APACHE_RUN_GROUP www-data
#ENV APACHE_LOG_DIR   /var/log/apache2
#ENV APACHE_PID_FILE  /var/run/apache2/apache2.pid
#ENV APACHE_RUN_DIR   /var/run/apache2
#ENV APACHE_LOCK_DIR  /var/lock/apache2
#ENV APACHE_LOG_DIR   /var/log/apache2

#RUN mkdir -p $APACHE_RUN_DIR
#RUN mkdir -p $APACHE_LOCK_DIR
#RUN mkdir -p $APACHE_LOG_DIR


EXPOSE 80
VOLUME data

#RUN rm -rf /etc/apache2/sites-enabled/000-default.conf

RUN mkdir /var/www
RUN cp -rf /tmp/kernel_trees/sources/distributed /var/www/

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

#CMD ["apache2ctl","-D","FOREGROUND"]

CMD ["/usr/bin/python","/var/www/distributed/client.py"]
