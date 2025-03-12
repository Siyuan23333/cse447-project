FROM python:3.9

RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# You should install any dependencies you need here.
COPY src/requirements.txt /job/requirements.txt

RUN pip install --no-cache-dir -r /job/requirements.txt


