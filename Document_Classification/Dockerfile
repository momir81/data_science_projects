FROM python:3.7

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt 

RUN python -m nltk.downloader punkt

RUN python -m nltk.downloader stopwords

RUN python -m nltk.downloader averaged_perceptron_tagger

RUN python -m nltk.downloader wordnet

ENTRYPOINT ["python","main.py"]

CMD ["corr", "project", "balance"]  
