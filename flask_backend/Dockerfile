FROM python:3.7-buster

RUN mkdir -p /app
COPY . /app

WORKDIR /app

# ### Install dependencies ###
RUN pip install -r requirements.txt

# ### Set flask relate env variable ###
ENV FLASK_APP main.py
ENV FLASK_ENV development
EXPOSE 5000

CMD ["flask", "run", "--host", "0.0.0.0", "--with-threads"]