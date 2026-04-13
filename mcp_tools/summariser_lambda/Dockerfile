FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt .
RUN pip install -r requirements.txt --target /var/lang/lib/python3.11/site-packages/ \
    && rm -rf ~/.cache/pip

COPY handler.py .

CMD ["handler.handler"]
