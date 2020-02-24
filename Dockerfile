FROM docker.io/library/python:3.7-buster

WORKDIR /opt/app-root/
ENV PATH=/opt/app-root/bin:$PATH

RUN apt update \
    && apt install -y libproj-dev libgeos-dev git proj-bin gcc g++ \
    && apt clean \
    && rm -rf /var/lib/apt/lists/* \
    && /usr/local/bin/python -m venv /opt/app-root/ \
    && /opt/app-root/bin/pip install -U pip \
    && useradd -M -N -u 1001 -s /bin/bash -g 0 user


COPY . build/.

RUN pip install --no-cache-dir -r build/requirements.txt \
    && pip install build/. \
    && rm -rf build \
    && chown -R 1001:0 /opt/app-root
EXPOSE 8000
USER 1001

CMD ["/opt/app-root/bin/erebos"]
