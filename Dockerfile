FROM docker.io/library/python:slim

WORKDIR /opt/app-root/
ENV PATH=/opt/app-root/bin:$PATH

RUN TEMPPKG='gcc g++' \
    && apt update \
    && apt install -y libproj-dev libgeos-dev git proj-bin $TEMPPKG\
    && /usr/local/bin/python -m venv /opt/app-root/ \
    && git clone --depth 1 --branch master \
       -- https://github.com/uarenforecasting/erebos build \
    && pip install --no-cache-dir -r build/requirements.txt \
    && pip install build/. \
    && rm -rf build \
    && apt autoremove -y $TEMPPKG \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

CMD /opt/app-root/bin/erebos
