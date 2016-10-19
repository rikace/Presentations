FROM microsoft/aspnet

COPY . /app

WORKDIR /app

RUN ["dnu", "restore"]

RUN ["dnu", "publish", "src/AkkaDocker"]

ENTRYPOINT ["src/AkkaDocker/bin/output/AkkaDocker"]