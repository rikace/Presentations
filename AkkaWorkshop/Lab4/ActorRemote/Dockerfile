FROM mono:4.4.0

MAINTAINER Riccardo Terrell

ENV CODEDIR /var/code
RUN mkdir -p $CODEDIR

copy . $CODEDIR

WORKDIR $CODEDIR

RUN xbuild ./ActorRemote.sln /property:Configuration=Release # /property:OutDir=/bin/Release

EXPOSE 9234

ENTRYPOINT ["mono", "./AkkaExamples/bin/Release/AkkaExamples.exe"]
