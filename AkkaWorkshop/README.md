# Actor Model in F# with Akka.NET and Docker
## Workshop

This session will be a mix of presentation and hands-on labs.  If you want to participate in the hands-on portion, we'll be using the following tools and servers:

## Pre-requisites
  - Visual Studio Code
  - Visual Studio 2015-2017
  - Docker 

There will be a handful of labs that we'll work on.  Most will have a scaffolded solution that contains entities and helper code (along with NuGet packages - using Paket).  You can either code along or view the solutions in hidden files.  _All solutions will be placed in a GitHub repo_.


### Visual Studio Code
Download: https://code.visualstudio.com

### Visual Studio 
Download: https://www.visualstudio.com/downloads/

### Docker (community edition)
It’s easy to get started with Docker. First, you have to install the Docker Engine on your machine (or your server). Follow the official instructions for Windows 10, Mac, or Linux.

Download: https://www.docker.com/community-edition

### RethinkDB
RethinkDB (to act as a local topic bus)

RethinkDB is an open-source, distributed database built to store JSON documents and effortlessly scale to multiple machines. It's easy to set up and learn and features a simple but powerful query language that supports table joins, groupings, aggregations, and functions.
by default, RethinkDB only accepts connections from localhost, otherwise, `rethinkdb --bind all` will bind to all network interfaces available to the container 

To start a container run the docker command  
- `docker run --name some-rethink -v "$PWD:/data" -d rethinkdb`

Connect the instance to an application
docker run --name some-app --link some-rethink:rdb -d application-that-uses-rdb

You can download the server here: http://rethinkdb.com/docs/install/
However, it will be used a RethinkDB Docker instance

Download the EXE and place it in a directory of your choice.
Open a command prompt, navigate to your EXE's directory and type:

    rethinkdb.exe --http-port 55000
(The HTTP Port can be of your choosing.  The default port requires you to run the command line as an administrator.)

After running the EXE, open a browser and navigate to

    http://localhost:55000
You should see the RethinkDB dashboard.  Play around with it -- we'll be using it a bit to create new tables and clear out data.


### EventStore
EventStore (event store)
  
To get started, pull the docker image
- `docker pull eventstore/eventstore`

Run the container using 
- `docker run --name eventstore-node -it -p 2113:2113 -p 1113:1113 eventstore/eventstore`

The admin UI and atom feeds will only work if you publish the node's http port to a matching port on the host. (i.e. you need to run the container with `-p 2113:2113` ex `http://192.168.99.100:2113/`)
Username and password is `admin` and `changeit` respectively.

You can download the server here: https://geteventstore.com/downloads/
However, it will be used an EventStore Docker instance

Download the ZIP file and extract it into a directory of your choice.
Open up a command prompt **AS AN ADMINISTRATOR**.
Navigate to where you extracted the ZIP file, and type:

    EventStore.ClusterNode.exe
There are a number of other command line options, so check the documentation for additional details.

Once running, open up a browser and navigate to:

    http://localhost:2113
You should be prompted for credentials.  Type **admin** for the user name and **changeit** for the password.  (You can change these later if you'd like.)
You should then be taken to the EventStore dashboard.


## Workshop Agenda
####	Lab 1:
- Implementing and sending messages to in-proc Actor
- Implementing a Remote Actor System
- Send messages to remote (out-of-proc) Actor 
- Remote deployment of arbitrary actor using <@ code-quotations @>

####	Lab 2:
- Docker images and containers
- Pull & Run Docker images
- Run interactive bash
- Pull, set and run Docker RethinkDB & EventStore

####	Lab 3:
- Implementing a Data Loader using RethinkDB, Reactive Extensions and Akka.NET
- Distribute the work-load using round-robin routing
- Plugging EventStore to Data Loader pipeline

####	Lab 4:
- Distribute work-load in Akka.NET Cluster 
- Converting Data Loader to Akka.NET Cluster
- Implementing a Docker image with Akka.NET Actor-System using DockerFiler
- Distribute work-load in Akka.NET Cluster using Docker

	*	An order in which processes should be started:

		- Lighthouse node to establish common communication point for the rest of the cluster. It doesn't do anything else.
		- API node where actual user nodes will live on. It may be run multiple times, just wait a second or two between firing next node.
		- Client node, used to communicate with user nodes. It doesn't spawn any user actors by itself, but has scheduled message requests and printer actor, that is supposed to receive recommendations.

			- This plugin uses SQLite database for persistence. Default db file is E:\\test.db - probably needs to be changed on your computer. However this path must be common for cluster nodes.

		References:

		- https://petabridge.com/blog/introduction-to-cluster-sharding-akkadotnet/
		- https://petabridge.com/blog/cluster-sharding-technical-overview-akkadotnet/


