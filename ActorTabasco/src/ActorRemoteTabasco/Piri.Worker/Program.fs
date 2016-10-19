open System
open Akka.FSharp
open Piri

let system = System.create "cluster" <| Configuration.parse """
akka {
    #loglevel = "DEBUG"
    actor {
        provider = "Akka.Cluster.ClusterActorRefProvider, Akka.Cluster"
        serializers {
          wire = "Akka.Serialization.WireSerializer, Akka.Serialization.Wire"
        }
        serialization-bindings {
          "System.Object" = wire
        }
    }
    remote {
        helios.tcp {
            hostname = "localhost"
            port = 0
        }
    }
    cluster {
        seed-nodes = [ "akka.tcp://cluster@localhost:8091/" ]
    }
    tabasco {
        assembly-cache-path = "tmp/"
        known-assemblies = [ "Akka.FSharp", "Akka", "Akka.Remote", "Akka.Cluster" ]
    }
}
"""

let piri = PiriPlugin.Get system
// worker supervisor
let server = piri.Server

Console.ReadLine()

