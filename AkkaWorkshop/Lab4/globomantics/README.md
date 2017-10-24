An order in which processes should be started:

1. Lighthouse node to establish common communication point for the rest of the cluster. It doesn't do anything else.
2. API node where actual user nodes will live on. It may be run multiple times, just wait a second or two between firing next node.
3. Client node, used to communicate with user nodes. It doesn't spawn any user actors by itself, but has scheduled message requests and printer actor, that is supposed to receive recommendations.

This plugin uses SQLite database for persistence. Default db file is E:\\test.db - probably needs to be changed on your computer. However this path must be common for cluster nodes.

References:

- https://petabridge.com/blog/introduction-to-cluster-sharding-akkadotnet/
- https://petabridge.com/blog/cluster-sharding-technical-overview-akkadotnet/