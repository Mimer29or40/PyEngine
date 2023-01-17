from neat import Connection
from neat import Node
from neat import NodeType
from neat import seed

from NeatTests.neat_test import run
from NeatTests.neat_test import setup
from NeatTests.neat_test import test

if __name__ == "__main__":

    @setup
    def setup(genome, node_inno, con_inno):
        seed(1337)

        genome.add_node(Node(node_inno.inc, 0, NodeType.INPUT))
        genome.add_node(Node(node_inno.inc, 0, NodeType.INPUT))
        genome.add_node(Node(node_inno.inc, 1, NodeType.OUTPUT))

        genome.add_connection(Connection(con_inno.inc, 0, 2, 0.5, True))
        genome.add_connection(Connection(con_inno.inc, 1, 2, 1.0, True))

    @test
    def test(genome, node_inno, con_inno):
        n = 100000
        for _ in range(n):
            genome.mutate(0.7)

    run(__file__, 1)
