from neat import Connection
from neat import Genome
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

        for _ in range(3):
            genome.add_node(Node(node_inno.inc, 0, NodeType.INPUT))
        genome.add_node(Node(node_inno.inc, 2, NodeType.OUTPUT))
        genome.add_node(Node(node_inno.inc, 1, NodeType.HIDDEN))

        genome.add_connection(Connection(con_inno.inc, 0, 3, 1.0, True))
        genome.add_connection(Connection(con_inno.inc, 1, 3, 1.0, False))
        genome.add_connection(Connection(con_inno.inc, 2, 3, 1.0, True))
        genome.add_connection(Connection(con_inno.inc, 1, 4, 1.0, True))
        genome.add_connection(Connection(con_inno.inc, 4, 3, 1.0, True))
        genome.add_connection(Connection(con_inno.inc, 0, 4, 1.0, True))
        # genome.add_connection(Connection(con_inno.inc, 2, 4, 1.0, True))

        Genome.save(genome, "output/save_load")

    @test
    def test(genome, node_inno, con_inno):
        Genome.load("output/save_load.json", genome)

    run(__file__, 1)
