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

        input1 = Node(node_inno.inc, 0, NodeType.INPUT)
        input2 = Node(node_inno.inc, 0, NodeType.INPUT)
        output = Node(node_inno.inc, 1, NodeType.OUTPUT)

        genome.add_node(input1)
        genome.add_node(input2)
        genome.add_node(output)

    @test
    def test(genome, node_inno, con_inno):
        genome.connection_mutation(con_inno, 10)

    run(__file__)
