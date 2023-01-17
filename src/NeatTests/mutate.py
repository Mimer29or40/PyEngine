from neat import NodeType, Node, Connection, seed
from NeatTests.neat_test import setup, test, run

if __name__ == '__main__':
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
