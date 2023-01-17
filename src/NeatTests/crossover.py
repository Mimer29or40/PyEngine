from neat import NodeType, Node, Connection, Genome, seed
from NeatTests.neat_test import setup, test, run

if __name__ == '__main__':
    @setup
    def setup(genome, node_inno, con_inno):
        seed(1337)
        
        for _ in range(3):
            genome[0].add_node(Node(node_inno[0].inc, 0, NodeType.INPUT))
        genome[0].add_node(Node(node_inno[0].inc, 2, NodeType.OUTPUT))
        genome[0].add_node(Node(node_inno[0].inc, 1, NodeType.HIDDEN))
        
        genome[0].add_connection(Connection(0, 0, 3, 1.0, True))
        genome[0].add_connection(Connection(1, 1, 3, 1.0, False))
        genome[0].add_connection(Connection(2, 2, 3, 1.0, True))
        genome[0].add_connection(Connection(3, 1, 4, 1.0, True))
        genome[0].add_connection(Connection(4, 4, 3, 1.0, True))
        genome[0].add_connection(Connection(8, 0, 4, 1.0, True))
        
        for _ in range(3):
            genome[1].add_node(Node(node_inno[1].inc, 0, NodeType.INPUT))
        genome[1].add_node(Node(node_inno[1].inc, 3, NodeType.OUTPUT))
        genome[1].add_node(Node(node_inno[1].inc, 1, NodeType.HIDDEN))
        genome[1].add_node(Node(node_inno[1].inc, 2, NodeType.HIDDEN))
        
        genome[1].add_connection(Connection(0, 0, 3, 0.5, True))
        genome[1].add_connection(Connection(1, 1, 3, 0.5, False))
        genome[1].add_connection(Connection(2, 2, 3, 0.5, True))
        genome[1].add_connection(Connection(3, 1, 4, 0.5, True))
        genome[1].add_connection(Connection(4, 4, 3, 0.5, False))
        genome[1].add_connection(Connection(5, 4, 5, 0.5, True))
        genome[1].add_connection(Connection(6, 5, 3, 0.5, True))
        genome[1].add_connection(Connection(8, 2, 4, 0.5, True))
        genome[1].add_connection(Connection(10, 0, 5, 0.5, True))
    
    
    @test
    def test(genome, node_inno, con_inno):
        genome[2] = Genome.crossover(genome[0], genome[1], 0.10)
        print(genome[2])
        pass
    
    
    run(__file__, 3, spacing=50)
