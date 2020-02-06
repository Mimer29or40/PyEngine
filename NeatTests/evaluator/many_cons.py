from NeatTests.evaluator.eval_test import *
from neat import *

if __name__ == '__main__':
    inputs = [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
    correct_results = [0.0, 1.0, 1.0, 0.0]

    settings = Settings(pop_size=100)
    
    @setup
    def setup(genome, node_inno, con_inno):
        genome.add_node(Node(node_inno.inc, 0, NodeType.INPUT))
        genome.add_node(Node(node_inno.inc, 0, NodeType.INPUT))
        genome.add_node(Node(node_inno.inc, 1, NodeType.OUTPUT))
        
        genome.add_connection(Connection(con_inno.inc, 0, 2, 0.5, True))
        genome.add_connection(Connection(con_inno.inc, 1, 2, 0.5, True))
    
    def evaluator(genome):
        return len(genome.connections)
    
    def print_results(fittest_genome):
        print(len(fittest_genome.connections))
    
    run(__file__, settings, evaluator, print_results, generations=100)
