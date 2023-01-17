from neat import *

from NeatTests.evaluator.eval_test import *

if __name__ == "__main__":
    inputs = [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
    correct_results = [0.0, 1.0, 1.0, 0.0]

    settings = Settings(100)

    @setup
    def setup(genome, node_inno, con_inno):
        genome.add_node(Node(node_inno.inc, 0, NodeType.INPUT))
        genome.add_node(Node(node_inno.inc, 0, NodeType.INPUT))
        genome.add_node(Node(node_inno.inc, 1, NodeType.OUTPUT))

        genome.add_connection(Connection(con_inno.inc, 0, 2, 0.5, True))
        genome.add_connection(Connection(con_inno.inc, 1, 2, 0.5, True))

    def evaluator(genome):
        weight_sum = sum(abs(c.weight) for c in genome.connections.values() if c.enabled)
        return 1000.0 / abs(100.0 - weight_sum)

    def print_results(fittest_genome):
        weight_sum = sum(abs(c.weight) for c in fittest_genome.connections.values() if c.enabled)
        print(weight_sum)

    run(__file__, settings, evaluator, print_results)
