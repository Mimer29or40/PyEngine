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

    settings = Settings(1000)

    @setup
    def setup(genome, node_inno, con_inno):
        genome.add_node(Node(node_inno.inc, 0, NodeType.INPUT))
        genome.add_node(Node(node_inno.inc, 0, NodeType.INPUT))
        genome.add_node(Node(node_inno.inc, 0, NodeType.INPUT))
        genome.add_node(Node(node_inno.inc, 1, NodeType.OUTPUT))

        genome.add_connection(Connection(con_inno.inc, 0, 3, 0.5, True))
        genome.add_connection(Connection(con_inno.inc, 1, 3, 0.5, True))
        genome.add_connection(Connection(con_inno.inc, 2, 3, 0.5, True))

    def evaluator(genome):
        net = Network(genome)

        total = 0
        for result, input in zip(correct_results, inputs):
            outputs = net.calculate(input)
            distance = abs(result - outputs[0])
            total += distance * distance

        if len(genome.connections) > 20:
            total += len(genome.connections) - 20

        return 100 - total * 5

    def print_results(fittest_genome):
        net = Network(fittest_genome)
        for input in inputs:
            outputs = net.calculate(input)
            print(f"{outputs[0]}, ", end="")
        print()

    run(__file__, settings, evaluator, print_results)
