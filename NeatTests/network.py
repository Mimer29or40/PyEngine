from neat import NodeType, Node, Connection, seed, Genome, Neuron, Network
from NeatTests.neat_test import setup, run

if __name__ == '__main__':
    @setup
    def setup(genome, node_inno, con_inno):
        seed(1337)
        
        print('========== Test 1 ==========')
        genome = Genome()
        genome.add_node(Node(0, 0, NodeType.INPUT))
        genome.add_node(Node(1, 1, NodeType.OUTPUT))
        
        genome.add_connection(Connection(0, 0, 1, 0.5, True))
        
        net = Network(genome)
        input = [1.0]
        for _ in range(3):
            output = net.calculate(input)
            print(f'Output len={len(output)}, output[0]={output[0]}=0.9192')
        print()
        
        print('========== Test 2 ==========')
        genome = Genome()
        genome.add_node(Node(0, 0, NodeType.INPUT))
        genome.add_node(Node(1, 1, NodeType.OUTPUT))
        
        genome.add_connection(Connection(0, 0, 1, 0.1, True))
        
        net = Network(genome)
        input = [-0.5]
        for _ in range(3):
            output = net.calculate(input)
            print(f'Output len={len(output)}, output[0]={output[0]}=0.50973')
        print()
        
        print('========== Test 3 ==========')
        genome = Genome()
        genome.add_node(Node(0, 0, NodeType.INPUT))
        genome.add_node(Node(1, 2, NodeType.OUTPUT))
        genome.add_node(Node(2, 1, NodeType.HIDDEN))
        
        genome.add_connection(Connection(0, 0, 2, 0.4, True))
        genome.add_connection(Connection(1, 2, 1, 0.7, True))
        
        net = Network(genome)
        input = [0.9]
        for _ in range(3):
            output = net.calculate(input)
            print(f'Output len={len(output)}, output[0]={output[0]}=0.9524')
        print()
        
        print('========== Test 4 ==========')
        genome = Genome()
        genome.add_node(Node(0, 0, NodeType.INPUT))
        genome.add_node(Node(1, 0, NodeType.INPUT))
        genome.add_node(Node(2, 0, NodeType.INPUT))
        genome.add_node(Node(3, 2, NodeType.OUTPUT))
        genome.add_node(Node(4, 1, NodeType.HIDDEN))
        
        genome.add_connection(Connection(0, 0, 4, 0.4, True))
        genome.add_connection(Connection(1, 1, 4, 0.7, True))
        genome.add_connection(Connection(2, 2, 4, 0.1, True))
        genome.add_connection(Connection(3, 4, 3, 1.0, True))
        
        net = Network(genome)
        input = [0.5, 0.75, 0.9]
        for _ in range(3):
            output = net.calculate(input)
            print(f'Output len={len(output)}, output[0]={output[0]}=0.9924')
        print()
        
        print('========== Test 5 ==========')
        genome = Genome()
        genome.add_node(Node(0, 0, NodeType.INPUT))
        genome.add_node(Node(1, 0, NodeType.INPUT))
        genome.add_node(Node(2, 0, NodeType.INPUT))
        genome.add_node(Node(3, 2, NodeType.OUTPUT))
        genome.add_node(Node(4, 1, NodeType.HIDDEN))
        genome.add_node(Node(5, 1, NodeType.HIDDEN))
        
        genome.add_connection(Connection(0, 0, 4, 0.4, True))
        genome.add_connection(Connection(1, 1, 4, 0.7, True))
        genome.add_connection(Connection(2, 2, 4, 0.1, True))
        genome.add_connection(Connection(3, 4, 3, 1.0, True))
        genome.add_connection(Connection(4, 2, 5, 0.2, True))
        genome.add_connection(Connection(5, 5, 4, 0.75, True))
        genome.add_connection(Connection(6, 5, 3, 0.55, True))
        
        net = Network(genome)
        input = [1., 2., 3.]
        for _ in range(3):
            output = net.calculate(input)
            print(f'Output len={len(output)}, output[0]={output[0]}=0.99895')
        print()
    
    
    run(__file__, output=False)
