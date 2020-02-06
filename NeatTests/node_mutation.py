from neat import *
from NeatTests.neat_test import setup, test, run

if __name__ == '__main__':
    viewer = GenomeViewer()
    name = 'mutation/mutate'
    
    @setup
    def setup(genome, node_inno, con_inno):
        seed(1337)
        
        input1 = Node(node_inno.inc, 0, NodeType.INPUT)
        input2 = Node(node_inno.inc, 0, NodeType.INPUT)
        output = Node(node_inno.inc, 1, NodeType.OUTPUT)

        genome.add_node(input1)
        genome.add_node(input2)
        genome.add_node(output)

        con1 = Connection(con_inno.inc, 0, 2, 0.5, True)
        con2 = Connection(con_inno.inc, 1, 2, 0.75, True)

        genome.add_connection(con1)
        genome.add_connection(con2)

        viewer.gen(genome).save(f'{name}_0.png')

    @test
    def test(genome, node_inno, con_inno):
        for i in range(100):
            print(i)
            genome.node_mutation(node_inno, con_inno)
            for _ in range(2):
                genome.connection_mutation(con_inno, 100)
            viewer.gen(genome).save(f'{name}_{i + 1}.png')

    run(__file__, output=False)
