from neat import NodeType, Node, Connection, Genome
from NeatTests.neat_test import setup, test, run

if __name__ == '__main__':
    @setup
    def setup(genome, node_inno, con_inno):
        nodes = 10
        for _ in range(nodes):
            node = Node(node_inno[0].inc, -1, NodeType.HIDDEN)
            genome[0].add_node(node)
            genome[1].add_node(node)
        
        cons = 5
        for _ in range(cons):
            con = Connection(con_inno[0].inc, 0, 0, 1., True)
            genome[0].add_connection(con)
            genome[1].add_connection(con)
        
        print(genome[0])
        print(genome[1])
        
        matching = Genome.count_matching_genes(*genome)
        disjoint = Genome.count_disjoint_genes(*genome)
        excess = Genome.count_excess_genes(*genome)
        print(f'Matching Genes: {matching}\t - Correct Answer: {nodes + cons}')
        print(f'Disjoint Genes: {disjoint}\t - Correct Answer: 0')
        print(f'Excess Genes:   {excess}\t - Correct Answer: 0')
        print()
        
        genome[0].add_node(Node(node_inno[0].inc, -1, NodeType.HIDDEN))
        genome[0].add_node(Node(node_inno[0].inc, -1, NodeType.HIDDEN))
        genome[0].add_node(Node(node_inno[0].inc, -1, NodeType.HIDDEN))
        genome[0].add_connection(Connection(con_inno[0].inc, 0, 0, 1., True))
        genome[0].add_connection(Connection(con_inno[0].inc, 0, 0, 1., True))
        genome[0].add_connection(Connection(con_inno[0].inc, 0, 0, 1., True))
        
        matching = Genome.count_matching_genes(*genome)
        disjoint = Genome.count_disjoint_genes(*genome)
        excess = Genome.count_excess_genes(*genome)
        print(f'Matching Genes: {matching}\t - Correct Answer: {nodes + cons}')
        print(f'Disjoint Genes: {disjoint}\t - Correct Answer: 0')
        print(f'Excess Genes:   {excess}\t - Correct Answer: 6')
        print()
        
        genome[1].add_node(Node(node_inno[0].inc, -1, NodeType.HIDDEN))
        genome[1].add_node(Node(node_inno[0].inc, -1, NodeType.HIDDEN))
        genome[1].add_node(Node(node_inno[0].inc, -1, NodeType.HIDDEN))
        genome[1].add_connection(Connection(con_inno[0].inc, 0, 0, 1., True))
        genome[1].add_connection(Connection(con_inno[0].inc, 0, 0, 1., True))
        genome[1].add_connection(Connection(con_inno[0].inc, 0, 0, 1., True))
        
        matching = Genome.count_matching_genes(*genome)
        disjoint = Genome.count_disjoint_genes(*genome)
        excess = Genome.count_excess_genes(*genome)
        print(f'Matching Genes: {matching}\t - Correct Answer: {nodes + cons}')
        print(f'Disjoint Genes: {disjoint}\t - Correct Answer: 6')
        print(f'Excess Genes:   {excess}\t - Correct Answer: 6')
        print()
        
        matching = Genome.count_matching_genes(*genome[::-1])
        disjoint = Genome.count_disjoint_genes(*genome[::-1])
        excess = Genome.count_excess_genes(*genome[::-1])
        print('Counting genes between same genomes, but with opposite parameters:')
        print(f'Matching Genes: {matching}\t - Correct Answer: {nodes + cons}')
        print(f'Disjoint Genes: {disjoint}\t - Correct Answer: 6')
        print(f'Excess Genes:   {excess}\t - Correct Answer: 6')
        print()
        
        node = Node(node_inno[0].inc, -1, NodeType.HIDDEN)
        genome[0].add_node(node)
        genome[1].add_node(node)

        con = Connection(con_inno[0].inc, 0, 0, 1., True)
        genome[0].add_connection(con)
        genome[1].add_connection(con)
        
        matching = Genome.count_matching_genes(*genome)
        disjoint = Genome.count_disjoint_genes(*genome)
        excess = Genome.count_excess_genes(*genome)
        print(f'Matching Genes: {matching}\t - Correct Answer: {nodes + cons + 2}')
        print(f'Disjoint Genes: {disjoint}\t - Correct Answer: 12')
        print(f'Excess Genes:   {excess}\t - Correct Answer: 0')
        print()
    
    
    run(__file__, 2, output=False, spacing=50)
