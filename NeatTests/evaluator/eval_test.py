from neat import *


def setup_func(genome, node_inno, connection_inno):
    pass


def setup(func):
    global setup_func
    setup_func = func
    return func


def run(file, settings, evaluator, print_results, generations=1000, **kwargs):
    genome, node_inno, con_inno = Genome(), Counter(), Counter()

    setup_func(genome, node_inno, con_inno)
    
    def generator():
        g = genome.copy()
        for con in g.connections.values():
            con.weight = random_gaussian()
        return g
    
    eva = Evaluator(settings, generator, evaluator, node_inno, con_inno)
    
    printer = GenomeViewer()

    name = 'output/' + file.split('/')[-1].replace('.py', '')
    
    fitness_over_time = []
    for i in range(generations):
        eva.evaluate_generation()
        
        fitness_over_time.append(eva.fittest_genome.fitness)
        
        print(f'Generation {i}')
        print(f'\tHighest Fitness: {eva.fittest_genome.fitness}')
        print(f'\tAmount of genomes: {len(eva.genomes)}')
        print('\tResults from best network:')
        
        print('\t\t', end='')
        print_results(eva.fittest_genome)
        
        if i % int(generations // 2) == 0:
            printer.gen(eva.fittest_genome).save(f'{name}_{i}.png')

    fitness_plot(fitness_over_time).save(f'{name}_fitness.png', **kwargs)

    printer.gen(eva.fittest_genome).save(f'{name}_{generations}.png')
    Genome.save(eva.fittest_genome, f'{name}_fittest')
