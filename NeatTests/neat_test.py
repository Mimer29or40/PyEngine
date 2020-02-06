from neat import Counter, Genome, GenomeViewer


def setup_func(genome, node_inno, connection_inno):
    pass


def test_func(genome, node_inno, connection_inno):
    pass


def setup(func):
    global setup_func
    setup_func = func
    return func


def test(func):
    global test_func
    test_func = func
    return func


def run(file, num=1, output=True, **kwargs):
    node_inno = [Counter() for _ in range(num)] if num > 1 else Counter()
    con_inno = [Counter() for _ in range(num)] if num > 1 else Counter()
    
    genome = [Genome() for _ in range(num)] if num > 1 else Genome()
    
    setup_func(genome, node_inno, con_inno)
    
    name = 'output/' + file.split('/')[-1].replace('.py', '')
    
    viewer = GenomeViewer(**kwargs)
    viewer.orientation = GenomeViewer.RIGHT
    
    if output:
        if num > 1:
            for i, g in enumerate(genome):
                viewer.gen(g).save(f'{name}_{i}_before.png', format='PNG')
        else:
            viewer.gen(genome).save(f'{name}_before.png', format='PNG')
    
    test_func(genome, node_inno, con_inno)

    if output:
        if num > 1:
            for i, g in enumerate(genome):
                viewer.gen(g).save(f'{name}_{i}_after.png', format='PNG')
        else:
            viewer.gen(genome).save(f'{name}_after.png', format='PNG')
