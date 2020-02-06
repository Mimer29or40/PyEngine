import json
import random as r
from enum import Enum, auto

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def step(x):
    return np.where(x < 0, 0., 1.)


def sigmoid(x):
    return 1. / (1. + np.exp(-4.9 * x))


def seed(seed):
    r.seed(seed)


def random(lower=None, upper=None):
    lower = 1 if lower is None else lower
    lower, upper = (0., lower) if upper is None else (lower, upper)
    return lower + r.random() * (upper - lower)


def random_int(lower, upper):
    return r.randint(lower, upper)


def random_bool():
    return r.random() < 0.5


def random_index(array):
    return list(array)[r.randint(0, len(array) - 1)]


def random_gaussian():
    return r.normalvariate(0.0, 1.0)


class Settings:
    C1 = 1.0
    """Excess gene weight. Used ingGenomic distance calculation."""
    
    C2 = 1.0
    """Disjoint gene weight. Used in genomic distance calculation."""
    
    C3 = 0.4
    """Excess gene weight. Used in genomic distance calculation."""
    
    DT = 3.0
    """Genomic distance before two genomes are different species."""
    
    ASEXUAL_REPRODUCTION_RATE = 0.25
    """Rate at which mutation happens without crossover."""
    
    MUTATION_RATE = 0.8
    """Rate at which a genome will have its weights mutated."""
    
    PERTURBING_RATE = 0.9
    """Rate at which a genome's weights will be nudged."""
    
    DISABLED_GENE_INHERITING_CHANCE = 0.75
    """Rate at which a child genome's connection will be disabled."""
    
    ADD_CONNECTION_RATE = 0.05
    """Rate at which a connection will be created."""
    
    ADD_NODE_RATE = 0.05
    """Rate at which a node will be created."""
    
    def __init__(self, pop_size):
        self._pop_size = pop_size
    
    @property
    def pop_size(self):
        """Population size."""
        return self._pop_size


class Counter:
    def __init__(self, start=0):
        self._value = start
    
    @property
    def inc(self):
        value, self._value = self._value, self._value + 1
        return value


class NodeType(Enum):
    INPUT, HIDDEN, OUTPUT = auto(), auto(), auto()


class Node:
    def __init__(self, id, layer, type):
        self.id = id
        self.layer = layer
        self.type = type
    
    def __repr__(self):
        return f'Node<id={self.id}, layer={self.layer}, {self.type}>'
    
    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id
    
    def copy(self):
        return Node(self.id, self.layer, self.type)


class Connection:
    def __init__(self, id, in_node, out_node, weight, enabled):
        self.id = id
        self.in_node = in_node
        self.out_node = out_node
        self._weight = weight
        self.enabled = enabled
    
    def __repr__(self):
        return 'Connection<id={}, {}->{}, weight={}, enabled={}>'.format(
            self.id,
            self.in_node,
            self.out_node,
            self.weight,
            self.enabled
        )
    
    def __eq__(self, other):
        return isinstance(other, Connection) and self.id == other.id
    
    @property
    def weight(self):
        return self._weight
    
    @weight.setter
    def weight(self, value):
        self._weight = np.clip(value, -1, 1)
    
    def copy(self):
        return Connection(
            self.id,
            self.in_node,
            self.out_node,
            self.weight,
            self.enabled
        )


class Genome:
    def __init__(self):
        self.nodes = {}
        self.connections = {}
        
        self.layers = 0
        
        self.fitness = 0
    
    def __repr__(self):
        return 'Genome[nodes={} connections={}, fitness={}]'.format(
            len(self.nodes.values()),
            len(self.connections.values()),
            self.fitness
        )
    
    def add_node(self, node):
        """
        Adds a node to the node dict with the innovation number as the key.
        
        Arguments:
        node -- The node to add.
        """
        self.nodes[node.id] = node
        
        self.layers = max(self.layers, node.layer + 1)
    
    def add_connection(self, connection):
        """
        Adds a connection to the connection dict with the innovation number as
        the key.
        
        Arguments:
        connection -- The connection to add.
        """
        self.connections[connection.id] = connection
    
    def mutate(self, perturbing_probability):
        """
        Mutates the weights for all connections in genome
        
        Arguments:
        probability -- The probability to nudge the weight by an amount.
        """
        for con in self.connections.values():
            if random() < perturbing_probability:
                con.weight *= random_gaussian()
            else:
                con.weight = random(-1, 1)
    
    def node_mutation(self, node_inno, connection_inno):
        """
        Creates a node between an existing connection and creates two
        connections.
        
        Arguments:
        node_inno -- Counter object that represents the node innovation.
        connection_inno -- Counter object that represents the connection
        innovation.
        
        Returns:
            If it was successful
        """
        # Find all expressed connections
        suitable_cons = list(filter(lambda c: c.enabled,
                                    self.connections.values()))
        
        if len(suitable_cons) == 0:
            return False
        
        con = random_index(suitable_cons)
        
        in_node = self.nodes[con.in_node]
        out_node = self.nodes[con.out_node]
        
        con.enabled = False
        
        new_node = Node(
            id=node_inno.inc,
            layer=in_node.layer + 1,
            type=NodeType.HIDDEN
        )
        con1 = Connection(
            id=connection_inno.inc,
            in_node=in_node.id,
            out_node=new_node.id,
            weight=1.0,
            enabled=True
        )
        con2 = Connection(
            id=connection_inno.inc,
            in_node=new_node.id,
            out_node=out_node.id,
            weight=con.weight,
            enabled=True
        )
        
        if new_node.layer == out_node.layer:
            for node in self.nodes.values():
                if node.layer >= new_node.layer:
                    node.layer += 1
                    self.layers = max(self.layers, node.layer + 1)
        
        self.add_node(new_node)
        self.add_connection(con1)
        self.add_connection(con2)
        return True
    
    def connection_mutation(self, connection_inno, attempts):
        """
        Attempts to create a connection between two random unconnected nodes.
        
        Arguments:
        connection_inno -- Counter object that represents the connection
        innovation.
        attempts -- Number of node pairs that will be tried if one fails.
        
        Returns:
            If it was successful.
        """
        tries = 0
        while tries < attempts:
            tries += 1
            
            # Get Random Nodes
            n1 = self.nodes[random_index(self.nodes.keys())]
            n2 = self.nodes[random_index(self.nodes.keys())]
            
            # Should Reverse
            if ((n1.type is NodeType.HIDDEN and n2.type is NodeType.INPUT) or
                (n1.type is NodeType.OUTPUT and n2.type is NodeType.HIDDEN) or
                    (n1.type is NodeType.HIDDEN and n2.type is NodeType.INPUT)):
                n1, n2 = n2, n1
            
            # Bad Connection Check 1
            if ((n1.type is NodeType.INPUT and n2.type is NodeType.INPUT) or
                (n1.type is NodeType.OUTPUT and n2.type is NodeType.OUTPUT) or
                    (n1.id == n2.id)):
                continue

            # Bad Connection Check 2
            if n1.layer == n2.layer:
                continue
            
            # Check for Circular Structures
            # List of nodes that should have their connections checked
            needs_checking = []
            # List of nodes that requires output from node2
            node_ids = []
            for con in self.connections.values():
                if con.in_node == n2.id:
                    # Connection comes from node2
                    needs_checking.append(con.out_node)
                    node_ids.append(con.out_node)
            
            while len(needs_checking) > 0:
                node_id = needs_checking.pop(0)
                for con in self.connections.values():
                    if con.in_node == node_id:
                        # Connection comes from needs_checking node
                        needs_checking.append(con.out_node)
                        node_ids.append(con.out_node)
            
            # Check if node1 is dependent on node2
            if any(i == n1.id for i in node_ids):
                continue
            
            # Existing or Reverse Existing Connection Check
            if any((con.in_node == n1.id and con.out_node == n2.id) or
                   (con.in_node == n2.id and con.out_node == n1.id)
                   for con in self.connections.values()):
                continue
            
            self.add_connection(Connection(
                id=connection_inno.inc,
                in_node=n1.id,
                out_node=n2.id,
                weight=random(-1, 1),
                enabled=True
            ))
            return True
        
        # print('could not mutate')
        return False
    
    def copy(self):
        genome = Genome()
        for node in self.nodes.values():
            genome.add_node(node.copy())
        for con in self.connections.values():
            genome.add_connection(con.copy())
        return genome
    
    @staticmethod
    def save(genome, file_name='genome'):
        nodes = {}
        for node in genome.nodes.values():
            nodes[node.id] = {
                'layer': node.layer,
                'type': str(node.type).replace('NodeType.', '')
            }
        
        cons = {}
        for con in genome.connections.values():
            cons[con.id] = {
                'in': con.in_node,
                'out': con.out_node,
                'weight': con.weight,
                'enabled': con.enabled
            }
        
        json.dump(
            {'nodes': nodes, 'cons': cons},
            open(f'{file_name}.json', 'w'),
            indent=4
        )
    
    @staticmethod
    def load(file_name, genome=None):
        obj = json.load(open(file_name, 'r'))
        
        if genome is None:
            genome = Genome()
        
        genome.nodes = {}
        for id, data in obj['nodes'].items():
            genome.add_node(Node(
                id=int(id),
                layer=data['layer'],
                type=NodeType[data['type']]
            ))

        genome.connections = {}
        for id, data in obj['cons'].items():
            genome.add_connection(Connection(
                id=int(id),
                in_node=data['in'],
                out_node=data['out'],
                weight=data['weight'],
                enabled=data['enabled'],
            ))
        return genome
    
    @staticmethod
    def crossover(genome1, genome2, disabled_gene_inheritance):
        """
        Created a child genome from parent genomes.

        Arguments:
        genome1 -- Parent genome one.
        genome2 -- Parent genome two.
        inherit -- Chance for new connection to be disabled.

        Returns:
            Child genome.
        """
        if genome2.fitness > genome1.fitness:
            genome1, genome2 = genome2, genome1
        
        child = Genome()
        
        for node in genome1.nodes.values():
            child.add_node(node.copy())
        
        for p1_con in genome1.connections.values():
            try:
                # Matching Gene
                p2_con = genome2.connections[p1_con.id]
                child_con = p1_con.copy() if random_bool() else p2_con.copy()
                disabled = not (p1_con.enabled and p2_con.enabled)
                child_con.enabled = not (disabled and
                                         random() < disabled_gene_inheritance)
                child.add_connection(child_con)
            except KeyError:
                # Disjointed or Excess Connection
                child.add_connection(p1_con.copy())
        return child
    
    @staticmethod
    def compatibility(genome1, genome2, c1, c2, c3):
        """
        Determines how similar two genomes are.

        Arguments:
        genome1 -- Genome one.
        genome2 -- Genome two.
        c1 -- Weight for excess genes.
        c2 -- Weight for disjoint genes.
        c3 -- Weight for average weight distance.

        Returns:
            Compatibility distance.
        """
        excess = Genome.count_excess_genes(genome1, genome2)
        disjoint = Genome.count_disjoint_genes(genome1, genome2)
        weight_diff = Genome.average_weight_diff(genome1, genome2)
        return excess * c1 + disjoint * c2 + weight_diff * c3
    
    @staticmethod
    def count_matching_genes(genome1, genome2):
        """
        Determines the number of matching connections and nodes between two
        genomes.

        Arguments:
        genome1 -- Genome one.
        genome2 -- Genome two.

        Returns:
            Number of matching genes.
        """
        count = 0
        
        inno1 = max(genome1.nodes.keys())
        inno2 = max(genome2.nodes.keys())
        
        for i in range(max(inno1, inno2) + 1):
            n1 = genome1.nodes.get(i, None)
            n2 = genome2.nodes.get(i, None)
            if not (n1 is None or n2 is None):
                count += 1

        inno1 = max(genome1.connections.keys())
        inno2 = max(genome2.connections.keys())
        
        for i in range(max(inno1, inno2) + 1):
            c1 = genome1.connections.get(i, None)
            c2 = genome2.connections.get(i, None)
            if not (c1 is None or c2 is None):
                count += 1
        
        return count
    
    @staticmethod
    def count_excess_genes(genome1, genome2):
        """
        Determines the number of excess connections and nodes between two
        genomes.

        Arguments:
        genome1 -- Genome one.
        genome2 -- Genome two.

        Returns:
            Number of excess genes.
        """
        count = 0
        
        inno1 = max(genome1.nodes.keys())
        inno2 = max(genome2.nodes.keys())
        
        for i in range(max(inno1, inno2) + 1):
            n1 = genome1.nodes.get(i, None)
            n2 = genome2.nodes.get(i, None)
            if ((n1 is None and inno1 < i and n2 is not None) or
                    (n2 is None and inno2 < i and n1 is not None)):
                count += 1
        
        inno1 = max(genome1.connections.keys())
        inno2 = max(genome2.connections.keys())
        
        for i in range(max(inno1, inno2) + 1):
            c1 = genome1.connections.get(i, None)
            c2 = genome2.connections.get(i, None)
            if ((c1 is None and inno1 < i and c2 is not None) or
                    (c2 is None and inno2 < i and c1 is not None)):
                count += 1
        
        return count
    
    @staticmethod
    def count_disjoint_genes(genome1, genome2):
        """
        Determines the number of disjoint connections and nodes between two
        genomes.

        Arguments:
        genome1 -- Genome one.
        genome2 -- Genome two.

        Returns:
            Number of disjoint genes.
        """
        count = 0
        
        inno1 = max(genome1.nodes.keys())
        inno2 = max(genome2.nodes.keys())
        
        for i in range(max(inno1, inno2) + 1):
            n1 = genome1.nodes.get(i, None)
            n2 = genome2.nodes.get(i, None)
            if ((n1 is None and inno1 > i and n2 is not None) or
                    (n2 is None and inno2 > i and n1 is not None)):
                count += 1
        
        inno1 = max(genome1.connections.keys())
        inno2 = max(genome2.connections.keys())
        
        for i in range(max(inno1, inno2) + 1):
            c1 = genome1.connections.get(i, None)
            c2 = genome2.connections.get(i, None)
            if ((c1 is None and inno1 > i and c2 is not None) or
                    (c2 is None and inno2 > i and c1 is not None)):
                count += 1
        
        return count
    
    @staticmethod
    def average_weight_diff(genome1, genome2):
        """
        Determines the average weight difference between two genomes.

        Arguments:
        genome1 -- Genome one.
        genome2 -- Genome two.

        Returns:
            Average weight difference.
        """
        count = 0
        weight_diff = 0
        
        inno1 = max(genome1.connections.keys())
        inno2 = max(genome2.connections.keys())
        
        for i in range(max(inno1, inno2)):
            c1 = genome1.connections.get(i, None)
            c2 = genome2.connections.get(i, None)
            if not (c1 is None or c2 is None):
                count += 1
                weight_diff += abs(c1.weight - c2.weight)
        
        return weight_diff / count if count > 0 else 2.


class Neuron:
    def __init__(self):
        self.inputs = []
        self.output = 0.
        self.output_ids = []
        self.weights = []
    
    def __repr__(self):
        return 'Neuron<inputs={} output={} outputs={} weights={}>'.format(
            self.inputs,
            self.output,
            self.output_ids,
            self.weights
        )
    
    def add_input(self):
        self.inputs = [None] * (len(self.inputs) + 1)
    
    def feed_input(self, input):
        for i, _input in enumerate(self.inputs):
            if _input is None:
                self.inputs[i] = input
                return
        raise RuntimeError('could not feed input')
    
    def add_output(self, output_id, weight):
        self.output_ids.append(output_id)
        self.weights.append(weight)
    
    def is_ready(self):
        for f in self.inputs:
            if f is None:
                return False
        return True
    
    def calculate(self):
        self.output = sigmoid(sum(self.inputs))
        return self.output
    
    def reset(self):
        self.inputs = [None] * len(self.inputs)
        self.output = 0.


class Network:
    def __init__(self, genome):
        self.input = []
        self.output = []
        self.neurons = {}
        
        for node in genome.nodes.values():
            neuron = Neuron()
            if node.type is NodeType.INPUT:
                neuron.add_input()
                self.input.append(node.id)
            elif node.type is NodeType.OUTPUT:
                self.output.append(node.id)
            self.neurons[node.id] = neuron
        
        for con in genome.connections.values():
            if not con.enabled:
                continue
            self.neurons[con.in_node].add_output(con.out_node, con.weight)
            self.neurons[con.out_node].add_input()
    
    def calculate(self, parameters):
        if len(parameters) != len(self.input):
            raise ValueError('invalid parameters')
        
        for n in self.neurons.values():
            n.reset()
        
        unprocessed = [n for n in self.neurons.values()]
        
        for input, parameter in zip(self.input, parameters):
            n = self.neurons[input]
            n.feed_input(parameter)
            n.calculate()
            for output, weight in zip(n.output_ids, n.weights):
                self.neurons[output].feed_input(n.output * weight)
            unprocessed.remove(n)
        
        loops = 0
        while len(unprocessed) > 0:
            loops += 1
            if loops > 1000:
                return None
            
            for n in unprocessed.copy():
                if n.is_ready():
                    n.calculate()
                    for id, weight in zip(n.output_ids, n.weights):
                        self.neurons[id].feed_input(n.output * weight)
                    unprocessed.remove(n)
        
        return [self.neurons[o].output for o in self.output]


class Evaluator:
    def __init__(self, settings, generator, evaluator, node_innovation,
                 con_innovation):
        self.settings = settings
        self.node_innovation = node_innovation
        self.con_innovation = con_innovation
        self.fittest_genome = None
        
        self.genomes = [generator() for _ in range(settings.pop_size)]
        self.evaluator = evaluator
        
        self.next_generation = []
        self.last_results = []
    
    def evaluate_generation(self):
        # TODO - Thread this bitch
        for genome in self.genomes:
            genome.fitness = self.evaluator(genome)
        
        self.genomes.sort(key=lambda g: g.fitness, reverse=True)
        
        self.last_results.clear()
        self.last_results.extend(self.genomes)
        
        champion = self.genomes[0]
        if self.fittest_genome is None or\
                champion.fitness > self.fittest_genome.fitness:
            self.fittest_genome = champion
        
        self.genomes = self.genomes[:int(len(self.genomes) / 10)]
        
        self.next_generation.clear()
        self.next_generation.append(champion)
        
        while len(self.next_generation) < self.settings.pop_size:
            if random() < self.settings.ASEXUAL_REPRODUCTION_RATE:
                parent = random_index(self.genomes)
                child = parent.copy()
                child.mutate(self.settings.PERTURBING_RATE)
            else:
                parent1 = random_index(self.genomes)
                parent2 = random_index(self.genomes)
                chance = self.settings.DISABLED_GENE_INHERITING_CHANCE
                child = Genome.crossover(parent1, parent2, chance)
                
                if random() < self.settings.MUTATION_RATE:
                    child.mutate(self.settings.PERTURBING_RATE)
                
                if random() < self.settings.ADD_NODE_RATE:
                    child.node_mutation(self.node_innovation, self.con_innovation)
                
                if random() < self.settings.ADD_CONNECTION_RATE:
                    child.connection_mutation(self.con_innovation, 100)

            self.next_generation.append(child)
        
        self.genomes.clear()
        self.genomes.extend(self.next_generation)


class GenomeViewer:
    RIGHT, LEFT, UP, DOWN = range(4)
    
    def __init__(self, node_size=10, spacing=10, scale=4):
        self._node_size = node_size
        self.spacing = spacing
        self._scale = scale
        self.font = ImageFont.truetype('arial.ttf', node_size * scale)

        self.render_disabled_cons = False
        
        self.text = (0, 0, 0)
        
        self.background = (255, 255, 255)
        self.node_outline = (0, 0, 0)
        self.input_node = (50, 200, 100)
        self.output_node = (50, 100, 200)
        self.hidden_node = (200, 100, 50)
        
        self.pos_con = (0, 0, 255, 100)
        self.neg_con = (255, 0, 0, 100)
        self.dis_con = (0, 0, 0, 100)
        
        self.orientation = GenomeViewer.RIGHT
    
    @property
    def node_size(self):
        return self._node_size
    
    @node_size.setter
    def node_size(self, value):
        self._node_size = value
        self.font = ImageFont.truetype('arial.ttf', value * self._scale)
    
    @property
    def scale(self):
        return self._scale
    
    @scale.setter
    def scale(self, value):
        self._scale = value
        self.font = ImageFont.truetype('arial.ttf', self._node_size * value)
    
    def gen(self, genome):
        spacing = (self.spacing + self._node_size * 2) * self._scale
        
        # Group all nodes into layer groups
        all_nodes, max_layer_count = [], 0
        for i in range(genome.layers):
            layer = [n.id for n in genome.nodes.values() if n.layer == i]
            max_layer_count = max(max_layer_count, len(layer))
            all_nodes.append(layer)
        
        border = self._node_size * 2 * self._scale

        # Generate point for each node in each layer
        nodes = {}
        image_w = max((max_layer_count - 1) * spacing + 2 * border, 10)
        image_h = max((genome.layers - 1) * spacing + 2 * border, 10)
        if self.orientation in [GenomeViewer.LEFT, GenomeViewer.RIGHT]:
            image_w, image_h = image_h, image_w
        
        if self.orientation == GenomeViewer.UP:
            for i, layer in enumerate(all_nodes):
                layer_len = len(layer)
                y_pos = border + (genome.layers - i - 1) * spacing
                for j, node in enumerate(layer):
                    if layer_len == max_layer_count:
                        x_pos = border + j * spacing
                    else:
                        x_pos = (j + 1) * image_w / (layer_len + 1)
                    nodes[node] = int(x_pos), int(y_pos)
        elif self.orientation == GenomeViewer.DOWN:
            for i, layer in enumerate(all_nodes):
                layer_len = len(layer)
                y_pos = border + i * spacing
                for j, node in enumerate(layer):
                    if layer_len == max_layer_count:
                        x_pos = border + j * spacing
                    else:
                        x_pos = (j + 1) * image_w / (layer_len + 1)
                    nodes[node] = int(x_pos), int(y_pos)
        elif self.orientation == GenomeViewer.RIGHT:
            for i, layer in enumerate(all_nodes):
                layer_len = len(layer)
                x_pos = border + i * spacing
                for j, node in enumerate(layer):
                    if layer_len == max_layer_count:
                        y_pos = border + j * spacing
                    else:
                        y_pos = (j + 1) * image_h / (layer_len + 1)
                    nodes[node] = int(x_pos), int(y_pos)
        elif self.orientation == GenomeViewer.LEFT:
            for i, layer in enumerate(all_nodes):
                layer_len = len(layer)
                x_pos = border + (genome.layers - i - 1) * spacing
                for j, node in enumerate(layer):
                    if layer_len == max_layer_count:
                        y_pos = border + j * spacing
                    else:
                        y_pos = (j + 1) * image_h / (layer_len + 1)
                    nodes[node] = int(x_pos), int(y_pos)
        
        image = Image.new('RGB', (image_w, image_h), color=self.background)
        draw = ImageDraw.Draw(image, mode='RGBA')
        
        # Draw connections first so they are under nodes
        for con in genome.connections.values():
            in_node, out_node = nodes[con.in_node], nodes[con.out_node]
            if con.enabled:
                color = self.pos_con if con.weight > 0 else self.neg_con
                width = int(abs(con.weight) * 5 * self._scale)
                draw.line([*in_node, *out_node], fill=color, width=width)
            elif self.render_disabled_cons:
                color = self.dis_con
                width = self._scale
                draw.line([*in_node, *out_node], fill=color, width=width)
        
        # Draw nodes
        for node_id, pos in nodes.items():
            node = genome.nodes[node_id]
            
            x, y = pos
            
            width = (self._node_size + 1) * self._scale
            bounds = [x - width, y - width, x + width, y + width]
            color = self.node_outline
            draw.ellipse(bounds, fill=color)
            
            width = self._node_size * self._scale
            bounds = [x - width, y - width, x + width, y + width]
            if node.type is NodeType.INPUT:
                color = self.input_node
            elif node.type is NodeType.OUTPUT:
                color = self.output_node
            else:
                color = self.hidden_node
            draw.ellipse(bounds, fill=color)
            
            text = str(node_id)
            w, h = self.font.getsize(text)
            pos = [x - w / 2, y - h / 2 - 1 * self._scale]
            draw.text(pos, text, fill=self.text, font=self.font)
        
        return image


def fitness_plot(fitness, segments=4):
    canvas = plt.gcf().canvas
    plt.plot(fitness)
    plt.title('Fitness over time')
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    locations, labels = [], []
    for i in range(segments + 1):
        loc = int(i / segments * len(fitness)) - 1
        locations.append(loc)
        labels.append(plt.Text(loc, 0, str(loc + 1)))
    plt.xticks(locations, labels)
    canvas.draw()
    return Image.fromarray(np.roll(
        np.fromstring(
            canvas.tostring_argb(),
            dtype=np.uint8
        ).reshape((*canvas.get_width_height()[::-1], 4)),
        3,
        axis=2
    ))
