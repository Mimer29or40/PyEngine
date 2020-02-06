from neat import NodeType, Node, Connection, seed, Genome, Neuron
from NeatTests.neat_test import setup, test, run

if __name__ == '__main__':
    @setup
    def setup(genome, node_inno, con_inno):
        seed(1337)
        
        print('========== Test 1 ==========')
        test1 = Neuron()
        test1.add_input()
        test1.add_input()
        test1.add_input()
        
        test1.feed_input(1.0)
        print(f'test1.is_ready()={test1.is_ready()}, Actual: False')
        test1.feed_input(1.0)
        print(f'test1.is_ready()={test1.is_ready()}, Actual: False')
        test1.feed_input(1.0)
        print(f'test1.is_ready()={test1.is_ready()}, Actual: True')
        print('sum=3')
        print()
        
        print('Calculating . . . ', end='')
        print(test1.calculate())
        
        print('========== Test 2 ==========')
        test2 = Neuron()
        test2.add_input()
        test2.add_input()
        test2.add_input()

        test2.feed_input(0.0)
        test2.feed_input(0.5)
        test2.feed_input(-0.5)
        print('sum=0')
        
        print('Calculating . . . ', end='')
        print(test2.calculate())
        
        print('========== Test 3 ==========')
        test3 = Neuron()
        test3.add_input()
        test3.add_input()
        test3.add_input()

        test3.feed_input(-2.0)
        test3.feed_input(-2.0)
        test3.feed_input(-2.0)
        print('sum=-6')
        
        print('Calculating . . . ', end='')
        print(test3.calculate())
        
        print('========== Test 4 ==========')
        test4 = Neuron()
        test4.add_input()
        test4.add_input()
        test4.add_input()

        test4.feed_input(-20.0)
        test4.feed_input(-20.0)
        test4.feed_input(-20.0)
        print('sum=-60')
        
        print('Calculating . . . ', end='')
        print(test4.calculate())
    
    
    run(__file__, 1, output=False)
