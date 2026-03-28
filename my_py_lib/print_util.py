import random
def print_randomly(a ,p=1):
    """
    p is the probability of printing.
    """
    if p<1:
        if random.random()>=p:
            return
    print(a)


__printed_values = {}
def print_once( a, id_ ):
    if id_ not in __printed_values:
        print(a)
        __printed_values[id_] = True


__printed_count = {}
def print_randomly_with_limit(
    a,
    id_,
    p=1, 
    MAX_prints=5,
):
    """
    p: the probability of printing
    MAX_prints: the maximum number of times to allow printing of 'a'
    """
    if id_ not in __printed_count:
        __printed_count[id_] = 0
    if __printed_count[id_] >= MAX_prints:
        return
    if p < 1:
        if random.random() >= p:
            return
    print(a)
    __printed_count[id_] += 1
