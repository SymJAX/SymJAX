import symjax.tensor as T
import symjax as sj
import jax
import networkx as nx
import matplotlib.pyplot as plt


bin_rules = [
    [
        lambda *args: args[0] == 0 or args[1] == 0,
        jax.numpy.add,
        lambda *args: args[0] if args[1] == 0 else args[0],
    ],
    [lambda *args: len(args) == 1, jax.numpy.add, lambda *args: 2 * args[0]],
    [lambda *args: args[1] == 1, jax.numpy.true_divide, lambda *args: args[0]],
    [
        lambda *args: len(args) == 1,
        jax.numpy.true_divide,
        lambda *args: T.ones(args[0].shape, args[0].dtype),
    ],
    [
        lambda *args: args[0] == 1 or args[1] == 1,
        jax.numpy.multiply,
        lambda *args: args[0] if args[1] == 1 else args[1],
    ],
]


def simplify_add(graph):
    to_search = list(graph.nodes.keys())
    while len(to_search):
        j = to_search[-1]
        if type(j) == T.Op:

            for rule in bin_rules:

                if graph.get_node_attribute(j, "jax_function") == rule[1]:

                    predecessors = list(graph.predecessors(j))
                    try:
                        values = rule[0](*predecessors)
                    except IndexError:
                        values = False
                    if not values:
                        continue
                    new_node = rule[2](*predecessors)

                    for s in graph.successors(j):
                        graph.add_edge(new_node, s)
                        graph[new_node][s]["name"] = graph[j][s]["name"]

                        graph.remove_node(j)
                    break

        to_search.pop(-1)


a = T.ones(10)
b = a + 0
c = b / 1
d = c * 2

graph = sj.current_graph()

plt.subplot(211)
nx.draw(
    graph,
    with_labels=True,
    node_size=600,
    alpha=0.4,
    node_shape="s",
    pos=nx.spring_layout(graph),
)

simplify_add(sj.current_graph())

plt.subplot(212)
nx.draw(
    graph,
    with_labels=True,
    node_size=600,
    alpha=0.4,
    node_shape="s",
    pos=nx.spring_layout(graph),
)

f = sj.function(outputs=d)

print(f(), f())
plt.axis("off")
plt.show()
