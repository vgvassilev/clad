#include "clad/Differentiator/Differentiator.h"

#include <iostream>
#include <string>

#include "gtest/gtest.h"

struct Node {
  std::string name;
  int id;

  Node(std::string name, int id) : name(name), id(id) {}

  bool operator==(const Node& other) const {
    return name == other.name && id == other.id;
  }

  // String operator for printing the node.
  operator std::string() const { return name + std::to_string(id); }
};

// Specialize std::hash for the Node type.
template <> struct std::hash<Node> {
  std::size_t operator()(const Node& n) const {
    return std::hash<std::string>()(n.name) ^ std::hash<int>()(n.id);
  }
};

TEST(DynamicGraphTest, Printing) {
  clad::DynamicGraph<Node> G;
  for (int i = 0; i < 6; i++) {
    Node n("node", i);
    if (i == 0)
      G.addNode(n, /*isSource=*/true);
    Node m("node", i + 1);
    G.addEdge(n, m);
  }
  std::vector<Node> nodes = G.getNodes();
  EXPECT_EQ(nodes.size(), 7);

  // Edge from node 0 to node 3 and node 4 to node 0.
  G.addEdge(nodes[0], nodes[3]);
  G.addEdge(nodes[4], nodes[0]);

  // Check the printed output.
  std::stringstream ss;
  std::streambuf* coutbuf = std::cout.rdbuf();
  std::cout.rdbuf(ss.rdbuf());
  G.print();
  std::cout.rdbuf(coutbuf);
  std::string expectedOutput = "node0: #0 (source), (unprocessed)\n"
                               "node1: #1, (unprocessed)\n"
                               "node2: #2, (unprocessed)\n"
                               "node3: #3, (unprocessed)\n"
                               "node4: #4, (unprocessed)\n"
                               "node5: #5, (unprocessed)\n"
                               "node6: #6, (unprocessed)\n"
                               "0 -> 1\n"
                               "0 -> 3\n"
                               "1 -> 2\n"
                               "2 -> 3\n"
                               "3 -> 4\n"
                               "4 -> 0\n"
                               "4 -> 5\n"
                               "5 -> 6\n";
  EXPECT_EQ(ss.str(), expectedOutput);
}
