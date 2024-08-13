#ifndef CLAD_DIFFERENTIATOR_DYNAMICGRAPH_H
#define CLAD_DIFFERENTIATOR_DYNAMICGRAPH_H

#include <algorithm>
#include <functional>
#include <iostream>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace clad {
template <typename T> class DynamicGraph {
private:
  /// Storing nodes in the graph. The index of the node in the vector is used as
  /// a unique identifier for the node in the adjacency list.
  std::vector<T> m_nodes;

  /// Store the nodes in the graph as an unordered map from the node to a
  /// boolean indicating whether the node is processed or not. The second
  /// element in the pair is the id of the node in the nodes vector.
  std::unordered_map<T, std::pair<bool, size_t>> m_nodeMap;

  /// Store the adjacency list for the graph. The adjacency list is a map from
  /// a node to the set of nodes that it has an edge to. We use integers inside
  /// the set to avoid copying the nodes.
  std::unordered_map<size_t, std::set<size_t>> m_adjList;

  /// Set of source nodes in the graph.
  std::set<size_t> m_sources;

  /// Store the id of the node being processed right now.
  int m_currentId = -1; // -1 means no node is being processed.

  /// Maintain a queue of nodes to be processed next.
  std::queue<size_t> m_toProcessQueue;

public:
  DynamicGraph() = default;

  /// Add an edge from the source node to the destination node.
  /// \param src
  /// \param dest
  void addEdge(const T& src, const T& dest) {
    std::pair<bool, size_t> srcInfo = addNode(src);
    std::pair<bool, size_t> destInfo = addNode(dest);
    size_t srcId = srcInfo.second;
    size_t destId = destInfo.second;
    m_adjList[srcId].insert(destId);
  }

  /// Add a node to the graph. If the node is already present, return the
  /// id of the node in the graph. If the node is a source node, add it to the
  /// queue of nodes to be processed.
  /// \param node
  /// \param isSource
  /// \returns A pair of a boolean indicating whether the node is already
  /// processed and the id of the node in the graph.
  std::pair<bool, size_t> addNode(const T& node, bool isSource = false) {
    if (m_nodeMap.find(node) == m_nodeMap.end()) {
      size_t id = m_nodes.size();
      m_nodes.push_back(node);
      m_nodeMap[node] = {false, id}; // node is not processed yet.
      m_adjList[id] = {};
      if (isSource) {
        m_sources.insert(id);
        m_toProcessQueue.push(id);
      }
    }
    return m_nodeMap[node];
  }

  /// Add an edge from the current node being processed to the
  /// destination node.
  /// \param dest
  /// \param alreadyProcessed If the destination node is already processed.
  void addEdgeToCurrentNode(const T& dest, bool alreadyProcessed = false) {
    if (m_currentId != -1)
      addEdge(m_nodes[m_currentId], dest);
    if (alreadyProcessed)
      m_nodeMap[dest].first = true;
  }

  /// Set the current node being processed.
  /// \param node
  void setCurrentProcessingNode(const T& node) {
    if (m_nodeMap.find(node) != m_nodeMap.end())
      m_currentId = m_nodeMap[node].second;
  }

  /// Mark the current node being processed as processed and add the
  /// destination nodes to the queue of nodes to be processed.
  void markCurrentNodeProcessed() {
    if (m_currentId != -1) {
      m_nodeMap[m_nodes[m_currentId]].first = true;
      for (size_t destId : m_adjList[m_currentId])
        if (!m_nodeMap[m_nodes[destId]].first)
          m_toProcessQueue.push(destId);
    }
    m_currentId = -1;
  }

  /// Check if currently processing a node.
  /// \returns True if currently processing a node, false otherwise.
  bool isProcessingNode() { return m_currentId != -1; }

  /// Get the nodes in the graph.
  const std::vector<T>& getNodes() { return m_nodes; }

  /// Print the nodes and edges in the graph.
  void print() {
    // First print the nodes with their insertion order.
    for (const T& node : m_nodes) {
      std::pair<bool, int> nodeInfo = m_nodeMap[node];
      std::cout << (std::string)node << ": #" << nodeInfo.second;
      if (m_sources.find(nodeInfo.second) != m_sources.end())
        std::cout << " (source)";
      if (nodeInfo.first)
        std::cout << ", (done)\n";
      else
        std::cout << ", (unprocessed)\n";
    }
    // Then print the edges.
    for (int i = 0; i < m_nodes.size(); i++)
      for (size_t dest : m_adjList[i])
        std::cout << i << " -> " << dest << "\n";
  }

  /// Get the next node to be processed from the queue of nodes to be
  /// processed.
  /// \returns The next node to be processed.
  T getNextToProcessNode() {
    if (m_toProcessQueue.empty())
      return T();
    size_t nextId = m_toProcessQueue.front();
    m_toProcessQueue.pop();
    return m_nodes[nextId];
  }
};
} // end namespace clad

#endif // CLAD_DIFFERENTIATOR_DYNAMICGRAPH_H
