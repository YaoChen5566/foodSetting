#include <iostream>
#include <vector>
#include <queue>

using namespace std;

//topological sort with Kahn's algorithm

class topo
{
public:
	topo(int num);

	void addEdge(int start, int end);
	void delEdge(int start, int end);
	void topoSort();
	void printAdjList();
	vector<int> sortResult;
	void dfs(int s);
	bool isCyclic();    // returns true if there is a cycle in this graph
private:
	int _num;
	bool _flag;
	vector<vector<int> > _adjList;
	vector<int> _count;
	vector<int> _visited;
};