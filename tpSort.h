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
	void topoSort();
	void printAdjList();
	vector<int> sortResult;
private:
	int _num;
	vector<vector<int>> _adjList;
	vector<int> _count;
};