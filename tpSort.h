#include <iostream>
#include <vector>
#include <queue>

using namespace std;

//topological sort with Kahn's algorithm

class topo
{
public:
	topo(int num);

	void addEdge(int start, int end, double weight);
	void delEdge(int start, int end, double weight);
	double getWeight(int start, int end);
	void topoSort();
	void printAdjList();
	vector<int> sortResult;
	void dfs(int s);
	void isCyclic();    // check if there is a cycle in this graph and delete the min edge in cycle
private:
	int _num;
	bool _flag;
	vector<vector<int> > _adjList;
	vector<vector<double> > _adjListW;
	vector<int> _count;
	vector<int> _visited;
};