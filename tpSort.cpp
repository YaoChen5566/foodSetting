#include <iostream>
#include <vector>
#include <queue>

#include "tpSort.h"

using namespace std;

//topological sort with Kahn's algorithm

topo::topo(int num)
{
	_num = num;
	cout <<"num: "<<_num<<endl;
	for(int i = 0 ; i < _num ; i++)
	{
		vector<int> tmp;
		_adjList.push_back(tmp);
		_count.push_back(0);
	}
}

void topo::addEdge(int start, int end)
{
	_adjList[start].push_back(end);
}

void topo::delEdge(int start, int end)
{
	vector<int>::iterator it = find (_adjList[start].begin(), _adjList[start].end(), end);
	if (it != _adjList[start].end())
	{
		_adjList[start].erase(it);
		//cout << "Element found in myvector: " << *it << '\n';
	}
	else
	{
		cout << "Element not found in myvector\n";
	}

}

void topo::printAdjList()
{
	for(int i = 0 ; i < _adjList.size() ; i++)
	{
		cout << i << " :";
		for(int j = 0 ; j < _adjList[i].size() ; j++)
		{
			cout << " "<<_adjList[i][j];
		}
		cout << endl;
	}
}

void topo::topoSort()
{
	// indegree for each node
	for(int i = 0 ; i < _adjList.size() ; i++)
		for(int j = 0 ; j < _adjList[i].size() ; j++)
			_count[_adjList[i][j]]++;

	queue<int> Q;
	for(int i = 0 ; i < _count.size() ; i++)
		if(_count[i] == 0)
			Q.push(i);

	for(int i = 0 ; i < _num ; i++)
	{
		if(Q.empty())
			break;

		int s = Q.front();
		Q.pop();
		sortResult.push_back(s);
		
		_count[s] = -1;

		for(int j = 0 ; j < _adjList[s].size() ; j++)
		{
			int t = _adjList[s][j];
			_count[t]--;
			if(_count[t] == 0)
				Q.push(t);
		}
	}
}
 
// Returns true if the graph contains a cycle, else false.
bool topo::isCyclic()
{
	_flag = false;
	//-1 not explore, 0 been explore, 1 fully explored
	//vector<int> visited;
	_visited.clear();
	for(int i = 0 ; i < _num ; i++)
		_visited.push_back(-1);

	for(int i = 0 ; i < _num ; i++)
	{
		if(_visited[i] == -1)
			dfs(i);
		if(_flag)
			break;
	}

	if(_flag == true)
		return true;
	else
		return false;
}

void topo::dfs(int s)
{	
	for(int i = 0 ; i < _num ; i++)
	{
		cout << _visited[i]<<" ";
	}
	cout << endl;
	_visited[s] = 0;
	for(int i = 0 ; i < _adjList[s].size() ; i++)
	{
		if(_visited[ _adjList[s][i]] == -1)
			dfs(_adjList[s][i]);
		else
		{
			_flag = true;
			return;
		}
	}
	_visited[s] = 1;
}