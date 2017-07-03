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