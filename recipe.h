#include <iostream>
#include <vector>

using namespace std;

class recipe
{
public:

	recipe();

	void addFood(int _index, int _num, string _name, bool _type);
	void reduceFood(int _index);
	void writeRecipeFile(string _path);
	void readRecipeFile(string _path);
	void print();
	void reset();
	int size;
	vector<int> foodIndex();
	vector<int> foodNum();
	vector<string> foodName();
	vector<bool> foodType();

private:
	vector<int> _foodIndex;
	vector<int> _foodNum;
	vector<string> _foodName;
	vector<bool> _foodType;

	vector<int> __foodIndex;
	vector<int> __foodNum;
	vector<string> __foodName;
	vector<bool> __foodType;
};