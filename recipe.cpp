#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <fstream>

#include "recipe.h"

using namespace std;

recipe::recipe()
{
	size = 0;
}

void recipe::addFood(int _index, int _num, string _name, bool _type)
{
	size++;
	_foodIndex.push_back(_index);
	_foodNum.push_back(_num);
	_foodName.push_back(_name);
	_foodType.push_back(_type);
	__foodIndex.push_back(_index);
	__foodNum.push_back(_num);
	__foodName.push_back(_name);
	__foodType.push_back(_type);
}

void recipe::reduceFood(int _index)
{
	vector<int>::iterator iter;
	iter = find(_foodIndex.begin(), _foodIndex.end(), _index);

	if(iter == _foodIndex.end())
		cout <<"dont have this food"<<endl;
	else
	{
		if(_foodNum[iter-_foodIndex.begin()] > 0)
			_foodNum[iter-_foodIndex.begin()]--;
	}
}

void recipe::writeRecipeFile(string _path)
{
	ofstream outfile(_path);
	outfile << size;
	outfile << endl;

	for(vector<int>::iterator iter = _foodIndex.begin() ; iter != _foodIndex.end() ; iter++)
	{
		if(iter != _foodIndex.begin())
			outfile<<" ";
		outfile << *iter;
	}

	outfile<<endl;

	for(vector<int>::iterator iter = _foodNum.begin() ; iter != _foodNum.end() ; iter++)
	{
		if(iter != _foodNum.begin())
			outfile<<" ";
		outfile << *iter;
	}

	outfile<<endl;

	for(vector<string>::iterator iter = _foodName.begin() ; iter != _foodName.end() ; iter++)
	{
		if(iter != _foodName.begin())
			outfile<<" ";
		outfile<<*iter;
	}

	outfile<<endl;

	for(vector<bool>::iterator iter = _foodType.begin() ; iter != _foodType.end() ; iter++)
	{		
		if(iter != _foodType.begin())
			outfile<<" ";
		outfile << *iter;
	}

	outfile.close();
}

void recipe::readRecipeFile(string _path)
{
	ifstream infile(_path);
	string s;
	string delimiter = " ";
	int line = 0;

	while(getline(infile, s))
	{
		string token;
		size_t pos = 0;
		if(line == 0)
		{
			size = stoi(s);
		}
		else
		{
			if(line == 1)
			{
				while((pos = s.find(delimiter)) != string::npos)
				{
					token = s.substr(0, pos);
					_foodIndex.push_back(stoi(token));
					__foodIndex.push_back(stoi(token));
					s.erase(0, pos + delimiter.length());
				}
				_foodIndex.push_back(stoi(s));
				__foodIndex.push_back(stoi(s));
			}
			else if(line == 2)
			{
				while((pos = s.find(delimiter)) != string::npos)
				{
					token = s.substr(0, pos);
					_foodNum.push_back(stoi(token));
					__foodNum.push_back(stoi(token));
					s.erase(0, pos + delimiter.length());
				}
				_foodNum.push_back(stoi(s));
				__foodNum.push_back(stoi(s));
			}
			else if(line == 3)
			{
				while((pos = s.find(delimiter)) != string::npos)
				{
					token = s.substr(0, pos);
					_foodName.push_back(token);
					__foodName.push_back(token);
					s.erase(0, pos + delimiter.length());
				}
				_foodName.push_back(s);
				__foodName.push_back(s);
			}
			else
			{
				while((pos = s.find(delimiter)) != string::npos)
				{
					token = s.substr(0, pos);
					_foodType.push_back(stoi(token) != 0);
					__foodType.push_back(stoi(token) != 0);
					s.erase(0, pos + delimiter.length());
				}
				_foodType.push_back(stoi(s) != 0);
				__foodType.push_back(stoi(s) != 0);
			}
		}
		line++;
	}
}

void recipe::print()
{
	cout << "size: " << size << endl;
	
	cout << "food Index:";
	for(vector<int>::iterator iter = _foodIndex.begin() ; iter != _foodIndex.end() ; iter++)
		cout << " " << *iter;
	cout << endl;

	cout << "food Num:";
	for(vector<int>::iterator iter = _foodNum.begin() ; iter != _foodNum.end() ; iter++)
		cout << " " << *iter;
	cout << endl;

	cout << "food Name:";
	for(vector<string>::iterator iter = _foodName.begin() ; iter != _foodName.end() ; iter++)
		cout << " " << *iter;
	cout << endl;

	cout << "food Type:";
	for(vector<bool>::iterator iter = _foodType.begin() ; iter != _foodType.end() ; iter++)
		cout << " " << *iter;
	cout << endl;

}

void recipe::reset()
{
	_foodIndex.clear();
	_foodNum.clear();
	_foodName.clear();
	_foodType.clear();

	_foodIndex.assign(__foodIndex.begin(), __foodIndex.end());
	_foodNum.assign(__foodNum.begin(), __foodNum.end());
	_foodName.assign(__foodName.begin(), __foodName.end());
	_foodType.assign(__foodType.begin(), __foodType.end());

}

vector<int> recipe::foodIndex()
{
	return _foodIndex;
}

vector<int> recipe::foodNum()
{
	return _foodNum;
}

vector<string> recipe::foodName()
{
	return _foodName;
}

vector<bool> recipe::foodType()
{
	return _foodType;
}