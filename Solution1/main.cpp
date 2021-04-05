#include <iostream>
#include <vector>

#include "Utils.h"

int main(int argc, char** argv) {

	vector<string> lines = {};

	vector<string> place = {};
	vector<int> year = {};
	vector<int> month = {};
	vector<int> day = {};
	vector<int> time = {};
	vector<int> temperature = {};

	ifstream myfile("temp_lincolnshire_short.txt");

	string line;
	char* token;

	while (getline(myfile, line))
	{
		//cout << temp << endl;
		lines.push_back(line);
		
	}




	cout << "Hello world" << endl;

};