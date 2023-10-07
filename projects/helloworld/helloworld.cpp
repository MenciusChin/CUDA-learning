#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

int main()
{
    vector<string> msg {"Hello", "C++", "World", "from", "VS Code", "and the C++ extension!"};

    for (const string& word : msg)
    {
        cout << word << " ";
    }
    cout << endl;

    cout << sqrt(64) << endl;
    cout << round(2.6) << endl;
    cout << log(2) << endl;

    int time = 20;
    string result = (time < 18) ? "Good day." : "Good evening.";
    cout << result;
}