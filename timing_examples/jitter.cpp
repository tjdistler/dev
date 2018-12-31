#include <fstream>
#include <iostream>
#include <chrono>
#include <thread>
#include <map>
#include <cfloat>	//FLT_MIN
#include <math.h>	//round()
#include <iomanip> 	//setprecision()

#define FPS 30.0
#define RUNTIME 180 //sec

using namespace std;

int main(int argc, char** argv)
{
	const chrono::nanoseconds delay( (unsigned long long)( (1.0/FPS) * 1000000000.0 ) );
	const chrono::seconds runtime( RUNTIME );

	cout << std::setprecision(15);

	chrono::time_point<chrono::high_resolution_clock> hrc_now, hrc_prev;
	chrono::duration<double> hrc_delta;
	double hrc_delta_ms;
	bool prev_valid = false;

	map<float,int> raw_data;

	auto loop_start_time = chrono::high_resolution_clock::now();
	while(true)
	{
		hrc_now = chrono::high_resolution_clock::now();

		if (prev_valid)
		{
			hrc_delta = hrc_now - hrc_prev;
			hrc_delta_ms = hrc_delta.count() * 1000.0;

			// Round to the nearest 1/10 millisecond and increment the historgram count.
			raw_data[round(hrc_delta_ms * 10)]++;
		}

		hrc_prev = hrc_now;
		prev_valid = true;

		// Time to exit?
		if ( (hrc_now - loop_start_time) >= runtime )
			break;

		this_thread::sleep_for( delay );
	}

	// Fill in any missing data
	map<float,int> histogram;
	float prev = FLT_MIN, current;
	for (const auto& pair : raw_data)
	{
		current = pair.first;
		if (prev != FLT_MIN)
		{
			while (current > prev + 1)
			{
				histogram[++prev] = 0;
			}
		}
		prev = current;

		histogram[current] = pair.second;
	}

	// Write CSV file
	ofstream file;
	file.open("out.csv", ofstream::out | ofstream::trunc);

	cout << "Writing data to csv... ";

	// Write out column headers
	for (const auto& pair : histogram)
	{
		file << pair.first / 10 << "," << pair.second << endl;
	}

	cout << "done" << endl;

	return 0;
}
