#include <fstream>
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <cfloat>	//FLT_MIN
#include <math.h>	//round()
#include <iomanip> 	//setprecision()

#define FPS 30.0
#define RUNTIME 300 //sec

using namespace std;

int main(int argc, char** argv)
{
	const chrono::microseconds delay( (unsigned long long)( (1.0/FPS) * 1000000.0 ) );
	const chrono::seconds runtime( RUNTIME );

	cout << std::setprecision(7);

	chrono::time_point<chrono::high_resolution_clock> hrc_now, hrc_prev;
	chrono::duration<double> hrc_delta;
	double hrc_delta_ms;
	bool prev_valid = false;

	chrono::duration<double> sleep_time = delay;

	vector<double> samples;
	vector<double> graph;
	int last_update_second = 0;

	samples.reserve(FPS*RUNTIME+1);
	graph.reserve(RUNTIME+1);

	auto loop_start_time = chrono::high_resolution_clock::now();
	while(true)
	{
		hrc_now = chrono::high_resolution_clock::now();

		if (prev_valid)
		{
			hrc_delta = hrc_now - hrc_prev;
			sleep_time = sleep_time - hrc_delta;
			sleep_time += delay;

			hrc_delta_ms = hrc_delta.count() * 1000.0;
			samples.push_back(hrc_delta_ms);
		}

		hrc_prev = hrc_now;
		prev_valid = true;
		auto loop_duration = hrc_now - loop_start_time;

		// Add a sample of the average each second
		int loop_duration_sec = (int)(loop_duration.count() / 1000000000.0);
		if ( loop_duration_sec > last_update_second)
		{
			double average = 0.0;
			for (auto &s : samples)
				average += s;
			average /= samples.size();
			graph.push_back(average);
			last_update_second = loop_duration_sec;

			cout << "average: " << average << " ms" << endl;
		}

		// Time to exit?
		if ( loop_duration >= runtime )
			break;

		this_thread::sleep_for( chrono::duration_cast<chrono::microseconds>(sleep_time) );
	}

	// Write CSV file
	ofstream file;
	file.open("correction.csv", ofstream::out | ofstream::trunc);
	file << std::setprecision(5);

	cout << "Writing data to csv... ";

	// Write out averages
	for (const auto &avg : graph)
	{
		file << avg << "," << endl;
	}

	cout << "done" << endl;

	return 0;
}
