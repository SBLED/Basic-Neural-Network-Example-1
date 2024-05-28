#pragma once
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

// Reads training data from file

class TrainingData {
private:
	std::ifstream m_trainingDataFile;

public:
	TrainingData(const std::string filename);
	bool isEof(void) { return m_trainingDataFile.eof(); }
	void getTopology(std::vector<unsigned>& topology);

	//Returns the number of inputVals read from the file:
	unsigned getNextInputs(std::vector<double>& inputVals);
	unsigned getTargetOutputs(std::vector<double>& targetOutputVals);

};




TrainingData::TrainingData(const std::string filename) {
	m_trainingDataFile.open(filename.c_str());
}

void TrainingData::getTopology(std::vector<unsigned>& topology) {

	std::string line, label;

	std::getline(m_trainingDataFile, line);
	std::stringstream ss(line);

	ss >> label;

	if (this->isEof() || label.compare("topology:") != 0) {
		abort();
	}

	while (!ss.eof()) {
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}

	return;

}

unsigned TrainingData::getNextInputs(std::vector<double>& inputVals) {
	inputVals.clear();

	std::string line, label;
	std::getline(m_trainingDataFile, line);
	std::stringstream ss(line);

	ss >> label;
	if (label.compare("in:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			inputVals.push_back(oneValue);
		}
	}

	return inputVals.size();

}

unsigned TrainingData::getTargetOutputs(std::vector<double>& targetOutputVals) {

	targetOutputVals.clear();

	std::string line, label;
	std::getline(m_trainingDataFile, line);
	std::stringstream ss(line);

	ss >> label;
	if (label.compare("out:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			targetOutputVals.push_back(oneValue);
		}
	}

	return targetOutputVals.size();
}