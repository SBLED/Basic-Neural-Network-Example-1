#pragma once
#include <vector>
#include "Neuron.h"


class Net {

private:
	std::vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
	double m_error, m_recentAverageError;
	static double m_recentAverageSmoothingFactor;

public:
	Net(const std::vector<unsigned>& topology);

	void feedForward(const std::vector<double>& inputVals);
	void backProp(const std::vector<double>& targetVals);
	void getResults(std::vector<double>& resultVals) const;
	double getRecentAverageError(void) const { return m_recentAverageError; } // Check me

};