/*
Spencer Bledsoe
Basic Neural Network, Version 1

Summary - Neural network to act as a XOR operator (either 0 or 1)

Goals:
    - Function as a XOR operator when given two inputs "1" or "0".
    - 97% accuracy achieved with tanh transfer function

*/

#include "Net.h"
#include "TrainingData.h"

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cassert>
#include <string>


using namespace std;

void showVectorVals(string, vector <double>&);

int main() {

    TrainingData trainData("trainingData.txt");

    vector<double> inputVals, targetVals, resultVals;
    vector<unsigned> topology;
    int trainingPass = 0;

    trainData.getTopology(topology);

    Net myNet(topology);

    while (!trainData.isEof()) {
        ++trainingPass;
        cout << endl << "Pass " << trainingPass;

        //Get new input data and feed forward
        if (trainData.getNextInputs(inputVals) != topology[0]) {
            break;
        }
        showVectorVals(": Inputs:", inputVals);
        myNet.feedForward(inputVals);


        // After training, use network and collect the net's actual results:
        myNet.getResults(resultVals); // To actually use the neural network (provide output)
        showVectorVals("Outputs: ", resultVals);

        // Train the net on what outputs should have been:
        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals);

        //Report training performance, averaged over recent passes
        cout << "Net recent average error: " << myNet.getRecentAverageError() << endl;

    }

    cout << endl << "Completed." << endl;

    system("Pause");

}

void showVectorVals(string label, vector <double> &v) {
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }
   // cout << endl;
}