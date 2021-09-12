#include <iostream>

#include "DNN.cpp"


class agentDQL{

    private:

        bool bDebug;

        // Main NN
        DNN *pDNN1;
        // Target NN
        DNN *pDNN2;


    public:


        agentDQL(std::vector<uint> topology, float learningRate);

        ~agentDQL();

        void debug_mode(bool bDebug = false);






};