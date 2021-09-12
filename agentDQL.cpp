#include "agentDQL.hpp"






agentDQL::agentDQL(std::vector<uint> topology, float learningRate){

    pDNN1 = new DNN(topology,learningRate);
    pDNN2 = new DNN(topology,learningRate);







}


agentDQL::~agentDQL(){}




void agentDQL::debug_mode(bool bDebug){
    this->bDebug = bDebug;
}


