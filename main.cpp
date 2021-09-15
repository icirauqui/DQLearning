#include <iostream>

#include "environment.hpp"
#include "agentDQL.cpp"




int main(){

    environment* pEnv = new environment({11,11});
    agentDQL* pAgent = new agentDQL(pEnv,{10,10,4},0.5,0.5,0.5);

    pAgent->train(1000000,100);

    return 0;
}



