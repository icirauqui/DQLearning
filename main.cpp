#include <iostream>


#include "agentDQL.cpp"




int main(){

    agentDQL* pAgent = new agentDQL({11,11},{10,10,4},0.5,0.5,0.5);

    pAgent->train(1000000,1000,100,4);
 
    return 0;
    
}



