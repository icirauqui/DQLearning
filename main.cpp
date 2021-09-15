#include <iostream>


#include "agentDQL.cpp"
//#include "environment.hpp"



void testfunc(int &a){
    a +=1;
}


int main(){

    //environment* pEnv = new environment({11,11});
    //agentDQL* pAgent = new agentDQL(pEnv,{10,10,4},0.5,0.5,0.5);
    //pAgent->train(1000000,100);

    int b = 1;

    std::cout << b << std::endl;
    testfunc(b);
    std::cout << b << std::endl;


    return 0;
}



