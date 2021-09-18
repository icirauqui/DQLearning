#include <iostream>
#include "DNN.cpp"

typedef float Scalar;

int main(){
    

    std::vector<int> topology = std::vector<int>{2,3,1};
    Scalar learning_rate = 0.5;

    DNN* pDNN1 = new DNN(topology,learning_rate);



    return 0;
}