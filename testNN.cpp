#include <iostream>
#include "DNN.cpp"
#include <eigen3/Eigen/Eigen>

typedef Eigen::RowVectorXf RowVector;

using namespace std;

DNN* pDNN = new DNN({2,10,10,4});

int main(){

    RowVector state(2);
    RowVector output(4);

    for (int i=0; i<state.size(); i++)
        state.coeffRef(i) = i+1;

    cout << "state =";
    for (int i=0; i<state.size(); i++)
        cout << " " << state.coeffRef(i);
    cout << endl;


    for (int i=0; i<10; i++){
        pDNN->forward(state,output);

        cout << "output =";
        for (int i=0; i<output.size(); i++)
            cout << " " << output.coeffRef(i);
        cout << endl;
    }






}