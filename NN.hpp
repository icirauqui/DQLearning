#ifndef NN_HPP
#define NN_HPP


#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <vector>

#include "NNLayer.cpp"


class NN{

    private:

        std::vector<int> topology;
        float learningRate;

        bool bDebug = false;

    public:
        NN(std::vector<int> topology, float learningRate = float(0.005), bool bDebug = false);
        ~NN();
        void debug_mode(bool bdbg = false);

        void forward(Eigen::RowVectorXf& input, Eigen::RowVectorXf& output);

        void backward(Eigen::RowVectorXf& actions, Eigen::RowVectorXf& experimentals);

        void update_time();


        std::vector<NNLayer*> model;

};






#endif