#ifndef NNLAYER_HPP
#define NNLAYER_HPP


#include <iostream>
#include <eigen3/Eigen/Eigen>

float f_activation(float x);
float f_activation_d(float x);

class NNLayer {

    private:

        int input_size;
        int output_size;
        float learning_rate;
        std::string f_act = "none";

        Eigen::MatrixXf* pWeights;
        Eigen::RowVectorXf* pQValues;
        Eigen::RowVectorXf* pQValuesU;
        Eigen::RowVectorXf* pInput;

        // ADAM Optimizer
        Eigen::MatrixXf* m;
        Eigen::MatrixXf* v;
        float beta_1 = 0.9;
        float beta_2 = 0.999;
        int time = 1;
        float adam_epsilon = 0.00000001;

        bool bDebug;

    public:

        NNLayer(int input_size, int output_size, std::string activation = "none", float lr = 0.001, bool bDebug = false);
        ~NNLayer();
        void debug_mode(bool bdbg = false);

        Eigen::RowVectorXf* forward(Eigen::RowVectorXf& input);

        void update_weights(Eigen::MatrixXf& gradient);

        Eigen::RowVectorXf* backward(Eigen::RowVectorXf& gradient_from_above);

        void set_weights(Eigen::MatrixXf* pWeights1);
        Eigen::MatrixXf* get_weights();

        void update_time();
};



#endif