#include "NNLayer.hpp"




float f_activation(float x){ if (x>0) return x; else return 0; }
float f_activation_d(float x){ if (x>0) return 1; else return 0; }

float f_identity(float x){ return x; }
float f_identity_d(float x){ return 1; }

float f_relu(float x){ if (x>0) return x; else return 0; }
float f_relu_d(float x){ if (x>0) return 1; else return 0; }

float f_sigmoid(float x){ return 1 / (1 + exp(-x)); }
float f_sigmoid_d(float x){ float sigmoid = 1 / (1 + exp(-x)); return sigmoid*(1-sigmoid); }



NNLayer::NNLayer(int input_size, int output_size, std::string activation, float lr, bool bDebug){
    this->input_size = input_size;
    this->output_size = output_size;
    this->learning_rate = lr;
    this->f_act = activation;

    // Initialize weights to random in range {-0.5,0.5}
    pWeights =  new Eigen::MatrixXf(input_size, output_size);
    pWeights->setRandom();
    *pWeights = (*pWeights + Eigen::MatrixXf::Constant(pWeights->rows(),pWeights->cols(),1.))*1./2.;
    *pWeights = (*pWeights + Eigen::MatrixXf::Constant(pWeights->rows(),pWeights->cols(),-0.5));

    pQValues = new Eigen::RowVectorXf(output_size);
    pQValues->setZero();
    pQValuesU = new Eigen::RowVectorXf(output_size);
    pQValuesU->setZero();

    // ADAM Optimizer
    m = new Eigen::MatrixXf(input_size,output_size);
    m->setZero();
    v = new Eigen::MatrixXf(input_size,output_size);
    v->setZero();
}


NNLayer::~NNLayer(){}


void NNLayer::debug_mode(bool bdbg){
    this->bDebug = bDebug;
}


Eigen::RowVectorXf* NNLayer::forward(Eigen::RowVectorXf& input){
    Eigen::RowVectorXf input_with_bias = Eigen::RowVectorXf(input.size()+1);
    for (unsigned int i=0; i<input.size(); i++)
        input_with_bias.coeffRef(i) = input.coeffRef(i);
    input_with_bias.coeffRef(input.size()) = 1;
    pInput = new Eigen::RowVectorXf(input.size()+1);
    *pInput = input_with_bias;

    (*pQValues) = input_with_bias * (*pWeights);

    *pQValuesU = *pQValues;

    //if (f_act != "none")
    //    (*pQValues) = pQValues->unaryExpr(std::ptr_fun(f_activation));

    if (f_act == "identity")
        (*pQValues) = pQValues->unaryExpr(std::ptr_fun(f_identity));
    else if (f_act == "relu")
        (*pQValues) = pQValues->unaryExpr(std::ptr_fun(f_relu));
    else if (f_act == "sigmoid")
        (*pQValues) = pQValues->unaryExpr(std::ptr_fun(f_sigmoid));

    return pQValues;
}



float f_identity(float x){ return x; }
float f_identity_d(float x){ return 1; }

float f_relu(float x){ if (x>0) return x; else return 0; }
float f_relu_d(float x){ if (x>0) return 1; else return 0; }

float f_sigmoid(float x){ return 1 / (1 + exp(-x)); }
float f_sigmoid_d(float x){ float sigmoid = 1 / (1 + exp(-x)); return sigmoid*(1-sigmoid); }


/*
void NNLayer::update_weights(Eigen::MatrixXf& gradient){
    (*pWeights) = (*pWeights) - (learning_rate*gradient);
}
*/


void NNLayer::update_weights(Eigen::MatrixXf& gradient){
    Eigen::MatrixXf m_temp = Eigen::MatrixXf(m->rows(),m->cols());
    Eigen::MatrixXf v_temp = Eigen::MatrixXf(v->rows(),v->cols());
    Eigen::MatrixXf m_vec_hat = Eigen::MatrixXf(m->rows(),m->cols());
    Eigen::MatrixXf v_vec_hat = Eigen::MatrixXf(v->rows(),v->cols());

    m_temp = *m;
    v_temp = *v;

    m_temp = beta_1*m_temp + (1-beta_1)*gradient;
    
    Eigen::MatrixXf gradient2 = Eigen::MatrixXf(gradient.rows(),gradient.cols());
    for (unsigned int i=0; i<gradient.rows(); i++)
        for (unsigned int j=0; j<gradient.cols(); j++)
            gradient2.coeffRef(i,j) = gradient.coeffRef(i,j) * gradient.coeffRef(i,j);
    v_temp = beta_2*v_temp + (1-beta_2)*gradient2;
    
    m_vec_hat = m_temp / (1-pow(beta_1,time+0.1));
    v_vec_hat = v_temp / (1-pow(beta_2,time+0.1));


    Eigen::MatrixXf weights_temp = Eigen::MatrixXf(pWeights->rows(),pWeights->cols());


    for (unsigned int i=0; i<pWeights->rows(); i++)
        for (unsigned j=0; j<pWeights->cols(); j++)
            pWeights->coeffRef(i,j) -= (learning_rate * m_vec_hat.coeffRef(i,j)) / (sqrt(v_vec_hat.coeffRef(i,j)) + adam_epsilon);
    
    *m = m_temp;
    *v = v_temp;
}



Eigen::RowVectorXf* NNLayer::backward(Eigen::RowVectorXf& gradient_from_above){
    Eigen::RowVectorXf adjusted_mul = Eigen::RowVectorXf(gradient_from_above.size());
    adjusted_mul = gradient_from_above;

    Eigen::RowVectorXf pQValues_temp = Eigen::RowVectorXf(pQValuesU->size());
    pQValues_temp = *pQValuesU;
    if (f_act != "none"){
        //pQValues_temp = pQValuesU->unaryExpr(std::ptr_fun(f_activation_d));
        if (f_act == "identity")
            pQValues_temp = pQValuesU->unaryExpr(std::ptr_fun(f_identity_d));
        else if (f_act == "relu")
            pQValues_temp = pQValuesU->unaryExpr(std::ptr_fun(f_relu_d));
        else if (f_act == "sigmoid")
            pQValues_temp = pQValuesU->unaryExpr(std::ptr_fun(f_sigmoid_d));

        for (unsigned int i=0; i<adjusted_mul.size(); i++)
            adjusted_mul.coeffRef(i) = (pQValues_temp.coeffRef(i)) * gradient_from_above.coeffRef(i);
    }
    
    Eigen::RowVectorXf* delta_i = new Eigen::RowVectorXf(pInput->size()-1);
    Eigen::RowVectorXf* delta_i_temp = new Eigen::RowVectorXf(input_size);
    (*delta_i_temp) = adjusted_mul * (pWeights->transpose());
    for (int i=0; i<delta_i->size(); i++)
        delta_i->coeffRef(i) = delta_i_temp->coeffRef(i);
    

    Eigen::MatrixXf* D_i = new Eigen::MatrixXf(pInput->size(),adjusted_mul.size());
    for (unsigned int i=0; i<pInput->size(); i++){
        for (unsigned int j=0; j<adjusted_mul.size(); j++){
            D_i->coeffRef(i,j) = pInput->coeffRef(i) * adjusted_mul.coeffRef(j);
        }
    }

    update_weights(*D_i);

    return delta_i;    
}


void NNLayer::set_weights(Eigen::MatrixXf* pWeights1){
    *pWeights = *pWeights1;
}


Eigen::MatrixXf* NNLayer::get_weights(){
    return pWeights;
}

void NNLayer::update_time(){
    this->time += 1;
}


