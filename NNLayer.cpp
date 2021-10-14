#include "NNLayer.hpp"




float f_activation(float x){
    if (x>0) return x;
    else return 0;
}

float f_activation_d(float x){
    if (x>0) return 1;
    else return 0;
}




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

    //std::cout << "NNLayer = " << input_size << " " << output_size << " " << f_act << std::endl;
    
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

    //std::cout << "Input           = " << input << std::endl;
    //std::cout << "Input with bias = " << input_with_bias << std::endl;

    (*pQValues) = input_with_bias * (*pWeights);
    //for (int i=0; i<pQValues->size(); i++){
    //    pQValues->coeffRef(i) = input_with_bias * pWeights->col(i);
    //}

    //std::cout << "f_act = " << f_act << std::endl;
    *pQValuesU = *pQValues;
    if (f_act != "none")
        (*pQValues) = pQValues->unaryExpr(std::ptr_fun(f_activation));

    //std::cout << "pQValuesU = " << *pQValuesU << std::endl;
    //std::cout << "pQValues  = " << *pQValues << std::endl;

    return pQValues;
}

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

    //std::cout << std::endl;
    m_temp = *m;
    v_temp = *v;

    //std::cout << " m " << m_temp.rows() << "/" << m_temp.cols() << "   gradient = " << gradient.rows() << "/" << gradient.cols() << std::endl;
    m_temp = beta_1*m_temp + (1-beta_1)*gradient;
    //std::cout << "m_temp = " << m_temp << std::endl;
    
    //std::cout << " v " << v_temp.rows() << "/" << v_temp.cols() << "   gradient = " << gradient.rows() << "/" << gradient.cols() << std::endl;
    Eigen::MatrixXf gradient2 = Eigen::MatrixXf(gradient.rows(),gradient.cols());
    for (unsigned int i=0; i<gradient.rows(); i++)
        for (unsigned int j=0; j<gradient.cols(); j++)
            gradient2.coeffRef(i,j) = gradient.coeffRef(i,j) * gradient.coeffRef(i,j);
    v_temp = beta_2*v_temp + (1-beta_2)*gradient2;
    //std::cout << "v_temp = " << v_temp << std::endl;
    
    //std::cout << " c " << std::endl;

    m_vec_hat = m_temp / (1-pow(beta_1,time+0.1));
    //std::cout << "m_vec_hat = " << m_vec_hat << std::endl;
    v_vec_hat = v_temp / (1-pow(beta_2,time+0.1));
    //std::cout << "v_vec_hat = " << v_vec_hat << std::endl;


    Eigen::MatrixXf weights_temp = Eigen::MatrixXf(pWeights->rows(),pWeights->cols());


    for (unsigned int i=0; i<pWeights->rows(); i++)
        for (unsigned j=0; j<pWeights->cols(); j++)
            pWeights->coeffRef(i,j) -= (learning_rate * m_vec_hat.coeffRef(i,j)) / (sqrt(v_vec_hat.coeffRef(i,j)) + adam_epsilon);
    //std::cout << "weights =" << *pWeights << std::endl;
    //std::cout << learning_rate << std::endl;
    //std::cout << adam_epsilon << std::endl;
    

    *m = m_temp;
    *v = v_temp;

    //std::cout << "m =" << *m << std::endl;
    //std::cout << "v =" << *v << std::endl;
}



Eigen::RowVectorXf* NNLayer::backward(Eigen::RowVectorXf& gradient_from_above){
    Eigen::RowVectorXf adjusted_mul = Eigen::RowVectorXf(gradient_from_above.size());
    adjusted_mul = gradient_from_above;

    Eigen::RowVectorXf pQValues_temp = Eigen::RowVectorXf(pQValuesU->size());
    pQValues_temp = *pQValuesU;
    if (f_act != "none"){
        pQValues_temp = pQValuesU->unaryExpr(std::ptr_fun(f_activation_d));
        for (unsigned int i=0; i<adjusted_mul.size(); i++)
            adjusted_mul.coeffRef(i) = (pQValues_temp.coeffRef(i)) * gradient_from_above.coeffRef(i);
    }
    
    //std::cout << "pQValuesU     = " << *pQValuesU << std::endl;
    //std::cout << "pQValues_temp = " << pQValues_temp << std::endl;
    
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


