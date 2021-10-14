/*
    std::vector<std::vector<float> > layer0 = std::vector<std::vector<float> > (5, std::vector<float>(24,0));
    std::vector<std::vector<float> > layer1 = std::vector<std::vector<float> > (25, std::vector<float>(24,0));
    std::vector<std::vector<float> > layer2 = std::vector<std::vector<float> > (25, std::vector<float>(2,0));

    std::ifstream fp0("layer_0.csv");
    for (unsigned int i=0; i<layer0.size(); i++)
        for (unsigned int j=0; j<layer0[i].size(); j++)
            fp0 >> layer0[i][j];
    std::ifstream fp1("layer_1.csv");
    for (unsigned int i=0; i<layer1.size(); i++)
        for (unsigned int j=0; j<layer1[i].size(); j++)
            fp1 >> layer1[i][j];
    std::ifstream fp2("layer_2.csv");
    for (unsigned int i=0; i<layer2.size(); i++)
        for (unsigned int j=0; j<layer2[i].size(); j++)
            fp2 >> layer2[i][j];
    

    Eigen::MatrixXf* weights0 = new Eigen::MatrixXf(layer0.size(),layer0[0].size());
    Eigen::MatrixXf* weights1 = new Eigen::MatrixXf(layer1.size(),layer1[0].size());
    Eigen::MatrixXf* weights2 = new Eigen::MatrixXf(layer2.size(),layer2[0].size());

    for (int i=0; i<layer0.size(); i++)
        for (int j=0; j<layer0[i].size(); j++)
            weights0->coeffRef(i,j) = layer0[i][j];
    for (int i=0; i<layer1.size(); i++)
        for (int j=0; j<layer1[i].size(); j++)
            weights1->coeffRef(i,j) = layer1[i][j];
    for (int i=0; i<layer2.size(); i++)
        for (int j=0; j<layer2[i].size(); j++)
            weights2->coeffRef(i,j) = layer2[i][j];

    model[0]->set_weights(weights0);
    model[1]->set_weights(weights1);
    model[2]->set_weights(weights2);

    for (int i=0; i<model.size(); i++){
        Eigen::MatrixXf* weights = model[i]->get_weights();
        std::cout << std::endl << std::endl << "Weights " << i << std::endl;
        for (int r=0; r<weights->rows(); r++){
            for (int c=0; c<weights->cols(); c++){
                std::cout << " " << weights->coeffRef(r,c);
            }
            std::cout << std::endl;
        }
    }
*/




    /*
    std::vector<float> initstate = std::vector<float> (4,0);    
    std::ifstream fp0("init_state.csv");
    for (unsigned int i=0; i<initstate.size(); i++)
        fp0 >> initstate[i];

    x = initstate[0];
    xdot = initstate[1];
    theta = initstate[2];
    thetadot = initstate[3];
    */
    