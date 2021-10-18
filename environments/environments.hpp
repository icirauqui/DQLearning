#include <iostream>

#include "env_warehouse.cpp"
#include "env_cart_pole.cpp"
#include "env_mountain_car.cpp"
#include "env_mountain_car_cont.cpp"

class environments {

    private:
        env_warehouse* p_env_warehouse;
        env_cart_pole* p_env_cart_pole;
        env_mountain_car* p_env_mountain_car;
        env_mountain_car_cont* p_env_mountain_car_cont;

        std::string environment;


    public:
        environments(std::string env);
        ~environments();

        env_warehouse* get_p_env_warehouse();
        env_cart_pole* get_p_env_cart_pole();
        env_mountain_car* get_p_env_mountain_car();
        env_mountain_car_cont* get_p_env_mountain_car_cont();

};


environments::environments(std::string env){
    if (env == "warehouse")
        p_env_warehouse = new env_warehouse();
    else if (env == "cart_pole")
        p_env_cart_pole = new env_cart_pole();
    else if (env == "mountain_car")
        p_env_mountain_car = new env_mountain_car();
    else if (env == "mountain_car_cont")
        p_env_mountain_car_cont = new env_mountain_car_cont();

    environment = env;
}


environments::~environments(){}


env_warehouse* environments::get_p_env_warehouse(){
    return p_env_warehouse;
}


env_cart_pole* environments::get_p_env_cart_pole(){
    return p_env_cart_pole;
}


env_mountain_car* environments::get_p_env_mountain_car(){
    return p_env_mountain_car;
}


env_mountain_car_cont* environments::get_p_env_mountain_car_cont(){
    return p_env_mountain_car_cont;   
}
