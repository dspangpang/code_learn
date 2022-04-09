#include <iostream>
class Housemaid{

public:
    std::string name;
    double      height;
    double      weight;

    void sweep_home(void); 
};

void Housemaid::sweep_home(void){

    std::cout<<"I'm sweeping the house\n";
}

int main(int argc, char ** argv){
    
    Housemaid jzz;
    jzz.height = 1.75;
    jzz.name   = "jzz";
    jzz.weight = 86.2;
    std::cout<<"my name is "<<jzz.name<<" i'm"<<jzz.height<<"m "<<jzz.weight<<"kg\n" ;

    jzz.sweep_home();
    return 0;
}



