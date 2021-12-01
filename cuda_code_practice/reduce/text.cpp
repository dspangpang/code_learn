#include<iostream>

int main(){
    int val = 4095;
    if(val & (val-1)){
        while (val & (val-1))
        {
            val &= (val-1); 
        }
        val <<= 1;
    }
    else {
        val = (val == 0)?(1):(val);
    }
        
    std::cout << val << std::endl;
}