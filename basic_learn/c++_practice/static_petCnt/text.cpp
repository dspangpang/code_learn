#include<iostream>
#include<string>

class Pet
{
private:

    static int Cnt;
    

public:

    Pet(std::string theName);
    ~Pet();
    static int petCount();
protected:

    std::string name;

};

int Pet::Cnt = 0;
 Pet::Pet(std::string theName)
{
    name = theName;
    Cnt++;

    std::cout << "有一只宠物出生了，名字叫作 " << name << std::endl;
}
 Pet::~Pet()
{
    Cnt--;

    std::cout << "有一只宠物死了，名字叫作 " << name << std::endl;
}

int Pet::petCount(){
    return Cnt;
}

class Dog : public Pet
{
private:
    /* data */
public:
    Dog(std::string theName);
    ~Dog();
};

Dog::Dog(std::string theName):Pet(theName)
{
}

Dog::~Dog()
{
}


class Cat : public Pet
{
private:
    /* data */
public:
    Cat(std::string theName);
    ~Cat();
};

Cat::Cat(std::string theName):Pet(theName)
{
}

Cat::~Cat()
{
}

int main(){

    Dog dog("tom");
    Cat cat("jerry");

    std::cout << "已经生出了" << Pet::petCount() << "只宠物" <<std::endl;

    { //通过构建区域的方法让这些类，在该区域结束的时候完成析构
        Dog dog_2("tom_2");
        Cat cat_2("jerry_2");
        std::cout << "已经生出了" << Pet::petCount() << "只宠物" <<std::endl;
    
    }
    std::cout << "现在还剩下" << Pet::petCount()<< "只宠物" <<std::endl;
    return 0;
}