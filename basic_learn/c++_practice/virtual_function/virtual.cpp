#include<cctype>
#include<iostream>
#include<string>
class person
{
    public:
        person(std::string name);
        virtual void eat();     //时候存在virtual会影响程序的结果
    protected:
        std::string name;
};

person::person(std::string name)
{
    std::cout << "你好！我叫" << name << std::endl;
    this->name = name;
}

void person::eat()
{
    std::cout << name << "开始吃饭" << std::endl;
}

class men : public person
{
    public:
        men(std::string name);
        void eat();
};

men::men(std::string name) : person(name)
{

}

void men::eat()
{
    std::cout << name <<"吃了十碗饭!" << std::endl;
}

class Women : public person
{
    public:
        Women(std::string name);
        void eat();
};

Women::Women(std::string name) : person(name)
{

}

void Women::eat()
{
    std::cout << name <<"吃了五碗饭!" << std::endl;
}


int main(void)
{
    // men m("张三");
    // Women wm("如花");              //一般调用方法
    person *m = new men("张三");
    // men *m = new men("张三");      //用new来进行空间开创
    person *fm = new Women("如花");
    m->eat();                         //因为是用基类来定义一个指针变量，为了效率会只编译基类定义的函数而不是将子类所覆盖的一同编译，所以需要virtual
    fm->eat();
    delete m;
    delete fm;
    return 0;
}
