#include<iostream>
#include<string.h>

class Lovers
{
private:
    /* data */
public:
    Lovers(std::string theName);
    ~Lovers();

    void kiss(Lovers *lover);
    void ask(Lovers *lover, std::string something);

protected:
    std::string name;

    friend class Others;

};

Lovers::Lovers(std::string theName)
{
    name = theName;
}

Lovers::~Lovers()
{
}

void Lovers::kiss(Lovers * lovers)
{
    std::cout << "亲亲我们家的" << lovers->name << std::endl;
}

void Lovers::ask(Lovers * lovers, std::string something)
{
    std::cout << "宝贝" << lovers->name << "帮我" << something << std::endl;
}

class Boyfirend : public Lovers
{
public:
    Boyfirend(std::string theName);
};

Boyfirend::Boyfirend(std::string theName):Lovers(theName){

}
class Girlfirend : public Lovers
{
public:
    Girlfirend(std::string theName);
};

Girlfirend::Girlfirend(std::string theName):Lovers(theName){
    
}

class Others
{
public:
    Others(std::string theName);
    void kiss(Lovers *lover);
protected:
    std::string name;
};

Others::Others(std::string theName){
    name = theName;
}

void Others::kiss(Lovers * lovers)
{
    std::cout << "亲亲我们家的" << lovers->name << std::endl;
}

int main(){
    Boyfirend boyfirend("A君");
    Girlfirend girlfirend("B妞");

    Others others("路人丁");

    girlfirend.kiss(&boyfirend);
    girlfirend.ask(&boyfirend, "洗衣服啦");

    std::cout << "传说中的路人丁来了" << std::endl;

    others.kiss(&boyfirend);
}