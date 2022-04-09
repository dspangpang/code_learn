#include <iostream>
#include <string>
 
class MyClass
{
public:
	MyClass(int *p);
	MyClass(const MyClass &rhs);
	~MyClass();
 
	MyClass &operator=(const MyClass &rhs);
	void print();
 
private:
	int *ptr;
};
 
MyClass::MyClass(int *p)
{
	std::cout << "进入主构造器" << std::endl;
	ptr = p;
	std::cout << "离开主构造器" << std::endl;
}
 
MyClass::MyClass(const MyClass &rhs)
{
	std::cout << "进入副本构造器" << std::endl;
	this -> ptr = new int(*rhs.ptr);
	std::cout << "离开副本构造器" << std::endl;
}
 
MyClass::~MyClass()
{
	std::cout << "进入析构器" << *ptr <<std::endl;
	delete ptr;
	std::cout << "离开析构器" << std::endl;
}
 
void MyClass::print()
{
	std::cout << *ptr << std::endl;
}
 
int main()
{ 
	MyClass obj3(new int(3));
	MyClass obj4 = obj3;
	obj3.print();
	obj4.print();
    
    return 0;
}