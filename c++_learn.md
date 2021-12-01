# C++学习笔记

## 基础知识

### this指针

&emsp;&emsp; this指针是类自动生成，自动隐藏的私有成员，它存在于类的非静态成员函数中，指向被调用函数所在对象的地址。当一个对象被创建的时候，该对象的this指针指向对象数据的首地址。

```c++
class Point
{
private:
    int x,y;
public:
    Point(int a, int b){
        x = a;
        y = b; 
    }

//实际上函数的原型为 void MovePoint( Point * this, int a, int b)
    void MovePoint(int a, int b){  
        x = a;
        y = b; //实际应为 this->x = a;
    }
    void print(){
        std::cout << "x = " << x << " y = " << y <<std::endl;
    }
};
int main(){

    Point point1(10, 10); //此时point1的地址传给了this
    point1.MovePoint(2, 2);//通过指针的方式修改了私有变量的值
    point1.print();

    return 0;
}
```

### 友元关系

友元关系是类之间的一种特殊关系，可以友元类访问对方类的所有方法和属性

```c++
    class Theone{
    ...
        frend class Others;
    };
    
    ...
    class Others{

    }; //即 Others 可以访问 Theone 里面的方法和属性
```

注：这条语句可以放在任何地方，放在public，protected，private段落里都可以。

### 静态属性和静态方法

&emsp;&emsp;C++允许把一个类里面的多个声明为属于某个类，而不仅属于该类的对象，这些成员可以在类的所有对象之间共享。

1. 静态成员是所有对象共享的，所以不能在静态方法里面访问非静态的元素（**无法访问this指**针）
2. 非静态的方法可以访问类的静态成员，也可以访问类的非静态成员
3. 调用静态方法时，使用类的名字调用，而不用对象的名字调用

```c++
class Pet
{ //可以通过调用函数petCount来读取私,甚至修改有变量cnt的值
private:
    static int Cnt; //需要为其创造一个内存区域并且初始化，内存中是和全局变量存放在一起
public:
    Pet(std::string theName);
    ~Pet();
    static int petCount();
protected:
    std::string name;
};
int Pet::Cnt = 0; //静态变量的初始化，分配内存
```