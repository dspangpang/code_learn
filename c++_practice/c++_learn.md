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

### 虚方法和抽象方法

 &emsp;&emsp;若一个方法的声明中含有 virtual 修饰符，则称该方法为虚方法。虚方法就是可以被子类重写的方法，如果子类重写了虚方法，则在运行时将运行重写的逻辑；如果子类没有重写虚方法，则在运行时将运行父类的逻辑. 在使用了 virtual 修饰符后，不允许再有 static、abstract 或者 override 修饰符。
 &emsp;&emsp;在实现一个多层次的类继承关系时，最顶层的基类应该只有虚方法。基类的析构器也是虚方法。
 &emsp;&emsp;在类中的虚方法后加上 ``` virtual void function() = 0; ```表明该方法为**抽象方法**，先不再定义里面实现。

作用：

1. 子类可以对父类进行扩展
2. 可以体现cpp的多态性,让程序清楚明了

特点：

1. 在虚方法前不能再有static,abstract以及override修饰符.
2. 不能在声明虚方法的同时指定重写虚方法,也就是说不能在虚方法前再添加override修饰符，因为虚方法在基类中声明的，所以不可再重写.
3. 虚方法不可为私有，由于在子类中要被继承，所以不能有private修饰

&emsp;&emsp;有了虚函数，基类指针指向基类对象时就使用基类的成员（包括成员函数和成员变量），指向子类对象时就使用子类的成员。
&emsp;&emsp;换句话说，基类指针可以按照基类的方式来做事，也可以按照派生类的方式来做事，它有多种形态，或者说有多种表现方式，我们将这种现象称为**多态**,同一条语句可以执行不同的操作，看起来有不同表现方式，这就是多态，所以体现了多态性.
[程序举例](./c++_practice/virtual_function/virtual.cpp)

### 运算符的重载

存在五个运算符不可以重载：

* **.** (成员访问符号)
  
* **.***(成员指针访问符)
  
* **::**(域运算符)

* **sizeof**(尺寸运算符)

* **?:**(条件运算符)

注意事项：

* 重载不能改变运算符的**运算对象**
* 重载不能改变运算符的**运算级别**
* 重载不能改变运算符的**结合性**
* 重载运算符函数**不能有默认参数**
* 重载的运算符必须和用户定义的自定义类型的对象一起使用，**参数不能全是C++的的标准类型**
  
### 多重继承和虚继承

#### 多重继承

一个类继承于多个类叫做多重继承。

```c++
class A{};
 
class B {};
 
class C :public B, public A //多重继承
{
};
```

##### 菱形继承

**概念**：A作为基类，B和C都继承与A。最后一个类D又继承于B和C，这样形式的继承称为菱形继承。

&emsp;&emsp;&emsp;&emsp;&emsp;![分段扫描举例](./imges/菱形继承.png)

**菱形继承存在缺点**:

* **数据冗余**：在D中会保存两份A的内容。
* **访问不明确（二义性）**：因为D不知道是以B为中介去访问A还是以C为中介去访问A，因此在访问某些成员的时候会发生二义性
  
**解决方式**:

* **数据冗余**：使用虚继承的方式。
* **访问不明确（二义性）**：通过作用域访问符::来明确调用。虚继承也可以解决这个问题。[示例](./c++_practice/Multiple_Inheritance/mul_inherite.cpp)

### 虚继承

* **虚继承的作用**：为了保证公共继承对象在创建时只保存一分实例
* **虚继承解决了菱形继承的两个问题**：
  * **数据冗余**：顶级基类在整个体系中只保存了一份实例
  * **访问不明确（二义性）**：可以不通过作用域访问符::来调用（原理就是因为顶级基类在整个体系中只保存了一份实例）[示例](./c++_practice/Multiple_Inheritance/mul_inherite_virtual.cpp)

### 副本构造器

&emsp;&emsp;当删除其中一个对象时，它包含的指针也将被删除，但万一此时另一个副本（对象）还在引用这个指针，就会出问题。

```c++
MyClass obj1;
MyClass obj2;
obj2 = obj1;
```

**解决办法**：

* 重载“=”操作符 [示例](./c++_practice/copy_constructor/override=.cpp)
* 亲自定义个副本构造器(系统会自动生成逐一复制的副本构造器)```MyClass(const MyClass &rhs);```[示例](./c++_practice/copy_constructor/copy_constuctor.cpp)

### 模板

* 模板是泛型编程的基础，泛型编程即以一种独立于任何特定类型的方式编写代码。
* 模板是创建泛型类或函数的蓝图或公式。库容器，比如迭代器和算法，都是泛型编程的例子，它们都使用了模板的概念。
* 可以使用模板来定义函数和类，接下来让我们一起来看看如何使用。

#### 函数模板

```c++
template <typename T> ret-type func-name(parameter list)
{
   // 函数的主体
}
```

T是函数所使用的数据类型的占位符名称。这个名称可以在函数定义中使用。

#### 类模板

就像定义函数模板一样，也可以定义类模板。泛型类声明的一般形式如下所示：

```c++
template <class type> class class-name {

}
```

以操作**栈**举例

```c++
template <class T>
class Stack { 
  private: 
    vector<T> elems;     // 元素 
 
  public: 
    void push(T const&);  // 入栈
    void pop();               // 出栈
    T top() const;            // 返回栈顶元素
    bool empty() const{       // 如果为空则返回真。
        return elems.empty(); 
    } 
}; 
 
template <class T>
void Stack<T>::push (T const& elem) 
{ 
    // 追加传入元素的副本
    elems.push_back(elem);    
} 
 
template <class T>
void Stack<T>::pop () 
{ 
    if (elems.empty()) { 
        throw out_of_range("Stack<>::pop(): empty stack"); 
    }
    // 删除最后一个元素
    elems.pop_back();         
} 
 
template <class T>
T Stack<T>::top () const 
{ 
    if (elems.empty()) { 
        throw out_of_range("Stack<>::top(): empty stack"); 
    }
    // 返回最后一个元素的副本 
    return elems.back();      
} 

```
