# C++学习笔记

## 基础知识

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
