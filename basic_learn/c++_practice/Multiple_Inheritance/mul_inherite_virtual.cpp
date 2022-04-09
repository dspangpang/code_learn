#include<iostream>

#include<string>

class Person
{
private:
    std::string name;
public:
    Person(std::string theName);
    ~Person();
    void introduce();
};

Person::Person(std::string theName)
{
    name = theName;
}

Person::~Person()
{

}

void Person::introduce(){

    std::cout << "my name is " << name << std::endl;
}

class Teacher : virtual public Person
{
private:
    std::string classes;

public:
    Teacher(std::string theName, std::string teachClass);
    ~Teacher();
    void teach();
};

Teacher::Teacher(std::string theName, std::string teachClass):Person(theName)
{
    classes = teachClass;
}

Teacher::~Teacher()
{

}

void Teacher::teach(){

    std::cout << "i'm teaching the " << classes << std::endl;
}

class Student : virtual public Person
{
private:
    std::string classes;
public:
    Student(std::string theName, std::string attendClass);
    ~Student();
    void attendclass();
};

Student::Student(std::string theName, std::string attendClass):Person(theName)
{
    classes = attendClass;
}

Student::~Student()
{

}

void Student::attendclass(){

    std::cout << "i'm attening the " << classes << std::endl;
}

class TeachingStudent : public Teacher, public Student
{
private:
    std::string teachClass;
    std::string attendClass;
public:
    TeachingStudent(std::string theName, std::string teachClass, std::string attendclass);
    ~TeachingStudent();

};

TeachingStudent::TeachingStudent(std::string theName, std::string teachClass, std::string attendClass) : Teacher(theName, teachClass), Student(theName, attendClass),Person(theName)
{

}

TeachingStudent::~TeachingStudent()
{
}


int main(){

    Teacher theacher ("jzz", "1班");
    Student student("yfn","1班");
    TeachingStudent theachingstudent("pig", "1班", "1班");

    theacher.introduce();
    theacher.teach();
    student.introduce();
    student.attendclass();

    theachingstudent.introduce(); //因为使用了虚继承只存在一个Person类，不会发生冲突
    theachingstudent.teach();
    theachingstudent.attendclass();

    return 0;
}