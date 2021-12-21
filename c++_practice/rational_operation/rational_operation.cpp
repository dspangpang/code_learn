#include<iostream>
#include<string>

#include<stdlib.h>

class Rational
{
private:
    void normalize();   

    int numerator;      //分子
    int denominator;    //分母
public:

    Rational(int num, int denom);

    Rational operator+ (Rational rhs);  // right hand side 
    Rational operator- (Rational rhs);
    Rational operator* (Rational rhs);
    Rational operator/ (Rational rhs);
    
    void print();
    ~Rational();
};

Rational::Rational(int num, int denom)
{
    numerator = num;
    denominator = denom;
}

Rational::~Rational()
{
}

/*
对分数进行简化操作
1、让符号始终在分子上
2、使用辗转相除法约分
*/
void Rational::normalize(){

    if(denominator < 0){
        numerator = - numerator;
        denominator = -denominator;
    }

    //辗转求余
    int a = abs(numerator);
    int b = abs(denominator);
    int tmp = 0;

    while(b > 0){
        tmp = a % b;
        a = b;
        b = tmp;
    }

    numerator /= a;
    denominator /= a;
}


void Rational::print(){

    normalize();
    if(numerator % denominator == 0){
        std::cout << numerator / denominator << "\n";
    }
    else{
        std::cout << numerator << "/" << denominator << "\n";
    }
}


Rational Rational::operator+ (Rational rhs){

    int a = numerator;
    int b = denominator;
    int c = rhs.numerator;
    int d = rhs.denominator;

    int e = a*d + c*b;
    int f = b * d ;

    return Rational(e, f);
}

Rational Rational::operator- (Rational rhs){

    int a = numerator;
    int b = denominator;
    int c = rhs.numerator;
    int d = rhs.denominator;

    int e = a*d - c*b;
    int f = b * d ;

    return Rational(e, f);
}

Rational Rational::operator* (Rational rhs){

    int a = numerator;
    int b = denominator;
    int c = rhs.numerator;
    int d = rhs.denominator;

    int e = a * c;
    int f = b * d ;

    return Rational(e, f);
}

Rational Rational::operator/ (Rational rhs){

    int a = numerator;
    int b = denominator;
    int c = rhs.numerator;
    int d = rhs.denominator;

    int e = a * d ;
    int f = b * c ;

    return Rational(e, f);
}

int main(){

    Rational num1(10, 2);
    Rational num2(1, 2);

    Rational res = num1 - num2;

    res.print();
    return 0;
}
