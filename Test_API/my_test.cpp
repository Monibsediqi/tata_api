
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <map>

using namespace std;


#define PRINT(x) std::cout << x << std::endl;
#define INFO(x) std::cout << "INFO: " << x << std::endl;
#define ERR(x) std::cerr << "ERROR: " << x << std::endl;


struct model_config {
    std::string path;
    std::string weight_name;
};

class A {
protected:
    model_config m_config;

public:
    A(const model_config& c) : m_config(c) {}

    void set_a(const model_config& c) {
        m_config = c;
    }

    model_config get_a() const {
        return m_config;
    }
};

class B : public A {
public:
    B(const model_config& c) : A(c) {}

    void run() {
        std::cout << "B::run()" << std::endl;
        std::cout << "b config path: " << m_config.path << std::endl;
        std::cout << "b config weight name: " << m_config.weight_name << std::endl;
    }
};

int main() {
    std::cout << "hello world" << std::endl;

    model_config c;
    c.path = "path to lung regression";
    c.weight_name = "lung_regression";

    A a_obj(c);
    B b_obj(c);

    b_obj.run();

    return 0;
}