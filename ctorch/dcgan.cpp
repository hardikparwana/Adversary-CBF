#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <sys/time.h>
#include <ctime>

using std::cout; using std::endl;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

torch::Tensor foo( torch::Tensor x, torch::Tensor y){
	torch::Tensor r;
    // if ( x.max() > y.max() ){
	// 	r  = x;
	// }
	// else
	// 	r  = y;
	// std::cout << "hello?" << std::endl;
	r = x;
	return r;
}


torch::Tensor bar2(torch::Tensor x, torch::Tensor y, torch::Tensor z){
	for (int i=0; i<10; i++){
		for (int j=0; j<10; j++){
			z = z + i / (j+1) * torch::ones(3);
		}
	}
	return foo(x,y) + z;
}

int main(){
	torch::Tensor tensor = torch::eye(3);
	std::cout << tensor << std::endl;

	auto x = torch::ones({2, 2}, torch::requires_grad());

	auto y = x + 2;
	std::cout << y.grad_fn()->name() << std::endl;

	auto z = y * y * 3;
	auto out = z.mean();

	std::cout << z << std::endl;
	std::cout << z.grad_fn()->name() << std::endl;
	std::cout << out << std::endl;
	std::cout << out.grad_fn()->name() << std::endl;

	auto a = torch::randn({2, 2});
	a = ((a * 3) / (a - 1));
	std::cout << a.requires_grad() << std::endl;

	a.requires_grad_(true);
	std::cout << a.requires_grad() << std::endl;

	auto b = (a * a).sum();
	std::cout << "b grad fn" << b.grad_fn()->name() << std::endl;

	out.backward();

	std::cout << "x grad" << x.grad() << std::endl;

	auto x2 = torch::randn(3, torch::requires_grad());

	auto y2 = x2 * 2;
	while (y2.norm().item<double>() < 1000) {
	y2 = y2 * 2;
	}

	

	std::cout << "y2: " << y2 << std::endl;
	std::cout << y2.grad_fn()->name() << std::endl;

	auto v = torch::tensor({0.1, 1.0, 0.0001}, torch::kFloat);
	y2.backward(v);

	std::cout << x2.grad() << std::endl;

	auto t0 = std::chrono::system_clock::now();
	bar2( torch::zeros(3), torch::zeros(3), torch::ones(3) );
	auto tf = std::chrono::system_clock::now();
	std::cout << "time spent " << std::chrono::duration_cast<std::chrono::microseconds>(tf-t0).count()/1000000.0 << std::endl;

	auto t02 = std::chrono::system_clock::now();
	bar2( torch::zeros(3), torch::zeros(3), torch::ones(3) );
	auto tf2 = std::chrono::system_clock::now();
	std::cout << "time spent " << std::chrono::duration_cast<std::chrono::microseconds>(tf2-t02).count()/1000000.0 << std::endl;
}


