import torch
import torchvision
import time

def foo(x, y):
    return 2 * x + y

traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))

@torch.jit.script
def bar(x):
    return traced_foo(x, x)

# @torch.jit.script
def foo2(x, y):
    # if x.max() > y.max():
    #     r = x
    # else:
    #     r = y
    r = x
    return r

# @torch.jit.script
def bar2(x, y, z):
    for i in range(1000):
        for j in range(1000):
            z = z + i/(j+1)*torch.ones(3)
    return foo2(x, y) + z
# traced_bar2 = torch.jit.trace(bar2, (torch.rand(3), torch.rand(3), torch.rand(3)))

# t0 = time.time()
# out1 = traced_bar2(torch.zeros(3), torch.zeros(3), torch.ones(3))
# print(f"time jit:{time.time()-t0}")

# t0 = time.time()
# out1 = traced_bar2(torch.zeros(3), torch.zeros(3), torch.ones(3))
# print(f"time jit:{time.time()-t0}")

# t0 = time.time()
# out1 = traced_bar2(torch.zeros(3), torch.zeros(3), torch.ones(3))
# print(f"time jit:{time.time()-t0}")

# t0 = time.time()
# out1 = traced_bar2(torch.zeros(3), torch.zeros(3), torch.ones(3))
# print(f"time jit:{time.time()-t0}")

print("starting function call")
t0 = time.time()
out2 = bar2(torch.zeros(3), torch.zeros(3), torch.ones(3))
out2 = bar2(torch.zeros(3), torch.zeros(3), torch.ones(3))
out2 = bar2(torch.zeros(3), torch.zeros(3), torch.ones(3))
out2 = bar2(torch.zeros(3), torch.zeros(3), torch.ones(3))
out2 = bar2(torch.zeros(3), torch.zeros(3), torch.ones(3))
out2 = bar2(torch.zeros(3), torch.zeros(3), torch.ones(3))
print(f"time w/o jit:{time.time()-t0}")

print("starting function call")
t0 = time.time()
out2 = bar2(torch.zeros(3), torch.zeros(3), torch.ones(3))
out2 = bar2(torch.zeros(3), torch.zeros(3), torch.ones(3))
out2 = bar2(torch.zeros(3), torch.zeros(3), torch.ones(3))
out2 = bar2(torch.zeros(3), torch.zeros(3), torch.ones(3))
out2 = bar2(torch.zeros(3), torch.zeros(3), torch.ones(3))
out2 = bar2(torch.zeros(3), torch.zeros(3), torch.ones(3))
print(f"time w/o jit:{time.time()-t0}")


# @torch.jit.script
def add_rnd(z):
    return z + torch.zeros(3)

def layer(x,y,z):
    # k = add_rnd(z)
    return foo2(x,y) + z #add_rnd(z)  # random noise not OK!!

traced_add_rnd = torch.jit.trace(add_rnd, (torch.rand(3)))

class MyScriptModule(torch.nn.Module):
    def __init__(self):
        super(MyScriptModule, self).__init__()
        self.means = torch.nn.Parameter(torch.tensor([103.939, 116.779, 123.68])
                                        .resize_(1, 3, 1, 1))
        self.resnet = torch.jit.trace(torchvision.models.resnet18(),
                                      torch.rand(1, 3, 224, 224))

    def forward(self, input):
        return self.resnet(input - self.means)

my_script_module = torch.jit.script(MyScriptModule())
