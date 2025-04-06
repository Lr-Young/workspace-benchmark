# import torch
# import dis

# 示例模型（含控制流）
# class MyModel(torch.nn.Module):
#     def forward(self, x):
#         print("Forward function called")
#         if x.sum() > 0:  # 控制流语句
#             x = x * 2
#         else:
#             x = x + 1
#         return x

# model = MyModel()
# example_input = torch.tensor([2.0])  # 示例输入

# # 图捕获过程
# traced_model = torch.jit.trace(model, example_input)
# print(traced_model.code)  # 查看生成的静态图代码
# print(traced_model.code_with_constants[1].const_mapping)
# print(traced_model(example_input))


# 含控制流的模型
# class MyModel(torch.nn.Module):
#     def forward(self, x):
#         print("Forward function called")
#         if x.sum() > 0:  # Python原生控制流
#             x = x * 2
#         else:
#             x = x - 1
#         for i in range(3):  # 循环语句
#             x += i
#         return x

# model = MyModel()

# # 脚本化编译
# scripted_model = torch.compile(model, fullgraph=True)
# print(scripted_model)  # 查看生成的静态图代码

# call_count = 0

# class MyModel(torch.nn.Module):
#     def forward(self, x):
#         global call_count
#         call_count += 1
#         return torch.rand(10) + x

# model = MyModel()

# compiled_model = torch.compile(model, fullgraph=True)
# print(compiled_model)

def foo1(x):
    return x + 1

def foo2(x):
    return x + 2

def foo(x):
    x = foo1(x); x = foo2(x)
    return x

foo(1)
# dis.dis(foo)
