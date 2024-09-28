# 基础知识

> 《python编程：从入门到实践》

## 第1章 起步

## 第2章 变量和简单数据类型

### 2.3 字符串

字符串就是一系列字符。在Python中，用引号括起的都是字符串，其中的引号可以是单引号，也可以是双引号。

对字符串使用方法

- 大小写改变

```python
name = "Ada Lovelace"
print(name.title())
print(name.upper())
print(name.lower())

输出：
Ada Lovelace
ADA LOVELACE
ada lovelace
```

- 剔除空白

```python
name = " Ada Lovelace "
print(name.rstrip())
print(name.lstrip())
print(name.strip())

输出：
' Ada Lovelace'
'Ada Lovelace '
'Ada Lovelace'
```

### 2.4 数字

- 将数字转换为字符，`str(int)`

### 2.6 Python之禅

```
>>>import this
The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
```

## 第3章 列表简介

### 3.2 修改、添加和删除元素

```python
cars = ['bmw', 'audi', 'toyota', 'subaru']
print(cars) #打印列表
cars.append('byd') #添加元素
cars.insert(0, 'byd') #插入元素到某位置
del cars[0] #删除元素
cars.pop() #删除最后一个元素并返回值
cars.pop(0) #删除某个元素并返回值
cars.remove('bmw') #删除任何元素并返回值，只能删除第一个指定的值
cars.index('bmw') #打印第一次出现‘bmw’时候的对应的索引
cars.count('bmw') #打印出现‘bmw’的次数
```

### 3.3 组织列表

```python
cars = ['bmw', 'audi', 'toyota', 'subaru']
cars.sort() #以字母顺序从小到大永久排序
cars.sort(reverse=True) #反方
sorted(cars) #暂时排序
cars.reverse() #逆序
len(cars) #元素个数
```

## 第4章 操作列表

```python
#遍历列表
magicians = ['alice', 'david', 'carolina'] 
for magician in magicians: 
    print(magician)

#数值列表,打印1-4
for value in range(1,5):
    print(value)
    
#将range()转化为列表
numbers = list(range(1,5))
numbers = list(range(1,5,2)) #最后一个参数为步长

#统计数字列表
>>> digits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
>>> min(digits)
0
>>> max(digits)
9
>>> sum(digits)
45

#列表解析
squares = [value**2 for value in range(1,11)]
print(squares) #输出[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

#列表切片
>>> players = ['charles', 'martina', 'michael', 'florence', 'eli']
>>> print(players[1:4])
['martina', 'michael', 'florence']
>>> print(players[:4])
['charles', 'martina', 'michael', 'florence']
>>> print(players[-3:])
['michael', 'florence', 'eli']

#复制列表
players_copy = players[:]
```

### 4.5 元组

```python
#元组定义，元素不可修改
dimensions = (200, 50)
#如下形式也可
dimensions = 200, 50
```

## 第5章 if语句

```python
#if语句的结构和一些表达式
if ... and ...:
    ...
elif 1 in range[1,5]:
    ...
elif 1 not in range[1,5]:
    ...  
else:
    ...
   
#判断列表是否为空
magicians = ['alice', 'david', 'carolina'] 
if magicians:
    print("There's element.")
```

## 第6章 字典

```python
#创建字典
>>> alien_0 = {'color': 'green', 'points': 5}
>>> print(alien_0['color'])
green
>>> print(alien_0['points'])
5

#添加
alien_0 = {}
print(alien_0)
alien_0['x_position'] = 0
alien_0['y_position'] = 25
print(alien_0)
# {}
# {'y_position': 25, 'x_position': 0}

#修改（类似添加

#删除
del alien_0['x_position']

#遍历
“favorite_languages = {
    'jen': 'python',
    'sarah': 'c',
    'edward': 'ruby',
    'phil': 'python',
    }

for name, language in favorite_languages.items(): 
    print(name.title() + "'s favorite language is " + 
        language.title() + ".")
    
#遍历所有键
#方法.keys()其实就是返回一个列表
for name in favorite_languages.keys(): 
	print(name.title())

#按顺序遍历
for name in sorted(favorite_languages.keys()):
    
#遍历所有值
for language in favorite_languages.values():
    
#去除重复的情况
for language in set(favorite_languages.values()):
```

## 第7章 用户输入和while循环

- pass：什么都不做
- continue：跳出本次循环，开始下一次循环
- break：退出循环

```python
#删除包含特定值的所有元素
pets = ['dog', 'cat', 'dog', 'goldfish', 'cat', 'rabbit', 'cat']
print(pets)

while 'cat' in pets:
    pets.remove('cat')

print(pets)
```

## 第8章 函数

- 传递参数有顺序传递和关键值传递

```python
#举个书中的例子，应该就能看懂参数传递
def get_formatted_name(first_name, last_name, middle_name=''): 
    """返回整洁的姓名"""
    if middle_name: 
        full_name = first_name + ' ' + middle_name + ' ' + last_name
    else: 
        full_name = first_name + ' ' + last_name
    return full_name.title()

musician = get_formatted_name('jimi', 'hendrix')
print(musician)

musician = get_formatted_name('john', 'hooker', 'lee') 
print(musician)
```

- 若参数传递的是列表，那么在函数内修改列表是永久性的，如果函数内不想影响到函数外，那么在调用函数的时候要使用切片

```python
function_name(list_name[:])
```

- 传递任意数量的参数。形参名*toppings中的星号让Python创建一个名为toppings的空元组，并将收到的所有值都封装到这个元组中。

```python 
def make_pizza(size, *toppings):
    """概述要制作的比萨"""
    print("\nMaking a " + str(size) +
          "-inch pizza with the following toppings:")
    for topping in toppings:
        print("- " + topping)

make_pizza(16, 'pepperoni')
make_pizza(12, 'mushrooms', 'green peppers', 'extra cheese')
```

- 传递任意数量的关键字实参

```python
def build_profile(first, last, **user_info):
    """创建一个字典，其中包含我们知道的有关用户的一切"""
    profile = {}
    profile['first_name'] = first 
    profile['last_name'] = last
    for key, value in user_info.items(): 
        profile[key] = value
    return profile

user_profile = build_profile('albert', 'einstein',
                             location='princeton',
                             field='physics')
print(user_profile)
```

- 导入某个文件的某个函数并改名

```python
from pizza import make_pizza as mp

mp(16, 'pepperoni')
```

- 导入某个模块并改名

```python
import pizza as p

p.make_pizza(16, 'pepperoni')
```

## 第9章 类

- 基本定义

```python
class Restaurant():
    def __init__(self, restaurant_name, cuisine_type):
        self.restaurant_name = restaurant_name
        self.cuisine_type    = cuisine_type
        self.number_served   = 0

    def describe_restaurant(self):
        print("Restaurant name: " + self.restaurant_name.title())
        print("Cuisine type: " + self.cuisine_type.title())
        print("Number served: " + str(self.number_served))

    def open_restaurant(self):
        print("Restaurant is open.")

    def set_number_served(self, number):
        self.number_served = number
        print("Number served: " + str(self.number_served))

    def increment_number_served(self, number):
        self.number_served += number
        print("Number served: " + str(self.number_served))
    
restaurant = Restaurant("the mean queen", "pizza")
restaurant.describe_restaurant()
restaurant.open_restaurant()
restaurant.set_number_served(10)
restaurant.increment_number_served(10)
```

- 继承（可以将实例用作属性，来减少一个类的之内的元素）

```python
class Restaurant():
    def __init__(self, restaurant_name, cuisine_type):
        self.restaurant_name = restaurant_name
        self.cuisine_type    = cuisine_type
        self.number_served   = 0

    def describe_restaurant(self):
        print("Restaurant name: " + self.restaurant_name.title())
        print("Cuisine type: " + self.cuisine_type.title())
        print("Number served: " + str(self.number_served))

    def open_restaurant(self):
        print("Restaurant is open.")

    def set_number_served(self, number):
        self.number_served = number
        print("Number served: " + str(self.number_served))

    def increment_number_served(self, number):
        self.number_served += number
        print("Number served: " + str(self.number_served))
    
class IceCreamStand(Restaurant):
    def __init__(self, restaurant_name, cuisine_type):
        super().__init__(restaurant_name, cuisine_type)
        self.flavors = ["chocolate", "vanilla", "strawberry"]

    def show_flavors(self):
        print("Flavors: " + str(self.flavors))

IceCreamStand = IceCreamStand("Ice Cream Stand", "Ice Cream")
IceCreamStand.open_restaurant()
IceCreamStand.describe_restaurant()
IceCreamStand.show_flavors()
```

- Python允许你将类存储在模块中，然后在主程序中导入所需的模块。其实与之前的类似。

```python
#从一个模块中导入一个或者多个类
from restaurant import Restaurant, IceCreamStand
#从一个模块中导入所有类
from restaurant import *
#导入模块
import restaurant
#在一个模块中导入另一个模块

```

- 类名应采用驼峰命名法，即将类名中的每个单词的首字母都大写，而不使用下划线。实例名和模块名都采用小写格式，并在单词之间加上下划线。
  对于每个类，都应紧跟在类定义后面包含一个文档字符串。这种文档字符串简要地描述类的功能，并遵循编写函数的文档字符串时采用的格式约定。每个模块也都应包含一个文档字符串，对其中的类可用于做什么进行描述。
  可使用空行来组织代码，但不要滥用。在类中，可使用一个空行来分隔方法；而在模块中，可使用两个空行来分隔类。
  需要同时导入标准库中的模块和你编写的模块时，先编写导入标准库模块的import语句，再添加一个空行，然后编写导入你自己编写的模块的import语句。在包含多条import语句的程序中，这种做法让人更容易明白程序使用的各个模块都来自何方。

## 第10章 文件和异常

**读取文件**

- 关键字with在不再需要访问文件之后将其关闭，这样可以免去close()的麻烦。
- 有了表示pi_digits.txt的文件对象后，我们使用方法read()（前述程序的第2行）读取这个文件的全部内容，并将其作为一个长长的字符串存储在变量contents中。
- “为何会多出这个空行呢？因为read()到达文件末尾时返回一个空字符串，而将这个空字符串显示出来时就是一个空行。要删除末尾的空行，可在print语句中使用rstrip()，Python方法rstrip()删除（剥除）字符串末尾的空白。

```python
filename = 'pi_digits.txt'
file_object = open(filename)
contents = file_object.read()
file_object.close()
```

```python
with open('pi_digits.txt') as file_object:
    contents = file_object.read()
    print(contents)
```

- open使用文件路径时候，windows要使用反斜杠。
- 逐行读取

```python
“filename = 'pi_digits.txt'
with open(filename) as file_object: 
    for line in file_object: 
    print(line.rstrip())
```

```python
“filename = 'pi_digits.txt'
with open(filename) as file_object:
    #方法readlines()从文件中读取每一行，并将其存储在一个列表中
    lines = file_object.readlines() 

for line in lines: 
    print(line.rstrip())
```

---

**写入文件**

```python
filename = 'programming.txt'
file_object = open(filename, 'w')
file_object.write("I love programming.")
file_object.close()
```

```python
filename = 'programming.txt'
with open(filename, 'w') as file_object: 
    file_object.write("I love programming.") 
```

- Python只能将字符串写入文本文件。要将数值数据存储到文本文件中，必须先使用函数str()将其转换为字符串格式。

- 函数write()不会在你写入的文本末尾添加换行符，因此写入多行时需要指定换行符。


---

**异常**

- Python使用被称为异常的特殊对象来管理程序执行期间发生的错误。每当发生让Python不知所措的错误时，它都会创建一个异常对象。如果你编写了处理该异常的代码，程序将继续运行；如果你未对异常进行处理，程序将停止，并显示一个traceback，其中包含有关异常的报告。
- 异常是使用try-except代码块处理的。try-except代码块让Python执行指定的操作，同时告诉Python发生异常时怎么办。使用了try-except代码块时，即便出现异常，程序也将继续运行：显示你编写的友好的错误消息，而不是令用户迷惑的traceback。

```python
try:
    file = open('eeee', 'r+')
# 捕获所有的异常
except Exception as e:
    print(e)
else
	file.write('ssss')
file.close()
```

```python
first_number = input("\nFirst number: ")
if first_number == 'q':
    break
second_number = input("Second number: ")
try:
    answer = int(first_number) / int(second_number)
except ZeroDivisionError: 
    print("You can't divide by 0!")
else: 
    print(answer)
```

- 不处理异常：使用python

```python
def count_words(filename):
    """计算一个文件大致包含多少个单词"""
    try:
        --snip--
    except FileNotFoundError:
        pass ❶
    else:
        --snip--

filenames = ['alice.txt', 'siddhartha.txt', 'moby_dick.txt', 'little_women.txt']
for filename in filenames:
    count_words(filename)
```

---

存储数据

- 利用json的load和dump来存储和读取数据；

```python
import json

# 如果以前存储了用户名，就加载它
# 否则，就提示用户输入用户名并存储它
filename = 'username.json'
try:
    with open(filename) as f_obj:
        username = json.load(f_obj)
except FileNotFoundError:
    username = input("What is your name? ")
    with open(filename, 'w') as f_obj:
        json.dump(username, f_obj)
        print("We'll remember you when you come back, " + username + "!")
else:
    print("Welcome back, " + username + "!")
```
改进
```python
import json

def get_stored_username():
    """如果存储了用户名，就获取它"""
    --snip--

def get_new_username():
    """提示用户输入用户名"""
    username = input("What is your name? ")
    filename = 'username.json
    with open(filename, 'w') as f_obj:
        json.dump(username, f_obj)
    return username

def greet_user():
    """问候用户，并指出其名字"""
    username = get_stored_username()
    if username:
        print("Welcome back, " + username + "!")
    else:
        username = get_new_username()
        print("We'll remember you when you come back, " + username + "!")

greet_user()
```

## 第11章 测试代码

### 测试函数

如下代码将名和姓合并成姓名并在中间加入一空格，再将他们的首字母都大写。

```python
def get_formatted_name(first, last):
    """Generate a neatly formatted full name."""
    full_name = first + ' ' + last
    return full_name.title()
```

使用Python模块unittest中的工具来测试代码。

```python
import unittest
from name_function import get_formatted_name

class NamesTestCase(unittest.TestCase):
    """测试name_function.py"""

    def test_first_last_name(self):
        """能够正确地处理像Janis Joplin这样的姓名吗？"""
        formatted_name = get_formatted_name('janis', 'joplin')
        self.assertEqual(formatted_name, 'Janis Joplin')

unittest.main() # 让Python运行这个文件中的测试
```

### 测试类

**unittest Module中的断言方法**

| 方法                    | 用途               |
| ----------------------- | ------------------ |
| assertEqual(a, b)       | 核实 a==b          |
| assertNotEqual(a, b)    | 核实 a!=b          |
| assertTrue(x)           | 核实x为True        |
| assertFalse(x)          | 核实x为False       |
| assertIn(item, list)    | 核实item在list中   |
| assertNotIn(item, list) | 核实item不在list中 |

**survey.py**

```python
class AnonymousSurvey():
    """收集匿名调查问卷的答案"""

    def __init__(self, question): 
        """存储一个问题，并为存储答案做准备"""
        self.question = question
        self.responses = []

    def show_question(self): 
        """显示调查问卷"""
        print(self.question)

    def store_response(self, new_response): 
        """存储单份调查答卷"""
        self.responses.append(new_response)

    def show_results(self): 
        """显示收集到的所有答卷"""
        print("Survey results:")
        for response in self.responses:
            print('- ' + response)
```

- 这个类首先存储一个指定的问题，并创建一个空列表来存储答案。
- 之后的几个方法见注释。

**test_survey.py**

```python
import unittest
from survey import AnonymousSurvey

class TestAnonymousSurvey(unittest.TestCase): 
    """针对AnonymousSurvey类的测试"""
    
    def test_store_three_responses(self):
        """测试三个答案会被妥善地存储"""
        question = "What language did you first learn to speak?"
        my_survey = AnonymousSurvey(question)
        responses = ['English', 'Spanish', 'Mandarin'] 
        for response in responses:
            my_survey.store_response(response)

        for response in responses: 
            self.assertIn(response, my_survey.responses)
            
unittest.main()
```

使用setUp函数，只需创建一次对象。

```python
import unittest
from survey import AnonymousSurvey

class TestAnonymousSurvey(unittest.TestCase):
    """针对AnonymousSurvey类的测试"""

    def setUp(self):
        """
        创建一个调查对象和一组答案，供使用的测试方法使用
        """
        question = "What language did you first learn to speak?"
        self.my_survey = AnonymousSurvey(question) 
        self.responses = ['English', 'Spanish', 'Mandarin'] 

    def test_store_single_response(self):
        """测试单个答案会被妥善地存储"""
        self.my_survey.store_response(self.responses[0])
        self.assertIn(self.responses[0], self.my_survey.responses)

    def test_store_three_responses(self):
        """测试三个答案会被妥善地存储"""
        for response in self.responses:
            self.my_survey.store_response(response)
        for response in self.responses:
            self.assertIn(response, self.my_survey.responses)

unittest.main()
```

---

> 以下来自[【莫烦Python】Python 基础教程]( https://www.bilibili.com/video/BV1wW411Y7ai/?p=4&share_source=copy_web&vd_source=8a9ee4e0aecd4c6d44821790577c572e)

## P6: for循环

```python
example_list = [1,2,3,4]
for i in example_list:
	print(i)
```

- mac：command + “[” 进行整体代码块的缩进修改

- win：control + “[”

```python
for i in range(1,5)
	print(i)
```

- range(1,5,1)：1,2,3,4
- range(1,5,2)：1,3

## P17: 文件读写

```python
filename = 'pi_digits.txt'
file_object = open(filename)
# 这时候contents是一个list
contents = file_object.readlines()
file_object.close()
```

## P29: zip lambda map

### zip

```python
>>> a = [1,2,3]
>>> b = [4,5,6]
>>> zip(a, b)
<zip object at 0x102de6700>
>>> list(zip(a, b))
[(1, 4), (2, 5), (3, 6)]
>>> for i,j in zip(a,b):\
...     print(i/2, j*2)
...
0.5 8
1.0 10
1.5 12
```

### lambda

使用lambda定义一个比较简单的函数，冒号前是输入的参数，冒号后是返回的值。

```python
>>> func = lambda x,y: x+y
>>> func(2,3)
5
```

### map

```python
>>> list(map(func, [1,2], [2,3]))
[3, 5]
```

## P30: 浅复制&深复制

直接赋值，那就是两个都引用到同一内存空间

```python
>>> import copy
>>> a = [1,2,3]
>>> b = a
>>> id(a)
4343505280
>>> id(b)
4343505280
>>> b[0] = 11
>>> a
[11, 2, 3]
>>> a[0] = 22
>>> b
[22, 2, 3]
```

copy只会复制父对象，而子对象是共用的

```python
>>> c = copy.copy(a) #浅复制
>>> id(c)
4343578240
>>> id(a)
4343505280
>>> a = [1, 2, [3,4]]
>>> d = copy.copy(a)
>>> id(a) == id(d)
False
>>> id(a[2]) == id(d[2])
True
```

deepcopy完全复制所有的父对象，任何东西都不会被引用到同一个内存空间

```python
>>> e = copy.deepcopy(a)
>>> id(e[2]) == id(a[2])
False
```

## P31: Python threading

多线程管理。

## P33: Python tkinter

python自带的GUI。

## P34: pickle 存放数据

写入变量

```python
import pickle
a_dict = {'da': 111, 2: [23, 1, 4], '23': {1:2, 'd':'sad'}}
# pickle a variable to a file
file = open('pickle_example.pickle', 'wb')
pickle.dump(a_dict, file)
file.close()
```

读取变量

```python
import pickle
a_dict = {'da': 111, 2: [23, 1, 4], '23': {1:2, 'd':'sad'}}
file = open('pickle_example.pickle', 'rb')
a_dict1 = pickle.load(file)
file.close()
print(a_dict)
print(a_dict1)
```

输出结果

```shell
{'da': 111, 2: [23, 1, 4], '23': {1: 2, 'd': 'sad'}}
{'da': 111, 2: [23, 1, 4], '23': {1: 2, 'd': 'sad'}}
```

## P35: set 找不同

```python
char_list = ['a', 'b', 'c', 'c', 'd', 'd', 'd']
print(set(char_list))
print(type(set(char_list)))

sentence = 'Welcome Back to This Tutorial'
print(set(sentence))

# set的操作
print("\nset的操作:")
# 添加
s = set(char_list)
s.add('x')
print(s)
# 清除
s.clear()
print(s)
# 去除某个数据
s = set(char_list)
s.remove('a')
print(s)
# 取查集
set1 = {'a', 'b', 'c', 'd'}
set2 = {'a', 'b', 'e', 'f'}
print(set1.difference(set2))
# 取交集
print(set1.intersection(set2))
```

## P36: RegEx 正则表达

### 导入模块

```python
import re
```

### 不用正则的判断

```python
pattern1 = "cat"
pattern2 = "bird"
string = "dog runs to cat"
print(pattern1 in string)  # True
print(pattern2 in string)  # False
```

### 用正则寻找配对

```python
pattern1 = "cat"
pattern2 = "bird"
string = "dog runs to cat"
print(re.search(pattern1, string))
print(re.search(pattern2, string))
```

### 匹配多种可能

前面加上r表示这是一个表达式而不是一个字符串，中间的[au]表示可以接受a或者u。

```python
re.search(r"r[au]n", "I run to you")
```

```python
print(re.search(r"f(ou|i)nd", "I find you"))
print(re.search(r"f(ou|i)nd", "I found you"))
```

如下`[0-9][a-z]`表示，0到9或者a到z任何一个都可以被匹配。

```python
print(re.search(r"r[0-9a-z]n", "dogs runs to cat"))
```

### 特殊种类匹配

| 特定标识 | 含义                                     | 范围                                                      |
| -------- | ---------------------------------------- | --------------------------------------------------------- |
| \d       | 任何数字                                 | [0-9]                                                     |
| \D       | 不是数字的                               |                                                           |
| \s       | 任何空白字符                             | [ \t\n\r\f\v]                                             |
| \S       | 空白字符以外的                           |                                                           |
| \w       | 任何大小写字母,数字和 _                  | [a-zA-Z0-9_]                                              |
| \W       | \w 以外的                                |                                                           |
| \b       | 匹配一个单词边界                         | 比如 er\b 可以匹配 never 中的 er，但不能匹配 verb 中的 er |
| \B       | 匹配非单词边界                           | 比如 er\B 能匹配 verb 中的 er，但不能匹配 never 中的 er   |
| \\\\     | 匹配 \                                   |                                                           |
| .        | 匹配任何字符 (除了 \n)                   |                                                           |
| ?        | 前面的模式可有可无                       | `Mon(day)?`，不管day有没有都匹配                          |
| *        | 重复**零次**或多次                       | `ab*`，b出现0或者多次都可以匹配                           |
| +        | 重复**一次**或多次                       | `ab+`，b出现1或者多次都可以匹配                           |
| {n,m}    | 重复 n 至 m 次                           |                                                           |
| {n}      | 重复 n 次                                |                                                           |
| +?       | 非贪婪，最小方式匹配 +                   |                                                           |
| *?       | 非贪婪，最小方式匹配 *                   |                                                           |
| ??       | 非贪婪，最小方式匹配 ?                   |                                                           |
| ^        | 匹配一行开头，在 re.M 下，每行开头都匹配 | `^dog`，dog要出现在句首才匹配                             |
| $        | 匹配一行结尾，在 re.M 下，每行结尾都匹配 | `dog$`，dog要出现在句尾才匹配                             |
| \A       | 匹配最开始，在 re.M 下，也从文本最开始   |                                                           |
| \B       | 匹配最结尾，在 re.M 下，也从文本最结尾   |                                                           |

再回到 email 匹配的例子，我们就不难发现，我用了 `\w` 这个标识符，来表示任意的字母和数字还有下划线。因为大多数 email 都是只包含这些的。 而且我还是用了 `+?` 用来表示让 `\w` 至少匹配 1 次，并且当我识别到 `@` 的时候做非贪婪模式匹配，也就是遇到 `@` 就跳过当前的重复匹配模式， 进入下一个匹配阶段。

```python
re.search(r"\w+?@\w+?\.com", "mofan@mofanpy.com")
```

除了邮箱在我们的文件管理系统中常见，我们再举一个常见的电话号码的识别例子。比如手机号的识别，假设我们只识别 138 开头的手机号码。 下面的 `\d{8}` 就是用来表示任意的数字，重复 8 遍。

```python
print(re.search(r"138\d{8}", "13812345678"))
print(re.search(r"138\d{8}", "138123456780000"))
```

### 中文

```
"中".encode("unicode-escape")
```

你看到 `中` 字用 Unicode 表示的样子后，会不会突然发现，这不就是一串英文吗？只要我把汉字的 Unicode 全写出来就好了， 好在 Unicode 是可以连续的，我们可以用英文那样类似的办法来处理。

```python
re.search(r"[\u4e00-\u9fa5]+", "我爱莫烦Python。")
```

这挺符合我们的预期，它将里面的中文都识别出来了，剔除掉了英文和标点。那有时候我们还是想留下对标点的识别的，怎么办？ 我们只需要将中文标点的识别范围，比如 `[！？。，￥【】「」]` 补进去就好了。

```python
re.search(r"[\u4e00-\u9fa5！？。，￥【】「」]+", "我爱莫烦。莫烦棒！")
```

### 获取特定信息

用 `group` 功能获取到不同括号中匹配到的字符串。

```python
string = "I have 2021-02-01.jpg, 2021-02-02.jpg, 2021-02-03.jpg"
match = re.finditer(r"(\d+?)-(\d+?)-(\d+?)\.jpg", string)
for file in match:
    print("matched string:", file.group(0), ",year:", file.group(1), ", month:", file.group(2), ", day:", file.group(3))
```

下面这个 `findall` 也可以达到同样的目的。只是它没有提供 `file.group(0)` 这种全匹配的信息。

```python
string = "I have 2021-02-01.jpg, 2021-02-02.jpg, 2021-02-03.jpg"
match = re.findall(r"(\d+?)-(\d+?)-(\d+?)\.jpg", string)
for file in match:
    print("year:", file[0], ", month:", file[1], ", day:", file[2])
```

有时候我们 group 的信息太多了，括号写得太多，让人眼花缭乱。我们还能用一个名字来索引匹配好的字段， 然后用 `group("索引")` 的方式获取到对应的片段。注意，上面方案中的 `findall` 不提供名字索引的方法， 我们可以用 `search` 或者 `finditer` 来调用 `group` 方法。为了索引，我们需要在括号中写上 `?P<索引名>` 这种模式。

```python
string = "I have 2021-02-01.jpg, 2021-02-02.jpg, 2021-02-03.jpg"
match = re.finditer(r"(?P<y>\d+?)-(?P<m>\d+?)-(?P<d>\d+?)\.jpg", string)
for file in match:
    print("matched string:", file.group(0), 
        ", year:", file.group("y"), 
        ", month:", file.group("m"), 
        ", day:", file.group("d"))
```

### 查找替换等更多功能

| 功能          | 说明                                                         | 举例                                                         |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| re.search()   | 扫描查找整个字符串，找到第一个模式匹配的                     | re.search(rrun, I run to you) > 'run'                        |
| re.match()    | 从字符的最开头匹配，找到第一个模式匹配的**即使用 re.M 多行匹配，也是从最最开头开始匹配** | re.match(rrun, I run to you) > None                          |
| re.findall()  | 返回一个不重复的 pattern 的匹配列表                          | re.findall(rr[ua]n, I run to you. you ran to him) > ['run', 'ran'] |
| re.finditer() | 和 findall 一样，只是用迭代器的方式使用                      | for item in re.finditer(rr[ua]n, I run to you. you ran to him): |
| re.split()    | 用正则分开字符串                                             | re.split(rr[ua]n, I run to you. you ran to him) > ['I ', ' to you. you ', ' to him'] |
| re.sub()      | 用正则替换字符                                               | re.sub(rr[ua]n, jump, I run to you. you ran to him) > 'I jump to you. you jump to him' |
| re.subn()     | 和 sub 一样，额外返回一个替代次数                            | re.subn(rr[ua]n, jump, I run to you. you ran to him) > ('I jump to you. you jump to him', 2) |

```python
print("search:", re.search(r"run", "I run to you"))
print("match:", re.match(r"run", "I run to you"))
print("findall:", re.findall(r"r[ua]n", "I run to you. you ran to him"))

for i in re.finditer(r"r[ua]n", "I run to you. you ran to him"):
    print("finditer:", i)

print("split:", re.split(r"r[ua]n", "I run to you. you ran to him"))
print("sub:", re.sub(r"r[ua]n", "jump", "I run to you. you ran to him"))
print("subn:", re.subn(r"r[ua]n", "jump", "I run to you. you ran to him"))
```

```
search: <re.Match object; span=(2, 5), match='run'>
match: None
findall: ['run', 'ran']
finditer: <re.Match object; span=(2, 5), match='run'>
finditer: <re.Match object; span=(18, 21), match='ran'>
split: ['I ', ' to you. you ', ' to him']
sub: I jump to you. you jump to him
subn: ('I jump to you. you jump to him', 2)
```

### 多模式匹配

| 模式 | 全称          | 说明                                                         |
| ---- | ------------- | ------------------------------------------------------------ |
| re.l | re.IGNORECASE | 忽略大小写                                                   |
| re.M | re.MULTILINE  | 多行模式，改变'^'和'$'的行为                                 |
| re.S | re.DOTALL     | 点任意匹配模式，改变'.'的行为, 使".“可以匹配任意字符         |
| re.L | re.LOCALE     | 使预定字符类 \w \W \b \B \s \S 取决于当前区域设定            |
| re.U | re.UNICODE    | 使预定字符类 \w \W \b \B \s \S \d \D 取决于unicode定义的字符属性 |
| re.X | re.VERBOSE    | 详细模式。这个模式下正则表达式可以是多行，忽略空白字符，并可以加入注释。以下两个正则表达式是等价的 |

### 更快地执行

提前编译。

```python
import time
n = 1000000
# 不提前 compile
t0 = time.time()
for _ in range(n):
    re.search(r"ran", "I ran to you")
t1 = time.time()
print("不提前 compile 运行时间：", t1-t0)

# 先做 compile
ptn = re.compile(r"ran")
for _ in range(n):
    ptn.search("I ran to you")
print("提前 compile 运行时间：", time.time()-t1)
```

> [正则表达式参考](https://mofanpy.com/tutorials/python-basic/interactive-python/regex#%E6%AD%A3%E5%88%99%E7%BB%99%E9%A2%9D%E5%A4%96%E4%BF%A1%E6%81%AF)

