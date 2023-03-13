### 1. namespace
#### 1.1 声明namespace

![image-20221127110334156](../../img/test/202211271103673.png)

#### 1.2 对namespace里的函数进行定义

![image-20221127110520212](../../img/test/202211271105230.png)

#### 1.3 使用namespace三种方式

![image-20221127110706849](../../img/test/202211271107863.png)

![image-20221127110720860](../../img/test/202211271107873.png)

![image-20221127110733717](../../img/test/202211271107731.png)

#### 1.4 一些注意点

- Ambiguities
- ![image-20221127111339916](../../img/test/202211271113931.png)
- Namespace aliases
- ![image-20221127111403633](/Users/zhz/Library/Application Support/typora-user-images/image-20221127111403633.png)

- Namespace conposition（合并namespace）
- ![image-20221127111608840](../../img/test/202211271116856.png)

- Namespaces are open
- ![image-20221127111702467](../../img/test/202211271117484.png)

### 2. Inheritance
- （引入情境讲了很久，但是其他就没什么了）；
- （用法就自己在书上看吧）；
- 父类的private在子类还是存在的，但是子类无法直接访问，而是得通过可以访问的父类函数（public和protected）访问；
- 父类的重载的函数，只要在子类重载过一个，那么父类的其他重载函数都没有了（cpp特有），叫做name hide（名字隐藏）；

