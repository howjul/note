#### 判断题T2
- After the first run of Insertion Sort, it is possible that no element is placed in its final position.
- 正确；
- 比如三个数，逆序排列；

#### 单选题T1
- Apply DFS to a directed acyclic graph, and output the vertex before the end of each recursion. The output sequence will be:
- reversely topologically sorted；
- 递归的顺序是遍历1->遍历2->遍历3->.....->遍历n->返回n->返回n-1...->返回3->返回2->返回1，返回的时候打印，因此是从最后一个结点开始打印到源点。是逆拓扑序。

