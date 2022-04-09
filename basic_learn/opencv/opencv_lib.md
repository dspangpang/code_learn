# 调用opencv库的方式

```shell
g++ `pkg-config opencv --cflags` text.cpp  -o opencv `pkg-config opencv --libs`
```


g++ `pkg-config opencv --cflags` main.cpp SemiGlobalMatching.cpp sgm_util.cpp -o opencv `pkg-config opencv --libs`