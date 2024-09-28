# 实验 4：RV64 用户模式

## 1. 实验目的

- 创建用户态进程，并设置 `sstatus` 来完成内核态转换至用户态。
- 正确设置用户进程的**用户态栈**和**内核态栈**， 并在异常处理时正确切换。
- 补充异常处理逻辑，完成指定的系统调用（ SYS_WRITE, SYS_GETPID ）功能。

## 2. 实验环境

- Ubuntu 20.04
- VMware

## 3. 背景知识

- [浙江大学23年春夏系统三实验指导](https://parfaity.gitee.io/sys3lab-2023-stu/lab4/)

## 4. 实验步骤

### 4.1 准备工程

- 此次实验基于 lab3 所实现的代码进行。
- 首先添加用户态程序`uapp`到`.data`段（注意！lds文件里面不能用C语言的单行注释）。

```
...

.data : ALIGN(0x1000){
        _sdata = .;

        *(.sdata .sdata*)
        *(.data .data.*)

        _edata = .;

        . = ALIGN(0x1000);
        uapp_start = .;
        *(.uapp .uapp*)
        uapp_end = .;
        . = ALIGN(0x1000);

    } >ramv AT>ram

...
```

- 在`defs.h`中添加如下内容：

```c
#define USER_START (0x0000000000000000) // user space start virtual address
#define USER_END   (0x0000004000000000) // user space end virtual address
```

- 从 `repo` 同步以下文件夹: `user`， `Makefile`，并按照以下文件结构将这些文件正确放置。其中，用新的 `Makefile` 替换原本对应位置的 `Makefile`，新的 `Makefile` 将 `user` 文件夹内编译出的 `uapp.o` 列入 `LD` 链接对象中。

```
.
├── arch
│   └── riscv
│       └── Makefile
└── user
    ├── Makefile
    ├── getpid.c
    ├── link.lds
    ├── printf.c
    ├── start.S
    ├── stddef.h
    ├── stdio.h
    ├── syscall.h
    └── uapp.S
```

- 修改**根目录**下的 `Makefile`，将 `user` 纳入工程管理：

```makefile
${MAKE} -C user all
${MAKE} -C user clean
```

### 4.2 创建用户态进程

#### `proc.h`

- 本次实验只需要创建 3 个用户态进程，所以把 `proc.h` 中的 `NR_TASKS` 为4。
- 在 `thread_struct`中加入`sepc`, `sstatus`, `sscratch`。
- 在`task_struct`中加入页表。

```c
typedef unsigned long* pagetable_t;

/* 线程状态段数据结构 */
struct thread_struct {
    uint64 ra;
    uint64 sp;
    uint64 s[12];
    
    // lab4
    // sepc：保存特权态中断处理完毕后sret的返回地址。
    // sstatus：控制信号，控制当前是否中断。
    // sscratch：保存另一个状态的 sp，用于在切换状态时更新sp。
    uint64 sepc, sstatus, sscratch;
};

/* 线程数据结构 */
struct task_struct {
    struct thread_info* thread_info;
    uint64 state;    // 线程状态
    uint64 counter;  // 运行剩余时间 
    uint64 priority; // 运行优先级 1最低 10最高
    uint64 pid;      // 线程id

    struct thread_struct thread;
    
    // lab4
    // 为每一个用户态进程创建一个页表
    pagetable_t pgd;
};
```

#### `task_init()`

- 状态位设置：
  - 对每个用户态进程我们需要将 `sepc` 修改为 `USER_START`，保存特权态中断处理完毕后sret的返回地址；
  - 设置 `sstatus` 中的 `SPP` （ 使得 sret 返回至 U-Mode ）， `SPIE` （ sret 之后开启中断 ）， `SUM` （ S-Mode 可以访问 User 页面 ）；![image-20230524101445785](./assets/image-20230524101445785.png)
  -  `sscratch` 设置为 `U-Mode` 的 sp，其值为 `USER_END` （即 `U-Mode Stack` 被放置在 `user space` 的最后一个页面）。

```c
    	task[i]->thread.sepc = USER_START;
    	task[i]->thread.sstatus = csr_read(sstatus);
    	task[i]->thread.sstatus &= ~(1<<8);
    	task[i]->thread.sstatus |=  0x00040020;//(1 << 5) | (1 << 18);
        csr_write(sstatus, task[i]->thread.sstatus);
    	task[i]->thread.sscratch = USER_END;
```

- 通过 `kalloc` 接口申请一个空的页面来作为 `U-Mode Stack` 。

```c
    	unsigned long u_mode_stack = (unsigned long)kalloc();
```

- 通过`kalloc`申请一个空的页面来做页表，为了避免 `U-Mode` 和 `S-Mode` 切换的时候切换页表，我们将内核页表 （ `swapper_pg_dir` ） 复制到每个进程的页表中。

```c
    	unsigned long pagetable_v = (unsigned long)kalloc();
    	for(int i = 0; i < 512; i++){
    	  *((unsigned long*)((unsigned long*)pagetable_v + i)) = swapper_pg_dir[i]; 
    	}
```

- 将 `uapp` （用户态运行程序）、以及 `U-Mode Stack` 在每个用户态进程新建立的页表里做相应的映射。（注意注意，这里调用的create_mapping_after函数和之前开启虚拟内存的时候不同，因为已经开启虚拟内存了，所以在create_mapping_after的时候使用虚拟内存，与之前的create_mapping有所区别）。具体函数与映射关系如下。

```c
    	create_mapping_after((unsigned long*)pagetable_v, USER_START, (unsigned long)uapp_start - PA2VA_OFFSET, (unsigned long)uapp_end - (unsigned long)uapp_start, 31);
    	create_mapping_after((unsigned long*)pagetable_v, USER_END - PGSIZE, u_mode_stack - PA2VA_OFFSET, PGSIZE, 23);
```

```
                PHY_START                                                                PHY_END
                         uapp_start   uapp_end
                   │         │            │                                                 │
                   ▼         ▼            ▼                                                 ▼
       ┌───────────┬─────────┬────────────┬─────────────────────────────────────────────────┐
 PA    │           │         │    uapp    │                                                 │
       └───────────┴─────────┴────────────┴─────────────────────────────────────────────────┘
                             ▲            ▲
       ┌─────────────────────┘            │
       │                                  │
       │            ┌─────────────────────┘
       │            │
       │            │
       ├────────────┼───────────────────────────────────────────────────────────────────┬────────────┐
 VA    │    UAPP    │                                                                   │u mode stack│
       └────────────┴───────────────────────────────────────────────────────────────────┴────────────┘
       ▲                                                                                             ▲
       │                                                                                             │

   USER_START                                                                                    USER_END
```

- 将页表的**物理地址**写入`task_struct`中（因为在进程切换的时候会直接将数值写入satp），注意，这里需要进行转化，具体按照satp寄存器的结构进行转化。

```c
        unsigned long cur_satp = csr_read(satp);
    	cur_satp = (cur_satp >> 44) << 44 | ((pagetable_v - PA2VA_OFFSET) >> 12);
    	task[i]->pgd = (unsigned long *)cur_satp;
```

#### `__switch_to`

-  需要加入保存/恢复 `sepc`, `sstatus`, `sscratch` 以及切换页表的逻辑。（注意和`thread_struct`中定义的顺序一致）

```assembly
    addi t0, a0, 152
    csrr t1, sepc
    sd t1, 0*8(t0)
    csrr t1, sstatus
    sd t1, 1*8(t0)
    csrr t1, sscratch
    sd t1, 2*8(t0)
    csrr t1, satp
    sd t1, 3*8(t0)
    
    ···
    
    addi t0, a1, 152
    ld t1, 0*8(t0)
    csrw sepc, t1
    ld t1, 1*8(t0)
    csrw sstatus, t1
    ld t1, 2*8(t0)
    csrw sscratch, t1
    ld t1, 3*8(t0)
    csrw satp, t1
    # flush tlb
    sfence.vma zero, zero
```

### 4.3 修改中断入口/返回逻辑 ( _trap ) 以及中断处理函数( trap_handler )

#### `__dummy`

- 这个函数是用户态进程最开始执行的地方（因为每个进程的ra设置为__dummy。
- 在运行这个函数的时候，还处于S-mode，sret之后才会进入U-mode，我们只需要从 `sscratch` 中读取 `U-Mode sp`，将当前 `sp` 寄存器（即 `S-Mode sp` ）写入 `sscratch` ，将 `U-Mode sp` 放入当前 `sp` 寄存器，这样在`sret` 进入 `U-Mode` 时，使用的就会是 `U-Mode Stack`。 这里还需要修改进入`U-Mode`的地址为0地址，即为用户虚拟空间下代码段的起始地址。

```assembly
__dummy:
    csrr t0, sscratch
    csrw sscratch, sp
    addi sp, t0, 0
    # la t0, dummy
    # csrw sepc, t0
    csrw sepc, x0
    sret
```

#### `_traps`

- `_traps`是我们在陷入到内核态时候最开始会执行的代码。同理在 `_traps` 的首尾我们都需要做类似上一步的操作。注意如果是内核线程( 没有 U-Mode Stack ) 触发了异常，则不需要进行切换。需要在 `_trap` 的首尾都对此情况进行判断。（内核线程的 sp 永远指向的 S-Mode Stack， sscratch 为 0）
- 注意，在call trap_handler的时候，我们需要传入寄存器的值，所以我们需要加一条指令`mv a2, sp`，传递寄存器组的值。

```assembly
_traps:
    csrr t0, sscratch
    beqz t0, smode1
    csrr t0, sscratch
    csrw sscratch, sp
    mv sp, t0
    # addi sp, t0, 0
    # csrw sscratch, sp
    # 1. save 32 registers and sepc to stack
smode1:   
    addi sp, sp, -264
    
    sd x0, 0(sp)
    sd x1, 1*8(sp)
    ...
    
    # 2. call trap_handler
    mv a2, sp
    csrr a1, sepc
    csrr a0, scause
    call trap_handler
    
    ...
    ld x31, 31*8(sp)
    ld x2, 2*8(sp)

    addi sp, sp, 264

    csrr t0, sscratch
    beqz t0, smode2
    csrr t0, sscratch
    csrw sscratch, sp
    mv sp, t0
    # addi sp, t0, 0
    # 4. return from trap
smode2:
    sret
```

#### `trap_handler()`

- `uapp` 使用 `ecall` 会产生 `ECALL_FROM_U_MODE` **exception**。因此我们需要在 `trap_handler` 里面进行捕获。

```c
void trap_handler(unsigned long scause, unsigned long sepc, struct pt_regs *regs) {
    unsigned long bit63 = 1UL << 63;
    if((scause & bit63) == bit63){
	    if((scause - bit63) == 5){
	    	 //printk("[S] Supervisor Mode Timer Interrupt\n
		 clock_set_next_event();
		 do_timer();
	    }
    }
	else if(scause == 8){
		syscall(regs);
		regs->sepc += (unsigned long)4;
	}
	return;
}
```

- 新增加的第三个参数 `regs`， 在 `_traps` 中我们将寄存器的内容**连续**的保存在 `S-Mode Stack`上， 因此我们可以将这一段看做一个叫做 `pt_regs`的结构体。我们可以从这个结构体中取到相应的寄存器的值（ 比如 syscall 中我们需要从 a0 ~ a7 寄存器中取到参数 ）。

```c
struct pt_regs{
    unsigned long x0;
    unsigned long ra, sp, gp, tp;
    unsigned long t0, t1, t2;
    unsigned long s0, s1;
    unsigned long a0, a1, a2, a3, a4, a5, a6, a7;
    unsigned long s2, s3, s4, s5, s6, s7, s8, s9, s10, s11;
    unsigned long t3, t4, t5, t6;
    unsigned long sepc;
};
```

### 4.4 添加系统调用

#### `syscall.c`

- 64 号系统调用 `sys_write(unsigned int fd, const char* buf, size_t count)` 该调用将用户态传递的字符串打印到屏幕上，此处fd为标准输出（1），buf为用户需要打印的起始地址，count为字符串长度，返回打印的字符数。
- 172 号系统调用 `sys_getpid()` 该调用从current中获取当前的pid放入a0中返回，无参数。

```c
#include "syscall.h"
#include "printk.h"
#include "proc.h"

extern struct task_struct* current;
void syscall(struct pt_regs *regs){
    if(regs->a7 == SYS_WRITE){
        if(regs->a0 == 1){
            *((char *)(regs->a1 + regs->a2)) = '\0';
            regs->a0 = printk((char *)regs->a1);
        }
    }else if(regs->a7 == SYS_GETPID){
        regs->a0 = current->pid;
    }
    return;
}
```

### 4.5 修改 head.S 以及 start_kernel

- OS boot 完成之后立即调度 uapp 运行，即设置好第一次时钟中断后，在 `main()` 中直接调用 `schedule()`。
- 在 start_kernel 中调用 `schedule()` ，放置在 `test()` 之前。
- 将 head.S 中 enable interrupt sstatus.SIE 逻辑注释，确保 schedule 过程不受中断影响。

### 4.6 编译及测试

![Snipaste_2023-05-24_12-58-21](./assets/Snipaste_2023-05-24_12-58-21.png)

## 5. 思考题

1. 我们将内核页表 （ `swapper_pg_dir` ） 复制到每个进程的页表中，为什么这时可以直接在虚拟地址的空间上赋值？
    答：这时的赋值操作可以直接在虚拟地址上完成是因为内核页表本身就是映射到物理地址上的，内核页表的虚拟地址与物理地址是一一对应的。因为之前已经开启了虚拟地址，所以可以直接在虚拟地址的空间上赋值。
2. 系统调用的返回参数放置在 `a0` 中，注意不可以直接修改寄存器， 应该修改参数 `regs` 中保存的内容，这是为什么？
    答：因为寄存器的值在调用trap_handler之前就保存到了栈上，当从trap_handler返回时会将栈上的值恢复到寄存器中，因此trap_handler对寄存器的修改没有用，会被恢复掉，我们需要直接对栈上的寄存器修改，而栈上寄存器的地址就是我们传入的regs结构的地址，我们通过对结构体的修改就可以直接修改栈上的寄存器，从而修改寄存器的值。
3. 针对系统调用这一类异常， 我们为什么需要手动将 `sepc + 4` ？
    答：sepc记录的是触发异常的那条指令的地址，但是我们进行系统调用，不需要重复执行触发异常的那条指令（不同于因为错误引发的异常），我们将sepc指向下一条指令即可。
4. 我们为什么要将 head.S 中 enable interrupt sstatus.SIE 逻辑注释，确保 schedule 过程不受中断影响？
    答：因为SPIE为1，进入用户态后会将SIE设置为1而把中断开启。SIE位置0的时候，会禁用所有的S态异常，为了在S态不引发时钟中断，我们将enable interrupt sstatus.SIE 逻辑注释，确保 schedule 过程不受中断影响。

## 6. 实验心得

实验的难度在提升，内容更加综合，细节的地方有很多，一不小心就会有错误。下面罗列一下实验过程中我遇到的问题：

- 在lds那里使用类似c语言的单行注释进行注释，结果导致编译出错。
- task_init()中进行create_mapping()时，这里的create_mapping()不再能使用物理地址（因为虚拟地址已经开启），所以在create_mapping逻辑上需要进行一些小小的修改，我是直接重写了另一个函数create_mapping_after()。这个地方卡了两天。
- 无法切换进程，发现是__switch_to的时候，satp出错，还是task_init()里面，stap没有设置好。
- 可以切换进程之后，无法进行系统调用，跟踪调试了一下，发现scause没有问题，但是regs根本没有传过去，最后发现，我在call trap_handler()的时候根本就没有传递regs地址的参数。

以上的问题很大程度上还是因为对于实验的原理搞的不是很清楚，但是经过了上面这些问题之后，我现在对于实验的流程和原理有了更深刻的理解，所以虽然过程很痛苦，但是收获还是很大的。
