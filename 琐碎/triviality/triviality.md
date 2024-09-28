#### 1. 提示栏和分栏
```
markdown_extensions:
  - admonition
  - pymdownx.details
  - pymdownx.superfences
```
```
markdown_extensions:
  - pymdownx.superfences
  - pymdownx.tabbed:
      alternate_style: true 
```
=== "note"

    !!! note

        这是note！

=== "abstract"

    !!! abstract

=== "info"

    !!! info

=== "success"

    !!! success

=== "tip"

    !!! tip

=== "question"

    !!! question

=== "warning"

    !!! warning

=== "failure"

    !!! failure

=== "danger"

    !!! danger

=== "bug"

    !!! bug

=== "example"

    !!! example

=== "quote"

    !!! quote

#### 2. 按钮
```
markdown_extensions:
  - attr_list
```
```
[内容](gh-pages中的绝对路径就可以下载，写相对路径就可以跳转){ .md-button }
[内容](gh-pages中的绝对路径就可以下载，写相对路径就可以跳转){ .md-button .md-button--primary}
```
#### 3. 代码块
```
markdown_extensions:
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - pymdownx.superfences
theme:
  features:
    - content.code.copy
    - content.code.annotate  
```
- 指定文件名、显示行数、指定两行高亮、指定连续高亮、去掉代码复制按钮、点击显示代码注释
#### 4. 表格
```
markdown_extensions:
  - tables
```
#### 5. 注脚
```
markdown_extensions:
  - footnotes
```
#### 6. 图标
```
markdown_extensions:
  - pymdownx.emoji:
      emoji_index: !!python/name:materialx.emoji.twemoji
      emoji_generator: !!python/name:materialx.emoji.to_svg

```
#### 7. 图片
```
markdown_extensions:
  - attr_list
  - md_in_html
```

[yml配置](https://shafish.cn/blog/mkdocs/){ .md-button .md-button--primary}
[图标编码](https://squidfunk.github.io/mkdocs-material/reference/icons-emojis/#mkdocsyml){.md-button .md-button--primary}
[官方插件](https://squidfunk.github.io/mkdocs-material/reference/){.md-button .md-button--primary}

<!-- <hr>
<span id="busuanzi_container_page_pv"><font size="3" color="grey">本文总阅读量<span id="busuanzi_value_page_pv"></span>次</font></span>
<br/> -->