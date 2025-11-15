## Before Commit!

Your commit message must follow Conventional Commits (https://www.conventionalcommits.org/) and your code should be formatted. The Git hooks will do most of the work automatically:

### Tool Requirements

You need a recent `clang-format` (>= 18). In a conda environment you can install:

```shell
conda install -c conda-forge clang-format=18
```

If you previously configured with an older version, remove the build directory and reconfigure:

```shell
rm -rf kt-kernel/build
```

Install `black` for Python formatting:

```shell
conda install black
```

### Install hook:
```shell
bash kt-kernel/scripts/install-git-hooks.sh
#or just cmake the kt-kernel
cmake -S kt-kernel -B kt-kernel/build
```

There are manual commands if you need format.

```shell
cmake -S kt-kernel -B kt-kernel/build
cmake --build kt-kernel/build --target format
```

## Developer Note

Formatting and commit message rules are enforced by Git hooks. After installing `clang-format` and `black`, just commit normally—the hooks will run formatting for you.

> [!NOTE]
> If formatting modifies files, the commit is aborted after staging those changes. Review them and run `git commit` again. Repeat until no further formatting changes appear.

---

### Conventional Commit Regex (Reference)

The commit-msg hook enforces this pattern:

```text
regex='^\[(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert|wip)\](\([^\)]+\))?(!)?: .+'
```

Meaning (English):
* `[type]` required — one of feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert|wip
* Optional scope: `(scope)` — any chars except `)`
* Optional breaking change marker: `!` right after type or scope
* Separator: `: ` (colon + space)
* Subject: free text (at least one character)

Examples:
```text
[feat]: add adaptive batching
[fix(parser)]: handle empty token list
[docs]!: update API section for breaking rename
```

You can bypass locally (not recommended) with:
```shell
git commit --no-verify
```
## 提交前提醒

提交信息必须满足 Conventional Commits 规范 (https://www.conventionalcommits.org/)，代码需要符合格式要求。Git 钩子已经集成了大部分工作：
### 软件要求

需要较新的 `clang-format` (>= 18)，在 conda 环境中安装：

```shell
conda install -c conda-forge clang-format=18
```

如果之前用老版本配置过，请删除构建目录重新配置：

```shell
rm -rf kt-kernel/build
```

安装 `black` 以进行 Python 文件格式化：

```shell
conda install black
```
### 安装钩子
```shell
bash kt-kernel/scripts/install-git-hooks.sh
#or just cmake the kt-kernel
cmake -S kt-kernel -B kt-kernel/build
```
如果你需要手动格式化：
```shell
cmake -S kt-kernel -B kt-kernel/build
cmake --build kt-kernel/build --target format
```

## 开发者说明

本仓库通过 Git hooks 自动执行代码格式化与提交信息规范检查。只需安装好 `clang-format` 与 `black` 后正常执行提交即可，钩子会自动格式化。

> [!NOTE]
> 如果格式化修改了文件，钩子会终止提交并已暂存这些改动。请查看修改后再次执行 `git commit`，重复直到没有新的格式化变更。

### 提交信息正则（参考）

钩子使用如下正则检查提交信息：
```text
regex='^\[(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert|wip)\](\([^\)]+\))?(!)?: .+'
```
含义：
* `[type]` 必填：feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert|wip
* 作用域可选：`(scope)`，不能包含右括号
* 可选的破坏性标记：`!`
* 分隔符：冒号+空格 `: `
* 描述：至少一个字符

示例：
```text
[feat]: 增加自适应 batch 功能
[fix(tokenizer)]: 修复空 token 列表处理
[docs]!: 更新接口文档（存在破坏性修改）
```

跳过钩子（不推荐，仅紧急时）：
```shell
git commit --no-verify
```

