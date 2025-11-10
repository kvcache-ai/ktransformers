# 如何使用 Tabby 和 ktransformers 在本地利用 236B 的大模型做代码补全？

[Tabby](https://tabby.tabbyml.com/docs/welcome/) 是一个开源的代码助手，用户可以手动配置后端使用的框架及模型，并在多个 IDE/编辑器 上使用，例如 VSCode 和 InteliJ。因为 Tabby 在框架侧可以对接到 Ollama，并且 ktransformers server 提供和 Ollama 一致的 API 接口，所以我们可以将 Tabby 对接到 ktransformers server。并在代码补全的场景中体验到 ktransformers 快速的异构推理。

1. 启动 ktransformers。
```bash
./ktransformers --port 9112
```
2. 安装 Tabby：按照 Tabby 的官方教程在带有英伟达 GPU 的 Linux 服务器或者 Windows PC 上[安装 Tabby](https://tabby.tabbyml.com/docs/quick-start/installation/linux/)。
3. 配置 Tabby：创建`~/.tabby/config.toml`，并加入以下配置。
```toml
[model.completion.http]
kind = "ollama/completion"
api_endpoint = "http://127.0.0.1:9112/"
model_name = "DeepSeek-Coder-V2-Instruct"
prompt_template = "<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>" # Prompt Template
```

在这个配置中，`kind` 指明 ktransformers 使用 Ollama 的标准 API 为 Tabby 提供服务；`api_endpoint` 与 ktransforer 启动时绑定的接口保持一致；`model_name` 设置为 ktransformers 使用的模型，这里使用 `DeepSeek-Coder-V2-Instruct` 作为后台推理的模型；`prompt_template` 是模型的提示词模板，针对不同的模型，使用相对应的模版才能正常使用模型 Fill In the Middle 的功能。
在这里演示的是 Tabby 使用 Ollama API 提供 Completion 功能的相关配置，有关 Tabby 其他可选功能的配置信息请参照[这里](https://tabby.tabbyml.com/docs/administration/model/)。


4. 启动 Tabby 服务：`./tabby serve`。
<img src="run-tabby.png" alt="image-20240709112329577" style="zoom:50%;" />

​	启动之后，期望会在 ktransformers 的命令行界面看到对 `/api/tags` 接口的访问(在 Tabby 新版本 v0.13.0 中变为对 `/api/show/` 接口的访问)。
<img src="visit-api-tags.png" alt="image-20240709111648215" style="zoom:67%;" />

6. 注册 Tabby 账户，获取 Token：在启动 Tabby 服务后，在浏览器中打开相应的链接(如上图的 0.0.0.0:8080)，并参照[教程](https://tabby.tabbyml.com/docs/quick-start/register-account/) 创建用户并获取 Token。

7. 启动 VScode 安装 Tabby 拓展插件，并在相关提示下，使用上一步获得的 Token 连接 Tabby Server，参照[这里](https://tabby.tabbyml.com/docs/extensions/installation/vscode/)。

8. 打开任意代码文件，体验 ktransformers 的快速异构推理。

