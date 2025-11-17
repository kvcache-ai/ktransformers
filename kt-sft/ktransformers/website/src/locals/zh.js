// zh.js
export default {
    home: {
        explore: '探索',
        language: '选择语言',
        english: '英语',
        chinese: '中文',
        today: '今天',
        previous:'历史',
        withoutAssistantTip:'本记录的KTransformers已被删除，用户只能查看历史对话信息而无法继续对话!',
        deleteThreadTip:'删除记录会清除历史信息哦～'
    },
    chat:{
        inputTip:"发送信息和 KTransformers 畅聊吧～",
    },
    explore:{
        description: "基于Lexllama，一起来创建你的专属KTransformers吧~",
        configuring: "配置中",
        completed: "完成",
        assistantName: "名称",
        assistantDescription: "描述",
        assistantStatus: "Status",
        createAssistant: "创建新的KTransformers",
        deleteAssistant: "是否确认删除KTransformers，删除KTransformers之后其KVCache也会被同步清理掉哦~",
    },
    config:{
        title:'配置你的KTransformers',
        fileTip:"仅支持上传文件格式为 .text, docx, .ppt, .pdf format.",
        secletFile:'选择文件',
        outOfSize:'文件大小超出10MB，请重新选择',
        fileExist:'文件已存在，请重新选择',
        createAssistant:'KTransformers创建成功，点击build按钮开始构建KVCache',
    },
    build:{
        title:'构建日志',
        step1:'解析上传文件',
        parsingFileStep1:'文件上传接收完成',
        parsingFileStep2:{
            parse:"正在解析第",
            file:"文件",
            total:'共',
        },
        parsingFileStep3:'Prompt装载完毕，准备生成KVCache',
        step2:'生成 KVCache',
        generateStep1:'生成KVCache计算计划',
        generateStep2:{
            calculate:"正在计算",
            token:"tokens",
            total:'共',
        },
        generateStep3:'KVCache已生成完成',
        durationTime:'持续时间：',
        remainTime:'剩余时间：',
        buildProgress:'构建进度',
        storageUsage:'存储使用：',
        
    }
}
