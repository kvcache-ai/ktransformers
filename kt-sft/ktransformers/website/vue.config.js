
module.exports = {
  // 配置 webpack-dev-server 行为。
  devServer: {
    open: false, // 编译后默认打开浏览器
    host: '0.0.0.0',  // 域名
    port: 8082,  // 端口
    https: false,  // 是否https
    proxy: {
        '/api': {
          target: 'http://localhost:9016/v1', // 你的后端服务器地址
          changeOrigin: true, // 是否允许跨域
          pathRewrite: {
            '/api': '' // 将 '/api' 前缀替换为空，如果你的后端不需要这个前缀
          }
        }
      }
},
publicPath: '/web/',  // 基本路径
outputDir: 'dist', // 构建时的输出目录
assetsDir: 'static', // 放置静态资源的目录
indexPath: 'index.html', // html 的输出路径
filenameHashing: true, // 文件名哈希值
lintOnSave: false, // 是否在保存的时候使用 `eslint-loader` 进行检查。

// 组件是如何被渲染到页面中的？ （ast：抽象语法树；vDom：虚拟DOM）
// template ---> ast ---> render ---> vDom ---> 真实的Dom ---> 页面
// runtime-only：将template在打包的时候，就已经编译为render函数
// runtime-compiler：在运行的时候才去编译template
runtimeCompiler: false,

transpileDependencies: [], // babel-loader 默认会跳过 node_modules 依赖。
productionSourceMap: false, // 是否为生产环境构建生成 source map

//调整内部的 webpack 配置
configureWebpack: () => {},

chainWebpack: () => {},
  
}